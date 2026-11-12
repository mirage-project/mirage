"""A decode step's glue in one launch: the argmax of the last step's logits (torch's: the first index of the maximum), the
generated token's slot, the step's token (the prompt's, else that argmax), its embedding row, and the page table for the
position. Replaces argmax + index_select + copies + fills (~28 us of small launches a step on H100)."""
import torch
import triton
import triton.language as tl

BLOCK = 4096


@triton.jit
def _glue(logits, V, key, count, nblk, prompt, P, pos, has_prev, gen, embed, x0, H, kv_indptr, last_page_len,
          PAGE: tl.constexpr, BLOCK: tl.constexpr, HBLOCK: tl.constexpr):
    b = tl.program_id(0)
    if has_prev:
        offs = b * BLOCK + tl.arange(0, BLOCK)
        v = tl.load(logits + offs, mask=offs < V, other=float("-inf"))
        v = v + 0.0  # -0 as +0 (torch compares the values)
        m = tl.max(v, axis=0)
        i = tl.min(tl.where(v == m, offs, V), axis=0)  # the first index of the block's maximum
        bits = m.to(tl.int32, bitcast=True)
        # an order-preserving signed key of the value, then the lower index on a tie
        hi = tl.where(bits >= 0, bits, bits ^ 0x7FFFFFFF)
        k = (hi.to(tl.int64) << 32) | (4294967295 - i.to(tl.int64))
        tl.atomic_max(key, k, sem="acq_rel", scope="gpu")
    last = tl.atomic_add(count, 1, sem="acq_rel", scope="gpu") == nblk - 1
    if last:
        tok = tl.load(prompt + tl.minimum(pos, P - 1)).to(tl.int64)
        if has_prev:
            k = tl.atomic_max(key, -9223372036854775807 - 1, sem="acq_rel", scope="gpu")
            nxt = 4294967295 - (k & 4294967295)
            if pos >= P:
                tl.store(gen + (pos - P), nxt)
                tok = nxt
            tl.store(key, -9223372036854775807 - 1)
        tl.store(count, 0)
        h = tl.arange(0, HBLOCK)
        row = tl.load(embed + tok * H + h, mask=h < H)
        tl.store(x0 + h, row, mask=h < H)
        n = pos // PAGE + 1
        tl.store(kv_indptr + 1, n)
        tl.store(last_page_len, pos + 1 - (n - 1) * PAGE)


class FusedGlue:
    def __init__(self, logits_row, embed, x0_row, kv_indptr, last_page_len, prompt, gen, page):
        assert kv_indptr.numel() == 2 and last_page_len.numel() == 1, "one request"
        self.logits, self.embed, self.x0 = logits_row, embed, x0_row
        self.kv_indptr, self.last_page_len, self.prompt, self.gen, self.page = kv_indptr, last_page_len, prompt, gen, page
        self.V, self.H = logits_row.numel(), embed.shape[1]
        self.nblk = triton.cdiv(self.V, BLOCK)
        self.key = torch.full((1,), -2**63, dtype=torch.int64, device="cuda")
        self.count = torch.zeros(1, dtype=torch.int32, device="cuda")
        self.P = prompt.numel()

    def __call__(self, pos: int, has_prev: bool):
        """Before step pos (has_prev: the last step's logits are there); pos == P + NEW - 1: the last token only."""
        _glue[(self.nblk if has_prev else 1,)](
            self.logits, self.V, self.key, self.count, self.nblk if has_prev else 1, self.prompt, self.P, pos,
            has_prev, self.gen, self.embed, self.x0, self.H, self.kv_indptr, self.last_page_len,
            PAGE=self.page, BLOCK=BLOCK, HBLOCK=triton.next_power_of_2(self.H), num_warps=8)
