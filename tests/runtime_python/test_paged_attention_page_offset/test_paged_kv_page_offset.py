# Regression test for #796: a request whose page table does not start at
# paged_kv_indices[0] must load its own pages. The same physical pages are
# addressed once as request 0 (first_page_pos = 0) and once as request 1
# behind another request's pages (first_page_pos = 3, not 16-byte aligned);
# the attention output must be identical.
#
#   cd tests/runtime_python/test_paged_attention_page_offset
#   pip install --no-build-isolation . && python test_paged_kv_page_offset.py
import torch
import paged_attention_page_offset

qo_heads, kv_heads, head_dim, page_size = 4, 1, 128, 64
max_num_pages = 64
num_tokens = 8
device = "cuda"
dtype = torch.bfloat16

torch.manual_seed(0)
k_cache = torch.randn((max_num_pages, page_size, kv_heads * head_dim), device=device, dtype=dtype)
v_cache = torch.randn((max_num_pages, page_size, kv_heads * head_dim), device=device, dtype=dtype)
qkv = torch.randn((num_tokens, (qo_heads + 2 * kv_heads) * head_dim), device=device, dtype=dtype)

pages = [7, 2, 11, 5, 9]  # 5 pages: enough to hit the bulk (4 pages at a time) copy
other_pages = [30, 31, 32]  # pages owned by the preceding request
last_page_len = 40


def run(request_id, qo_indptr, kv_indptr, kv_indices, last_page_lens):
    i32 = lambda x: torch.tensor(x, device=device, dtype=torch.int32)
    out = torch.zeros(num_tokens * qo_heads, head_dim, device=device, dtype=dtype)
    paged_attention_page_offset.paged_attention_4_16(
        qkv.clone(),
        k_cache.clone(),
        v_cache.clone(),
        out,
        i32(qo_indptr),
        i32(kv_indptr),
        i32(kv_indices),
        i32(last_page_lens),
        request_id,
    )
    return out


ref = run(0, [0, num_tokens], [0, len(pages)], pages, [last_page_len])
out = run(
    1,
    [0, 0, num_tokens],
    [0, len(other_pages), len(other_pages) + len(pages)],
    other_pages + pages,
    [page_size, last_page_len],
)

assert ref.abs().sum() > 0 and not torch.isnan(ref).any()
assert torch.equal(out, ref), f"max abs diff {(out.float() - ref.float()).abs().max().item()}"
print("PASS")
