"""Test-mode coverage for DeepSeek-V3's MLA cache on the KV 2.0 page pool.

DeepSeek-V3 is not runnable here (671B, and TP is blocked on NVSHMEM), so the
thing that CAN be checked is the part the migration actually changed: whether
the cache the planner hands a layer is addressed by the MLA kernels exactly the
way the hand-rolled `[num_layers, pages, page_size, 576]` tensor was.

`mla_kv_gather` is the whole KV path in one task -- it appends the new tokens
to the paged cache and gathers the sequence back out of it -- so running it
against a plan-backed view exercises the page table, the page stride and the
slot offset together, with a reference that is just the input rows.

What each check is for:
  * gathered == input                 the page table walk reads back what the
                                      append wrote, across a page boundary
  * pool rows at [slot, page, pos]    the append landed at the address the
                                      layout says, not merely somewhere
  * every other slot still zero       a layer cannot write into another
                                      layer's slot (the slot offset is right)
  * MTP is slot 61, not a 2nd tensor  the merged group gives the predictor a
                                      slot on the same pages
"""

import os
import sys

import torch

import mirage
from mirage.mpk.kv_planner import build_kv_cache
from mirage.mpk.models.deepseek_v3.builder import kv_streams
from mirage.mpk.persistent_kernel import PersistentKernel

D_K = 576          # 512 latent + 64 rope, fixed by the absorbed MLA layout
D_V = 512
ROPE_DIM = D_K - D_V
K_PE_ROW_STRIDE = 128   # builder pads k_pe rows to 128; task_register agrees

NUM_LAYERS = 4     # a small stand-in for 61; what matters is >1 slot
NUM_MTP = 1
PAGE_SIZE = 64
MAX_NUM_PAGES = 8
MAX_SEQ_LENGTH = 256
PROMPT_LEN = 100   # spans 2 pages at PAGE_SIZE=64, with a partial last page
MAX_BATCHED_TOKENS = 128

# The layers gathered in this graph: one ordinary layer and the MTP slot.
LAYERS_UNDER_TEST = (1, NUM_LAYERS)


class _Config:
    num_hidden_layers = NUM_LAYERS
    num_nextn_predict_layers = NUM_MTP


def main():
    torch.manual_seed(0)
    device = "cuda"

    plan = build_kv_cache(
        kv_streams(_Config(), PAGE_SIZE, world_size=1),
        max_num_pages=MAX_NUM_PAGES,
        max_seq_length=MAX_SEQ_LENGTH,
        max_num_batched_requests=1,
        max_num_batched_tokens=MAX_BATCHED_TOKENS,
        verbose=False)

    ok = True

    # ── Layout, before anything runs ──────────────────────────────────────
    # The declaration is two streams (attention and MTP are different
    # modules); the planner must fold them, or the pool carries one slot per
    # group and MTP costs 61x the scheduler work.
    if len(plan.groups) != 1:
        print(f"FAILED: {len(plan.groups)} groups, expected the MLA and MTP "
              f"streams to merge into 1: "
              f"{[(g.group_id, g.spec_name) for g in plan.groups]}")
        ok = False
    group = plan.groups[0]
    if len(group.layer_ids) != NUM_LAYERS + NUM_MTP:
        print(f"FAILED: group holds {len(group.layer_ids)} layers, expected "
              f"{NUM_LAYERS + NUM_MTP} (the MTP layer is layer {NUM_LAYERS})")
        ok = False

    view = plan.views(0)["kv"]
    slot_of = {layer: plan._layer_info(layer)[1] for layer in LAYERS_UNDER_TEST}
    print(f"pool {tuple(plan._pool.shape)}  view {tuple(view.shape)}  "
          f"stride {view.stride()}  slots {slot_of}")

    # A layer's own view is [pages, block, 576] strided by exactly one page --
    # the same shape and stride the hand-rolled ckv_kpe_cache[i] had. The pool
    # is slot-major, so a layer's pages stay contiguous; what the pool adds is
    # that the pages are shared, not that they are further apart.
    layer_view = view[slot_of[LAYERS_UNDER_TEST[0]]]
    want_stride = (PAGE_SIZE * D_K, D_K, 1)
    if tuple(layer_view.shape) != (plan.max_num_pages, PAGE_SIZE, D_K):
        print(f"FAILED: layer view is {tuple(layer_view.shape)}, expected "
              f"{(plan.max_num_pages, PAGE_SIZE, D_K)}")
        ok = False
    if layer_view.stride() != want_stride:
        print(f"FAILED: layer view stride {layer_view.stride()}, expected "
              f"{want_stride} -- the MLA kernels index "
              f"page_idx*PAGE_STRIDE + pos, so a page stride that is not the "
              f"block size silently reads a neighbouring page")
        ok = False

    # ── Build a graph with one gather per layer under test ────────────────
    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params.update(
        test_mode=True,
        num_workers=num_workers,
        num_local_schedulers=num_schedulers,
        max_seq_length=MAX_SEQ_LENGTH,
        max_num_batched_requests=1,
        max_num_batched_tokens=MAX_BATCHED_TOKENS,
        max_num_pages=plan.max_num_pages,
        kv_groups=plan.group_specs(),
    )
    params["meta_tensors"] = {
        "prompt_lengths": torch.tensor([PROMPT_LEN], dtype=torch.int32,
                                       device=device),
        **plan.build_meta_tensors(max_num_batched_requests=1,
                                  max_seq_length=MAX_SEQ_LENGTH),
    }
    pk = PersistentKernel(**params)
    pk.kv_plan = plan
    from mirage.mpk.kv_planner import KVEventLog
    event_log = KVEventLog(pk, plan)

    cases = []
    for layer in LAYERS_UNDER_TEST:
        tag = f"L{layer}"
        # Distinct data per layer, so a cross-slot write shows up as the wrong
        # values rather than as a coincidence.
        c_latent = torch.randn(MAX_BATCHED_TOKENS, D_V, dtype=torch.bfloat16,
                               device=device)
        # k_pe rows are padded to K_PE_ROW_STRIDE for SM100 MMA_M alignment;
        # the real rope dim is the first ROPE_DIM columns. The gather task
        # hardcodes that stride, so the test has to use the same ABI.
        k_pe = torch.randn(MAX_BATCHED_TOKENS, K_PE_ROW_STRIDE,
                           dtype=torch.bfloat16, device=device)
        contiguous_kv = torch.zeros(MAX_SEQ_LENGTH, D_K, dtype=torch.bfloat16,
                                    device=device)
        kv = plan.attach(pk, layer)
        pk.mla_kv_gather_layer(
            c_latent_new=pk.attach_input(c_latent, name=f"{tag}_c_latent"),
            k_pe_new=pk.attach_input(k_pe, name=f"{tag}_k_pe"),
            paged_cache=kv["kv_cache"],
            contiguous_kv=pk.attach_input(contiguous_kv, name=f"{tag}_kv"),
            mla_params=(D_K, D_V),
            group_id=kv["group_id"],
            grid_dim=(1, 1, 1),
            block_dim=(128, 1, 1),
        )
        cases.append((layer, c_latent, k_pe, contiguous_kv))

    print("Compiling test kernel...")
    pk.compile(output_dir=os.path.dirname(os.path.abspath(__file__)))
    print("Running test kernel...")
    pk()
    torch.cuda.synchronize()

    # ── Check ─────────────────────────────────────────────────────────────
    # The page table itself cannot be read back here: test mode retires the
    # request at the end of the graph, and retirement returns its pages and
    # collapses the indptr. The allocator's event log survives that, so the
    # page ids are OBSERVED rather than assumed to be 0, 1, ...
    events = event_log.log.cpu().tolist()
    page_ids = [events[4 * i + 4] for i in range(events[0])
                if events[4 * i + 1] == 1]
    seq_len = PROMPT_LEN
    want_pages = (PROMPT_LEN + PAGE_SIZE - 1) // PAGE_SIZE
    print(f"pages allocated for the request: {page_ids} "
          f"(seq_len {seq_len} over {want_pages} page(s))")
    if len(page_ids) != want_pages:
        print(f"FAILED: {len(page_ids)} page(s) allocated, expected "
              f"{want_pages}")
        ok = False
    if want_pages < 2:
        print(f"FAILED: the sequence fits in one page, so the page stride is "
              f"never exercised -- raise PROMPT_LEN")
        ok = False

    for layer, c_latent, k_pe, contiguous_kv in cases:
        expect = torch.cat([c_latent[:seq_len].float(),
                            k_pe[:seq_len, :ROPE_DIM].float()], dim=1)

        got = contiguous_kv[:seq_len].float()
        diff = (got - expect).abs().max().item()
        print(f"[layer {layer}] max |gathered - input| = {diff:.4f}")
        if diff != 0.0:
            print(f"[layer {layer}] FAILED: the gather did not read back what "
                  f"the append wrote")
            ok = False

        # The append must have landed at [slot, page_id, pos_in_page].
        slot = slot_of[layer]
        for pos in range(seq_len):
            page_id = page_ids[pos // PAGE_SIZE]
            row = view[slot, page_id, pos % PAGE_SIZE].float()
            if not torch.equal(row, expect[pos]):
                print(f"[layer {layer}] FAILED: row {pos} is not at pool"
                      f"[slot {slot}, page {page_id}, {pos % PAGE_SIZE}]")
                match = [j for j in range(seq_len)
                         if torch.equal(row, expect[j])]
                lat = [j for j in range(seq_len)
                       if torch.equal(row[:D_V], expect[j][:D_V])]
                pe = [j for j in range(seq_len)
                      if torch.equal(row[D_V:], expect[j][D_V:])]
                print(f"    DIAG whole row matches input rows {match}; "
                      f"latent half {lat}; rope half {pe}")
                ok = False
                break

    # Exactly the slots under test hold exactly seq_len rows, and no others.
    touched = {slot_of[layer] for layer in LAYERS_UNDER_TEST}
    written = [int((view[s].abs().sum(dim=-1) > 0).sum().item())
               for s in range(plan.num_slots)]
    print(f"rows written per slot: {written} (slots under test: "
          f"{sorted(touched)})")
    for slot, count in enumerate(written):
        expected = seq_len if slot in touched else 0
        if count != expected:
            print(f"FAILED: slot {slot} holds {count} rows, expected "
                  f"{expected} -- a layer wrote outside its own slot")
            ok = False

    pk.finalize()
    if not ok:
        sys.exit(1)
    print("\nPASSED: the MLA gather appends to and reads back from the "
          "plan-backed page pool, across a page boundary.")


if __name__ == "__main__":
    main()
