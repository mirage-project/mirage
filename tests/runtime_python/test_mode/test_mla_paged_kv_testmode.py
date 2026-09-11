"""DeepSeek-V3 MLA gather against the unified KV page pool: append + read
back across a page boundary, correct slot, correct page-stride addressing."""

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

NUM_LAYERS = 4     # small stand-in for 61; only >1 slot matters
NUM_MTP = 1
PAGE_SIZE = 64
MAX_NUM_PAGES = 8
MAX_SEQ_LENGTH = 256
PROMPT_LEN = 100   # spans 2 pages at PAGE_SIZE=64, with a partial last page
MAX_BATCHED_TOKENS = 128

LAYERS_UNDER_TEST = (1, NUM_LAYERS)  # one ordinary layer + the MTP slot


class _Config:
    num_hidden_layers = NUM_LAYERS
    num_nextn_predict_layers = NUM_MTP


def main():
    torch.manual_seed(0)
    device = "cuda"

    plan = build_kv_cache(
        kv_streams(_Config(), world_size=1),
        block_size=PAGE_SIZE,
        max_num_pages=MAX_NUM_PAGES,
        max_seq_length=MAX_SEQ_LENGTH,
        max_num_batched_requests=1,
        max_num_batched_tokens=MAX_BATCHED_TOKENS,
        verbose=False)

    ok = True

    # attention and MTP are declared as two streams; they must merge into one
    # group or MTP would cost 61x the scheduler work.
    if len(plan.groups) != 1:
        print(f"FAILED: {len(plan.groups)} groups, expected 1: "
              f"{[(g.group_id, g.spec_name) for g in plan.groups]}")
        ok = False
    group = plan.groups[0]
    if len(group.layer_ids) != NUM_LAYERS + NUM_MTP:
        print(f"FAILED: group holds {len(group.layer_ids)} layers, expected "
              f"{NUM_LAYERS + NUM_MTP}")
        ok = False

    view = plan.views(0)["kv"]
    slot_of = {layer: plan._layer_info(layer)[1] for layer in LAYERS_UNDER_TEST}
    print(f"pool {tuple(plan._pool.shape)}  view {tuple(view.shape)}  "
          f"stride {view.stride()}  slots {slot_of}")

    layer_view = view[slot_of[LAYERS_UNDER_TEST[0]]]
    want_stride = (PAGE_SIZE * D_K, D_K, 1)
    if tuple(layer_view.shape) != (plan.max_num_pages, PAGE_SIZE, D_K):
        print(f"FAILED: layer view is {tuple(layer_view.shape)}, expected "
              f"{(plan.max_num_pages, PAGE_SIZE, D_K)}")
        ok = False
    if layer_view.stride() != want_stride:
        print(f"FAILED: layer view stride {layer_view.stride()}, expected "
              f"{want_stride} -- the kernel indexes page_idx*PAGE_STRIDE + "
              f"pos, so a wrong stride silently reads a neighbouring page")
        ok = False

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
        c_latent = torch.randn(MAX_BATCHED_TOKENS, D_V, dtype=torch.bfloat16,
                               device=device)
        # k_pe rows are padded to K_PE_ROW_STRIDE for SM100 MMA_M alignment;
        # the real rope dim is the first ROPE_DIM columns.
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

    # test mode retires the request at graph end, which collapses the page
    # table -- read allocated pages from the event log instead.
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
        print("FAILED: the sequence fits in one page, page stride never "
              "exercised -- raise PROMPT_LEN")
        ok = False

    for layer, c_latent, k_pe, contiguous_kv in cases:
        expect = torch.cat([c_latent[:seq_len].float(),
                            k_pe[:seq_len, :ROPE_DIM].float()], dim=1)

        got = contiguous_kv[:seq_len].float()
        diff = (got - expect).abs().max().item()
        print(f"[layer {layer}] max |gathered - input| = {diff:.4f}")
        if diff != 0.0:
            print(f"[layer {layer}] FAILED: gather did not read back the "
                  f"append")
            ok = False

        slot = slot_of[layer]
        for pos in range(seq_len):
            page_id = page_ids[pos // PAGE_SIZE]
            row = view[slot, page_id, pos % PAGE_SIZE].float()
            if not torch.equal(row, expect[pos]):
                print(f"[layer {layer}] FAILED: row {pos} not at pool"
                      f"[slot {slot}, page {page_id}, {pos % PAGE_SIZE}]")
                ok = False
                break

    # exactly the slots under test hold exactly seq_len rows, no others.
    touched = {slot_of[layer] for layer in LAYERS_UNDER_TEST}
    written = [int((view[s].abs().sum(dim=-1) > 0).sum().item())
               for s in range(plan.num_slots)]
    print(f"rows written per slot: {written} (slots under test: "
          f"{sorted(touched)})")
    for slot, count in enumerate(written):
        expected = seq_len if slot in touched else 0
        if count != expected:
            print(f"FAILED: slot {slot} holds {count} rows, expected "
                  f"{expected}")
            ok = False

    pk.finalize()
    if not ok:
        sys.exit(1)
    print("\nPASSED")


if __name__ == "__main__":
    main()
