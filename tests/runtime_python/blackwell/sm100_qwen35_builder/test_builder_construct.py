"""The Qwen3.5 builder constructs the full 40-layer graph, and the two known
silent-corruption footguns fire (M2-I8 acceptance 2/3).

Three phases, each in its own process because `TaskRegister` is process-global:

  `full`     — build all 40 layers at mbt = mbr = 16 on dummy weights of the
               REAL shapes, then run `generate_task_graph()`. That is the step
               that runs `build_annotated_graph` (`runtime.cc:243`), i.e. the
               cycle check, the residual stripping and the case-2/case-3
               fork/join validation — the checks a wrongly-wired MoE fork or a
               mis-shared quantize task trips. nvcc is deliberately NOT run
               here (that is the single-layer test-mode gate's job); this phase
               exists to prove the 40-layer TOPOLOGY is legal and to report its
               task/event budget.
  `mbt_lt_mbr`  — NEGATIVE: mbt < mbr must raise before anything is wired.
  `page_short`  — NEGATIVE: max_num_pages < mbr * ceil(max_seq/page) must raise.

Dummy weights share one routed-expert bank across all 40 layers (the graph only
needs valid pointers and correct shapes), which keeps this phase at ~3 GB
instead of ~33 GB.

Run:  python .../test_builder_construct.py
"""

import os
import subprocess
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(_REPO, "python"))

import mirage                                                    # noqa: E402
from mirage.mpk.models.qwen3_5.builder import Qwen35Builder      # noqa: E402
from mirage.mpk.models.qwen3_5.weight_loader import Qwen35Config  # noqa: E402
from mirage.mpk.persistent_kernel import PersistentKernel        # noqa: E402

BF, F32, FP8 = torch.bfloat16, torch.float32, torch.float8_e4m3fn
B = 128


def real_config() -> Qwen35Config:
    """The shipped Qwen3.5-35B-A3B-FP8 shape, inline so this phase needs no
    checkpoint (`config.json` values, `vllm-graph.md` §1.1)."""
    return Qwen35Config(
        hidden_size=2048, num_layers=40,
        layer_types=["full_attention" if (i + 1) % 4 == 0 else "linear_attention"
                     for i in range(40)],
        vocab_size=248320, rms_norm_eps=1e-6, eos_token_id=248044,
        num_attention_heads=16, num_key_value_heads=2, head_dim=256,
        rotary_dim=64, rope_theta=1e7,
        linear_num_key_heads=16, linear_num_value_heads=32,
        linear_key_head_dim=128, linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        num_experts=256, num_experts_per_tok=8,
        moe_intermediate_size=512, shared_expert_intermediate_size=512,
    )


def dummy_weights(c: Qwen35Config, *, share_expert_bank=True, layers=None):
    dev = "cuda"
    z = lambda s, d: torch.zeros(s, dtype=d, device=dev)          # noqa: E731
    H, E = c.hidden_size, c.num_experts
    inter, si = c.moe_intermediate_size, c.shared_expert_intermediate_size
    w = {
        "embed_tokens": z((c.vocab_size, H), BF),
        "lm_head": z((c.vocab_size, H), BF),
        "model_norm": z((H,), BF),
    }
    shared_w13 = z((E, 2 * inter, H), FP8)
    shared_w13s = torch.ones((E, 2 * inter // B, H // B), dtype=F32, device=dev)
    shared_w2 = z((E, H, inter), FP8)
    shared_w2s = torch.ones((E, H // B, inter // B), dtype=F32, device=dev)
    for i in (layers if layers is not None else range(c.num_layers)):
        w[f"layer_{i}_input_layernorm"] = z((H,), BF)
        w[f"layer_{i}_post_attention_layernorm"] = z((H,), BF)
        w[f"layer_{i}_router"] = z((E, H), BF)
        w[f"layer_{i}_shared_expert_gate"] = z((1, H), BF)
        w[f"layer_{i}_shared_gate_up"] = z((2 * si, H), FP8)
        w[f"layer_{i}_shared_gate_up_scale"] = torch.ones(
            (2 * si // B, H // B), dtype=F32, device=dev)
        w[f"layer_{i}_shared_down"] = z((H, si), FP8)
        w[f"layer_{i}_shared_down_scale"] = torch.ones((H // B, si // B),
                                                       dtype=F32, device=dev)
        if share_expert_bank:
            w[f"layer_{i}_w13"], w[f"layer_{i}_w13_scale"] = shared_w13, shared_w13s
            w[f"layer_{i}_w2"], w[f"layer_{i}_w2_scale"] = shared_w2, shared_w2s
        else:
            w[f"layer_{i}_w13"] = z((E, 2 * inter, H), FP8)
            w[f"layer_{i}_w13_scale"] = torch.ones((E, 2 * inter // B, H // B),
                                                   dtype=F32, device=dev)
            w[f"layer_{i}_w2"] = z((E, H, inter), FP8)
            w[f"layer_{i}_w2_scale"] = torch.ones((E, H // B, inter // B),
                                                  dtype=F32, device=dev)
        if c.layer_types[i] == "linear_attention":
            w[f"layer_{i}_gdn_in_proj_qkv"] = z((c.conv_dim, H), FP8)
            w[f"layer_{i}_gdn_in_proj_qkv_scale"] = torch.ones(
                (c.conv_dim // B, H // B), dtype=F32, device=dev)
            w[f"layer_{i}_gdn_in_proj_z"] = z((c.gdn_z_dim, H), FP8)
            w[f"layer_{i}_gdn_in_proj_z_scale"] = torch.ones(
                (c.gdn_z_dim // B, H // B), dtype=F32, device=dev)
            w[f"layer_{i}_gdn_in_proj_ba"] = z((2 * c.linear_num_value_heads, H), BF)
            w[f"layer_{i}_gdn_conv1d"] = z((c.conv_dim, c.linear_conv_kernel_dim), BF)
            w[f"layer_{i}_gdn_alog_dtbias"] = z((2, c.linear_num_value_heads), F32)
            w[f"layer_{i}_gdn_norm"] = torch.ones((c.linear_value_head_dim,),
                                                  dtype=F32, device=dev)
            w[f"layer_{i}_gdn_out_proj"] = z((H, c.gdn_z_dim), FP8)
            w[f"layer_{i}_gdn_out_proj_scale"] = torch.ones(
                (H // B, c.gdn_z_dim // B), dtype=F32, device=dev)
        else:
            w[f"layer_{i}_qkvg_proj"] = z((c.qkvg_dim, H), FP8)
            w[f"layer_{i}_qkvg_proj_scale"] = torch.ones(
                (c.qkvg_dim // B, H // B), dtype=F32, device=dev)
            w[f"layer_{i}_q_norm"] = z((c.head_dim,), BF)
            w[f"layer_{i}_k_norm"] = z((c.head_dim,), BF)
            w[f"layer_{i}_o_proj"] = z((H, c.num_attention_heads * c.head_dim), FP8)
            w[f"layer_{i}_o_proj_scale"] = torch.ones(
                (H // B, c.num_attention_heads * c.head_dim // B), dtype=F32,
                device=dev)
    return w


def make_pk(*, mbt, mbr, pages, page_size, max_seq):
    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params.update(
        test_mode=True, num_workers=num_workers,
        num_local_schedulers=num_schedulers, mpi_rank=0, world_size=1,
        max_num_batched_tokens=mbt, max_num_batched_requests=mbr,
        max_num_pages=pages, page_size=page_size, max_seq_length=max_seq,
        meta_tensors={
            "tokens": torch.zeros((mbr, max_seq), dtype=torch.int64, device="cuda"),
            "prompt_lengths": torch.full((mbr,), 4, dtype=torch.int32, device="cuda"),
            # production shapes these [mbt, 1] (demo/qwen3/demo.py:261-262);
            # test mode's auto-default is 1-D, which argmax_reduce_layer rejects
            "input_tokens": torch.zeros((mbt, 1), dtype=torch.int64, device="cuda"),
            "output_tokens": torch.zeros((mbt, 1), dtype=torch.int64, device="cuda"),
        },
    )
    return PersistentKernel(**params)


def phase_full():
    c = real_config()
    pk = make_pk(mbt=16, mbr=16, pages=16, page_size=1280, max_seq=1280)
    assert pk.target_cc >= 100, "Qwen3.5 is Blackwell-only"
    b = Qwen35Builder(pk)
    b.build_from_weights(dummy_weights(c), c)
    res = pk.kn_graph.generate_task_graph(num_gpus=1, my_gpu_id=0)
    # `json_file` is the serialized graph itself, not a path (runtime.cc's
    # print_task_graph returns the text; compile() is what writes it out).
    import json as _json
    tg = _json.loads(res["json_file"])
    print(f"  RESULT layers=40 mbt=16 mbr=16 tasks={len(tg['all_tasks'])} "
          f"events={len(tg['all_events'])} first_tasks={len(tg['first_tasks'])}")
    print(f"  RESULT vocab padded to {b.padded_vocab_size} (no padding needed)")
    assert len(tg["all_tasks"]) > 0
    return 0


def _expect_assert(fn, needle):
    try:
        fn()
    except AssertionError as e:
        if needle in str(e):
            print(f"  RESULT fired: ...{needle}...")
            return 0
        print(f"  RESULT WRONG assertion: {e}")
        return 1
    print("  RESULT NO assertion raised - the footgun is unguarded")
    return 1


def phase_mbt_lt_mbr():
    c = real_config()
    pk = make_pk(mbt=8, mbr=16, pages=16, page_size=1280, max_seq=1280)
    return _expect_assert(
        lambda: Qwen35Builder(pk).build_from_weights(
            dummy_weights(c, layers=[0]), c),
        "max_num_batched_tokens")


def phase_page_short():
    # mbr 16 x ceil(1280/256) = 80 pages needed, only 16 provided.
    c = real_config()
    pk = make_pk(mbt=16, mbr=16, pages=16, page_size=256, max_seq=1280)
    return _expect_assert(
        lambda: Qwen35Builder(pk).build_from_weights(
            dummy_weights(c, layers=[0]), c),
        "max_num_pages")


PHASES = {"full": phase_full, "mbt_lt_mbr": phase_mbt_lt_mbr,
          "page_short": phase_page_short}
TITLES = {
    "full": "40-layer graph constructs and passes build_annotated_graph",
    "mbt_lt_mbr": "NEGATIVE: mbt < mbr must assert (stalled-request footgun)",
    "page_short": "NEGATIVE: short page pool must assert (KV-corruption footgun)",
}


def drive():
    failures = []
    for name in ("full", "mbt_lt_mbr", "page_short"):
        print(f"\n== {TITLES[name]} ==", flush=True)
        p = subprocess.run([sys.executable, os.path.abspath(__file__),
                            "--phase", name],
                           env=dict(os.environ, PYTHONUNBUFFERED="1"),
                           capture_output=True, text=True)
        for line in p.stdout.splitlines():
            if line.startswith("  RESULT"):
                print(line, flush=True)
        if p.returncode != 0:
            failures.append(f"phase {name} failed (rc={p.returncode})")
            print("  --- tail ---\n" +
                  "\n".join((p.stdout + p.stderr).splitlines()[-15:]), flush=True)
    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(" -", f)
        return 1
    print("\nBUILDER CONSTRUCTION + FOOTGUN TESTS PASSED")
    return 0


if __name__ == "__main__":
    if "--phase" in sys.argv:
        sys.exit(PHASES[sys.argv[sys.argv.index("--phase") + 1]]())
    sys.exit(drive())
