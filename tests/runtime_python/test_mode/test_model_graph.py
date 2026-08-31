"""The model as a graph, partitioned, lowered to MPK tasks.

Today task boundaries are decided in Python: someone writes
mpk.generated_linear_layer(...) then mpk.generated_silu_mul_layer(...) and that
IS the partition. node.py holds the same computation with the boundary
still open -- plain SSA nodes -- and lower.py turns a chosen grouping into
tasks via task_search.

These tests pin the two halves: that a grouping derives the right task
boundary (pure Python, no GPU), and that lowering one really does compute the
right thing (on GPU, against torch).
"""
import subprocess
import sys
import textwrap

import pytest
import torch

from mirage.mpk.lowering import (ModelGraph, make_group,
                                     group_to_taskspec_build)
from mirage.mpk.lowering import check_covers, default_grid

T, H, I = 8, 1024, 3072


def _qwen3_mlp():
    """silu(x @ Wg) * (x @ Wu) @ Wd -- Qwen3-0.6B shapes at batch 8."""
    g = ModelGraph("qwen3_mlp")
    x = g.new_input((T, H), "x", role="feed")
    wg = g.new_input((H, I), "w_gate")
    wu = g.new_input((H, I), "w_up")
    wd = g.new_input((I, H), "w_down")
    with g.scope(layer=0, tag="mlp"):
        gate = g.matmul(x, wg)
        up = g.matmul(x, wu)
        act = g.mul(g.silu(gate), up)
        out = g.matmul(act, wd)
    g.mark_output(out)
    return g


# ------------------------------------------------------------ the IR

def test_graph_shapes_follow_the_ops():
    g = _qwen3_mlp()
    assert len(g) == 5
    assert [n.op for n in g.nodes] == ["matmul", "matmul", "silu", "mul", "matmul"]
    assert g.nodes[0].output.dims == (T, I)     # gate
    assert g.nodes[4].output.dims == (T, H)     # down
    assert all(n.layer == 0 for n in g.nodes), "scope() must tag the layer"


def test_grouping_derives_the_task_boundary():
    """A group's inputs and output fall out of the node set -- nobody declares
    them. These are exactly today's MPK_COMPILED_MLP_IMPL=separate tasks."""
    g = _qwen3_mlp()
    expected = {
        "gate":     ([0],    ["x", "w_gate"],       (T, I)),
        "up":       ([1],    ["x", "w_up"],         (T, I)),
        "silu_mul": ([2, 3], ["mlp.v1", "mlp.v2"],  (T, I)),
        "down":     ([4],    ["mlp.v4", "w_down"],  (T, H)),
    }
    for tag, (ids, ins, dims) in expected.items():
        grp = make_group(g, ids, tag)
        assert [v.name for v in grp.external_inputs] == ins, tag
        assert grp.output.dims == dims, tag


def test_a_group_with_two_live_outputs_is_rejected():
    """MPK tasks may write several tensors -- generated_gate_up_layer does --
    but task_search only replays a schedule with exactly one TB_OUTPUT_OP, so
    a two-output group is not lowerable through this path. Catch it where the
    message can say why, not inside the transpiler."""
    g = _qwen3_mlp()
    with pytest.raises(ValueError, match="2 live outputs"):
        make_group(g, [0, 1], "gate+up")


def test_partition_must_cover_the_graph():
    """A leftover plain op is not a soft failure: build_annotated_graph skips
    anything that is not KN_CUSTOMIZED_OP and print_task_graph then asserts, so
    MPK produces zero layers and aborts."""
    g = _qwen3_mlp()
    full = [make_group(g, [0], "gate"), make_group(g, [1], "up"),
            make_group(g, [2, 3], "silu_mul"), make_group(g, [4], "down")]
    check_covers(g, full)

    with pytest.raises(ValueError, match="does not cover"):
        check_covers(g, full[:-1])
    with pytest.raises(ValueError, match="is in both"):
        check_covers(g, full + [make_group(g, [4], "down_again")])


def test_replay_reconstructs_the_ops():
    """The build lambda handed to TaskSpec must replay the group in order."""
    g = _qwen3_mlp()
    grp = make_group(g, [2, 3], "silu_mul")
    calls = []

    class FakeKN:
        def silu(self, a):
            calls.append(("silu", a)); return "silu_out"

        def mul(self, a, b):
            calls.append(("mul", a, b)); return "mul_out"

    out = group_to_taskspec_build(g, grp)(FakeKN(), ["GATE", "UP"])
    assert calls == [("silu", "GATE"), ("mul", "silu_out", "UP")]
    assert out == "mul_out"


def test_default_grid_is_one_64_wide_output_column():
    g = _qwen3_mlp()
    assert default_grid(make_group(g, [0], "gate")) == (I // 64, 1, 1)
    assert default_grid(make_group(g, [4], "down")) == (H // 64, 1, 1)


# ------------------------------------------------------------ on GPU

def _skip_reason():
    if not torch.cuda.is_available():
        return "CUDA is not available"
    major, minor = torch.cuda.get_device_capability()
    if major * 10 + minor < 100:
        return "generated task bodies are only emitted for the sm_100 backend"
    return None


_MLP_SRC = textwrap.dedent(
    """
    import sys, torch, mirage
    from mirage.mpk.persistent_kernel import PersistentKernel
    from mirage.mpk.lowering import ModelGraph, make_group
    from mirage.mpk.lowering import lower

    T, H, I = 8, 1024, 3072
    torch.manual_seed(0)
    x  = torch.randn(T, H, dtype=torch.bfloat16, device="cuda") * 0.05
    wg = torch.randn(H, I, dtype=torch.bfloat16, device="cuda") * 0.05
    wu = torch.randn(H, I, dtype=torch.bfloat16, device="cuda") * 0.05
    wd = torch.randn(I, H, dtype=torch.bfloat16, device="cuda") * 0.05
    o  = torch.zeros(T, H, dtype=torch.bfloat16, device="cuda")

    g = ModelGraph()
    vx  = g.new_input((T, H), "x", role="feed")
    vwg = g.new_input((H, I), "w_gate")
    vwu = g.new_input((H, I), "w_up")
    vwd = g.new_input((I, H), "w_down")
    with g.scope(layer=0, tag="mlp"):
        gate = g.matmul(vx, vwg); up = g.matmul(vx, vwu)
        act  = g.mul(g.silu(gate), up)
        out  = g.matmul(act, vwd)
    g.mark_output(out)

    partition = __PARTITION__

    nw, ns = mirage.get_configurations_from_gpu(0)
    p = PersistentKernel.get_default_init_parameters()
    p.update(test_mode=True, num_workers=nw, num_local_schedulers=ns,
             mpi_rank=0, world_size=1, max_num_batched_tokens=T,
             max_num_batched_requests=T)
    pk = PersistentKernel(**p)
    bind = {"x": pk.attach_input(x, name="x"),
            "w_gate": pk.attach_input(wg, name="wg"),
            "w_up": pk.attach_input(wu, name="wu"),
            "w_down": pk.attach_input(wd, name="wd")}
    od = pk.attach_input(o, name="o")

    # K=3072 is not a power of two and search never fuses such a matmul, so the
    # down projection falls back to the hand-written schedule -- see lower.py.
    def down_fallback(pk, ins, out, grid):
        pk.generated_linear_layer(input=ins[0], weight_t=ins[1], output=out,
                                  grid_dim=grid, block_dim=(256, 1, 1),
                                  forloop_range=ins[0].dim(1) // 64)

    lower(pk, g, partition, bind, outputs={out.name: od},
          fallbacks={"down": down_fallback}, verbose=True)
    pk.compile(output_dir=None)
    pk(); torch.cuda.synchronize()

    gate_r = (x.float() @ wg.float()).to(torch.bfloat16).float()
    up_r   = (x.float() @ wu.float()).to(torch.bfloat16).float()
    act_r  = (torch.nn.functional.silu(gate_r) * up_r).to(torch.bfloat16).float()
    ref    = act_r @ wd.float()
    rel = ((o.float() - ref).abs().max() / ref.abs().max()).item()
    print("REL", rel, flush=True)
    sys.exit(0 if rel < 0.02 else 1)
    """
)


# The two MLP shapes that matter, as the partition literal _MLP_SRC needs.
# `separate` is today's boundaries, so a failure there is the graph or the
# lowering. `fused` is a gate+up+SwiGLU group, whose task is SEARCHED -- its correctness rests on search's own
# equivalence check, which has false-accepted before (one draw in six once
# dropped an exp and still verified). Measuring it against torch is the only
# thing that says the winning partition computes the model.
_MLP_PARTITIONS = {
    "separate": ('[make_group(g, [0], "linear"), make_group(g, [1], "linear"),'
                 ' make_group(g, [2, 3], "silu_mul"), make_group(g, [4], "down")]'),
    "fused": ('[make_group(g, [0, 1, 2, 3], "matmul_matmul_silu_mul"),'
              ' make_group(g, [4], "down")]'),
}


@pytest.mark.skipif(_skip_reason() is not None, reason=_skip_reason() or "")
@pytest.mark.parametrize("shape", sorted(_MLP_PARTITIONS))
def test_lowered_mlp_matches_torch(shape):
    """The whole path: graph -> partition -> TaskSpec -> search -> register ->
    megakernel, against torch, at both MLP partitions."""
    src = _MLP_SRC.replace("__PARTITION__", _MLP_PARTITIONS[shape])
    proc = subprocess.run([sys.executable, "-c", src],
                          capture_output=True, text=True, timeout=3600)
    tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-25:])
    assert proc.returncode == 0, f"lowered MLP ({shape}) failed:\n{tail}"
    assert "[lower]" in proc.stdout, "lowering reported nothing"


# ------------------------------------------------------ Qwen3 as one graph

from mirage.mpk.models.qwen3.builder_low_level_ir import (
    Qwen3Shapes, build_qwen3, partition_as_today)
from mirage.mpk.lowering import is_opaque

QWEN3_06B = Qwen3Shapes(tokens=8, hidden=1024, intermediate=3072,
                        num_layers=28, num_q_heads=16, num_kv_heads=8,
                        head_dim=128, vocab=151936)


def test_one_layer_has_the_expected_shape():
    g = build_qwen3(QWEN3_06B, num_layers=1)
    ops = [n.op for n in g.nodes]
    # embedding | norm qkv attn o resid | norm gate up silu mul down resid |
    # final norm, lm_head, argmax
    assert ops == [
        "opaque:embedding",
        "opaque:rmsnorm", "matmul", "opaque:attention", "matmul", "add",
        "opaque:rmsnorm", "matmul", "matmul", "silu", "mul", "matmul", "add",
        "opaque:rmsnorm", "matmul", "opaque:argmax",
    ], ops
    assert g.nodes[2].output.dims == (8, QWEN3_06B.qkv_dim)   # fused qkv
    assert g.nodes[-2].output.dims == (8, QWEN3_06B.vocab)    # logits


def test_what_stays_opaque_and_why():
    """Three of these have no muGraph op at all: embedding is a gather, the
    KV-cache append inside attention is a stateful scatter, argmax reduces to
    indices. rmsnorm is different -- the op exists, but search returns no
    usable schedule for it, so it stays a hand-written task. If this list
    grows, the graph lost coverage rather than gained a feature."""
    g = build_qwen3(QWEN3_06B, num_layers=2)
    opaque = sorted({n.op for n in g.nodes if is_opaque(n.op)})
    assert opaque == ["opaque:argmax", "opaque:attention", "opaque:embedding",
                      "opaque:rmsnorm"]


def test_full_model_partitions_and_covers():
    g = build_qwen3(QWEN3_06B)
    assert len(g) == 340, len(g)
    parts = partition_as_today(g)
    check_covers(g, parts)                      # raises if it does not
    assert len(parts) == 312, len(parts)


def test_as_today_fuses_what_mpk_already_fuses():
    """rms_norm+mul is MPK's rmsnorm_layer; silu+mul is silu_mul_layer. The
    matmuls stay separate, which is MPK_COMPILED_MLP_IMPL=separate."""
    g = build_qwen3(QWEN3_06B, num_layers=1)
    tags = [p.tag for p in partition_as_today(g)]
    assert tags == [
        "embedding", "rmsnorm", "matmul", "attention", "matmul", "add",
        "rmsnorm", "matmul", "matmul", "silu_mul", "matmul", "add",
        "rmsnorm", "matmul", "argmax",
    ], tags


def test_a_matmul_group_asks_for_a_64_wide_k_tile():
    """Left free, search picks any forloop_range that VERIFIES, and verifying
    says nothing about speed. Measured on Qwen3-0.6B: for gate/up (K=1024) it
    chose forloop_range=64 -- a K tile of 16, the minimum bf16 MMA K-atom --
    and those tasks ran 24.5 us/call against 3.9-7.1 us for the hand-written
    ones, which is most of the graph path's deficit. Every generated_* layer
    uses K // 64, so lowering asks for that.

    Only where the matmul is the group's first op; elsewhere the forloop runs
    over the output and 1 is right.
    """
    from mirage.mpk.lowering import default_forloop
    g = _qwen3_mlp()
    assert default_forloop(g, make_group(g, [0], "gate")) == H // 64   # 16
    assert default_forloop(g, make_group(g, [4], "down")) == I // 64   # 48
    assert default_forloop(g, make_group(g, [2, 3], "silu_mul")) == 1


def test_the_k_loop_does_not_depend_on_where_the_matmul_sits():
    """A group that fuses `silu | mul | down` has a matmul as its LAST node,
    and it still needs a K loop over 3072.

    Testing only node 0 gave that group forloop_range=1, i.e. one 3072-wide K
    step -- past the 256 a TMA box side allows -- so search returned nothing
    and, with three inputs, _pick_fallback offered no fallback either. The
    group died at lowering for a reason that had nothing to do with it.
    """
    from mirage.mpk.lowering import (default_forloop, forloop_candidates,
                                  grid_candidates, make_group)

    g = build_qwen3(QWEN3_06B, num_layers=1)
    silu = next(i for i, n in enumerate(g.nodes) if n.op == "silu")
    mul = g.consumers(g.nodes[silu].output)[0]
    down = g.consumers(g.nodes[mul].output)[0]
    grp = make_group(g, [silu, mul, down], "silu_mul_matmul")

    k = grp.external_inputs[0].dims[-1]
    assert k == QWEN3_06B.intermediate
    assert default_forloop(g, grp, 128) == k // 128
    assert forloop_candidates(g, grp) == [k // 64, k // 128, k // 256]
    # the N tile is still a real choice for it, not the elementwise default
    assert len(grid_candidates(g, grp)) == 2


def test_the_validator_rejects_the_hang_without_losing_the_fusion():
    """The accumulator rule has to separate two shapes search proposes.

    REJECT: [input, input, accum, accum, silu, mul, output] -- accumulating the
    raw INPUTS. An identity at forloop_range == 1, so equivalence checking
    passes it; the task then registers, compiles and hangs the megakernel
    (measured: two hours, no token).

    KEEP: accum(matmul) with the activations AFTER, feeding the output. That
    is the fused gate+up+SwiGLU task, and an earlier, stricter version of this
    check ("the output must come from an accum") silently removed it from the
    partition search space.
    """
    from mirage.mpk.lowering.task_search import _check_accum_operands, TaskSearchError

    def ops(spec):
        return [{"op_type": t, "output_tensors": [{"guid": o}],
                 "input_tensors": [{"guid": i} for i in ins]}
                for t, o, ins in spec]

    hung = ops([("tb_input_op", 1, []), ("tb_input_op", 2, []),
                ("tb_forloop_accum_no_red_op", 3, [1]),
                ("tb_forloop_accum_no_red_op", 4, [2]),
                ("tb_silu_op", 5, [3]), ("tb_mul_op", 6, [5, 4]),
                ("tb_output_op", 7, [6])])
    with pytest.raises(TaskSearchError, match="accumulates a task input"):
        _check_accum_operands(hung)

    silu_mul = ops([("tb_input_op", 1, []), ("tb_input_op", 2, []),
                    ("tb_silu_op", 3, [1]), ("tb_mul_op", 4, [3, 2]),
                    ("tb_forloop_accum_no_red_op", 5, [4]),
                    ("tb_output_op", 6, [5])])
    fused_mlp = ops([("tb_input_op", 1, []), ("tb_input_op", 2, []),
                     ("tb_input_op", 3, []),
                     ("tb_matmul_op", 4, [1, 2]),
                     ("tb_forloop_accum_no_red_op", 5, [4]),
                     ("tb_matmul_op", 6, [1, 3]),
                     ("tb_forloop_accum_no_red_op", 7, [6]),
                     ("tb_silu_op", 8, [5]), ("tb_mul_op", 9, [8, 7]),
                     ("tb_output_op", 10, [9])])
    _check_accum_operands(silu_mul)      # raises if it regresses
    _check_accum_operands(fused_mlp)


def test_the_matmul_n_tile_is_a_searched_axis():
    """The N tile a matmul group gets becomes the MMA's M under swapAB, so it
    is a real choice with only two legal values.

    At decode M is the token count, so matmul_swaps_ab (m != 64 && m != 128)
    is always true and the OUTPUT's N lands in the MMA's M slot -- where
    tcgen05 1-SM accepts 64 or 128 and nothing else. The hand-written
    linear_sm100 hardcodes MMA_M = 128; lowering has always used 64, which
    issues an MMA covering half the output columns per instruction.

    Widening is not free: it halves the block count (192 -> 96 matmul blocks
    per layer), and at decode occupancy is scarce. So this is enumerated, not
    decided here.
    """
    from mirage.mpk.lowering import grid_candidates, default_grid, MATMUL_N_TILES

    assert MATMUL_N_TILES == (64, 128), "tcgen05 1-SM takes no other M tile"

    g = build_qwen3(QWEN3_06B, num_layers=1)
    parts = partition_as_today(g)
    mm = [p for p in parts if p.tag.startswith("matmul")]
    for p in mm:
        n = p.output.dims[-1]
        assert grid_candidates(g, p) == [
            (n // t, 1, 1) for t in MATMUL_N_TILES if n % t == 0], p.tag

    # a non-matmul group has no choice -- its tile is just the elementwise width
    (sm,) = [p for p in parts if p.tag == "silu_mul"]
    assert grid_candidates(g, sm) == [default_grid(sm)]


def test_k_tile_and_pipeline_depth_are_enumerated_per_group():
    """Two more axes that were fixed constants, now candidates.

    K tile: Mirage's search DOES explore franges on its own, but it picks any
    value that verifies, and verifying says nothing about speed -- for gate/up
    (K=1024) it chose forloop_range=64, a K tile of 16, the minimum bf16 MMA
    atom. Pinning it to K/64 fixed that but replaced one fixed policy with
    another; these are the legal alternatives.

    Pipeline depth: TranspilerConfig is built PER OP inside
    register_generated_task, so the plumbing was always per-task -- only the
    policy was global. It now travels as params[0].
    """
    from mirage.mpk.lowering import (MATMUL_K_TILES, MMA_K_ATOM,
                                  forloop_candidates, default_forloop)

    assert all(t % MMA_K_ATOM == 0 and t <= 256 for t in MATMUL_K_TILES), (
        "a K tile must be whole MMA K-atoms, and no TMA box side exceeds 256")

    g = build_qwen3(QWEN3_06B, num_layers=1)
    parts = partition_as_today(g)
    for p in parts:
        cands = forloop_candidates(g, p)
        if not p.tag.startswith("matmul"):
            assert cands == [1], p.tag          # forloop is over the output
            continue
        k = p.external_inputs[0].dims[-1]
        assert cands == [k // t for t in MATMUL_K_TILES if k % t == 0], p.tag
        # the default is the 64-wide tile, and it is among the candidates
        assert default_forloop(g, p) == k // 64
        assert default_forloop(g, p) in cands
