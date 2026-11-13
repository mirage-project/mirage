"""GLM-5.3 decode on the static megakernel inside a serving engine: one TP8 rank per process, one kernel per sublayer.

Each layer runs two kernels on this rank's TP shard: the attention (demo/glm53_dsa, "full" or "shared" by the config's
indexer_types) and the FFN (the dense MLP of dense_mlp.py for layers < first_k_dense_replace, else the MoE of
demo/glm53_moe). Every kernel is built for the TP8 group and ends the sublayer itself (as the layers of
demo/glm53_flash): allreduce_send (multicast into every rank's exchange region, torch symmetric memory over the TP group),
sum_gpus, residual_add into the residual stream R (in place), row_rmsnorm -> the next sublayer's input. A decode step is
then the embedding, layer 0's input norm and the two kernels of every layer: no engine all-reduce or norm between them. A
rank that runs alone (no TP group: one GPU) builds the same graphs for one GPU: its all-reduce is a copy (as the demos of
demo/glm53_flash with --gpus 1).
One graph and plan per kind serves every layer of that kind: each layer is an instance with its own tensors and its own
copy of the built layer (static_instance; the build cache compiles each layer.cu once).

The kernels read and write the engine's caches in place: the bf16 latent cache [slots, 576] and the index-K pages
[pages, 64 x 132] with fp32 key scales (dsa_index_k_write_layer key_scale="fp32"), so the engine's prefill writes rows
the kernels read and the kernels write the rows of each decode step. Positions map to slots through the engine's page
table (page = slot // 64). The graph tensors that change every step (seq_lens, the block table) are buffers of this
object, filled before the launches; the hidden states go through the residual stream R and two buffers A and B (a
sublayer's normed input):
  attention: x = A, R += out, RMSNorm(R) -> B;   FFN: x = B, R += out, RMSNorm(R) -> A.
Rows that are not a request (padding of a CUDA graph's batch, the rows past the batch) get seq_lens 0: no cache write,
no attention.

The MoE kernels have their own copy of this rank's experts (as demo/glm53_moe, demo/glm53_flash's StaticMoE): read from the
checkpoint into the piece layout of mirage.mpk.fp8_fp32scale_weights (the same bytes and scale values), about 1.2 GB per
layer. The engine keeps its own weights, so its prefill runs its own layers.

MIRAGE_STATIC_ALLREDUCE_IN_KERNEL=0: the engine ends the sublayers instead. Every kernel is built for one GPU and ends with
this rank's partial sum (the attention at o_proj; the FFN's all-reduce is a copy and its residual zero), written to the
other buffer (attention: A -> B; FFN: B -> A). The engine does the rest of the layer: the all-reduce over the TP group,
then the residual add and the next RMSNorm (fused_add_rmsnorm).
"""
import os

import torch

from demo.glm53_dsa import demo as dsa_demo
from demo.glm53_dsa import weights as dsa_weights
from demo.glm53_model import dense_mlp
from demo.glm53_moe import demo as moe_demo
from demo.glm53_moe import weights as moe_weights
from mirage.mpk import compiler
from mirage.mpk.static_megakernel import StaticMegakernel

# T: rows per decode step of the kernels (a token of each request), STATIC_TOKENS (8, 16 or 32; the demos read it too)
T = int(os.environ.get("STATIC_TOKENS", "8"))
H, PAGE, TOPK, E, TP = 6144, 64, 2048, 256, 8
assert T in (8, 16, 32) and dsa_demo.T == T and moe_demo.T == T and dense_mlp.T == T, "STATIC_TOKENS: 8, 16 or 32"
HERE = os.path.dirname(os.path.abspath(__file__))
# the MoE graph: the shared expert in the experts' bank (moe_demo.shared_bank, as demo/glm53_flash's MoE) for T > 8, in its
# own nodes (demo/glm53_moe's) for T = 8: the faster graph at each T
MOE_BANK = T > 8
# the plans of the four kernel kinds: the hand plans, or with MIRAGE_STATIC_PLAN=solver the solver plans of the demos
# (search.search_plan on B200 cost records; the dense layer has only a hand plan). These plans are of T = 8, with the MoE
# shared expert in its own nodes. For T = 16 / 32, only the attention has hand plans (dsa_demo.plan_file). The kinds with
# no plan use round robin (moe_demo.round_robin_plan).
_PLAN = os.environ.get("MIRAGE_STATIC_PLAN", "hand")
if _PLAN not in ("hand", "solver"):
    raise ValueError(f"MIRAGE_STATIC_PLAN={_PLAN}: hand or solver")
PLANS = {"attn_full": dsa_demo.plan_file(_PLAN, True), "attn_shared": dsa_demo.plan_file(_PLAN, False),
         "moe": os.path.join(os.path.dirname(moe_demo.__file__), f"{_PLAN}_plan.json"),
         "dense": os.path.join(HERE, "hand_plan_dense.json")} if T == 8 else {
             "attn_full": dsa_demo.plan_file("hand", True) if _PLAN == "hand" else None,
             "attn_shared": dsa_demo.plan_file("hand", False) if _PLAN == "hand" else None, "moe": None, "dense": None}
if MOE_BANK:
    PLANS["moe"] = None
# a plan file per kind instead (of this T and graph, e.g. a searched one): MIRAGE_STATIC_PLAN_ATTN_FULL, _ATTN_SHARED, _MOE,
# _DENSE
for _k in PLANS:
    if os.environ.get(f"MIRAGE_STATIC_PLAN_{_k.upper()}"):
        PLANS[_k] = os.environ[f"MIRAGE_STATIC_PLAN_{_k.upper()}"]
# the serving build: the kernel resets its own buffers
SERVING = ["-DSTATIC_RESET_IN_KERNEL"]
# MIRAGE_STATIC_PDL=1 (the default; 0: off): the kernels launch programmatically (PDL). The start of a kernel (its resets,
# barrier, TMEM) then overlaps the end of the kernel before it.
PDL = os.environ.get("MIRAGE_STATIC_PDL", "1") == "1"
# MIRAGE_STATIC_ALLREDUCE_IN_KERNEL=1 (the default; 0: the engine's all-reduce and norms, module docstring): the kernels
# are built for the TP group (for one GPU if the rank runs alone) and end their sublayer themselves: the all-reduce, the
# residual add and the next norm (the norm needs the reduced sum, so it goes with the all-reduce)
ALLREDUCE_IN_KERNEL = os.environ.get("MIRAGE_STATIC_ALLREDUCE_IN_KERNEL", "1") == "1"
# where a kernel lets the next kernel launch (config.cuh STATIC_PDL_TRIGGER): at residual_add (2), so the next kernel's
# start overlaps this kernel's norm; at its end (0) when the engine's norms run between the kernels
SERVING.append(f"-DSTATIC_PDL_TRIGGER={2 if ALLREDUCE_IN_KERNEL else 0}")
MAXCTX = 16384   # the context of the attention plans (the top-k grid depends on it): requests up to 16384 positions


def static_instance(mpk: StaticMegakernel, paths, t: dict, out_dir: str, build_dir: str, device: int, rank: int = None,
                    exchange: tuple = None):
    """Load one layer of a model: its own copy of the serving build of graph mpk (schedules `paths`) on tensors t, in
    out_dir (a loaded extension holds one layer's state, so each instance has its own directory). The build cache
    (MIRAGE_STATIC_BUILD_CACHE, default <build_dir>/cache) compiles each distinct layer.cu once. For a graph of several
    GPUs, the instance is GPU `rank` of the graph, and exchange = (multicast address, this GPU's address) of its exchange
    region."""
    os.environ.setdefault("MIRAGE_STATIC_BUILD_CACHE", os.path.join(build_dir, "cache"))
    return mpk.compile_static(paths, gpu_tensors=[t], out_dir=out_dir, extra_flags=SERVING, device=device,
                              **({"rank": rank, "exchange": exchange} if mpk.num_gpus > 1 else {}))


class ExchangePool:
    """The exchange regions of all kernel instances of this rank, in one allocation of torch symmetric memory over the TP
    group (one rendezvous, one multicast object; as demo/glm53_flash's ExchangePool, with regions of several sizes).
    sizes: the sizes of all regions in the order the instances take them. The pool rounds each size up to 64 KB. Every
    rank takes the regions in the same order. The pool owns the memory: keep it referenced as long as the kernels run."""

    def __init__(self, group, dev, sizes: list):
        from torch._C._distributed_c10d import _SymmetricMemory
        self.sizes = [-(-n // 65536) * 65536 for n in sizes]
        self.buf = _SymmetricMemory.empty_strided_p2p((sum(self.sizes),), [1], torch.uint8, dev, group.group_name)
        self.handle = _SymmetricMemory.rendezvous(self.buf)
        self.mc = int(self.handle.multicast_ptr)
        if self.mc == 0:
            raise RuntimeError("the TP group's symmetric memory has no multicast address (NVLS). "
                               "MIRAGE_STATIC_ALLREDUCE_IN_KERNEL=0 runs without it")
        self.buf.zero_()
        self.taken, self.off = 0, 0

    def take(self, nbytes: int) -> tuple:
        """(the multicast address, this GPU's address) of the next region (of nbytes, at most its listed size)."""
        assert self.taken < len(self.sizes) and nbytes <= self.sizes[self.taken], "ExchangePool: regions out of order"
        off = self.off
        self.off += self.sizes[self.taken]
        self.taken += 1
        return self.mc + off, self.buf.data_ptr() + off


class StaticDecoder:
    """The static-megakernel decode of one TP rank. load_weights: the kernels' own tensors (before the engine sizes its KV
    pool). setup: after the KV pool exists, before CUDA graph capture: the builds and the instances. Then decode() runs one
    decode step."""

    def __init__(self, cfg: dict, read, rank: int, device, maxctx: int = MAXCTX, build_dir: str = None, group=None):
        """group: the TP process group (ALLREDUCE_IN_KERNEL: the kernels' all-reduce runs over it, its symmetric memory
        holds their exchange regions); None: this rank runs alone (one GPU)."""
        if ALLREDUCE_IN_KERNEL and group is None and torch.distributed.is_initialized() and \
                torch.distributed.get_world_size() > 1:
            raise ValueError("MIRAGE_STATIC_ALLREDUCE_IN_KERNEL: the TP process group (StaticDecoder group=): the kernels' "
                             "all-reduce runs over it")
        self.group = group
        # the GPUs of a kernel's graph: the TP group, or one (a rank alone; the engine's all-reduce)
        self.gpus = TP if ALLREDUCE_IN_KERNEL and group is not None else 1
        if maxctx != MAXCTX:
            raise ValueError(f"maxctx {maxctx}: the attention plans are for a context of {MAXCTX}")
        if cfg["indexer_types"][0] != "full":
            raise ValueError("layer 0 must run its indexer: a shared layer reads the indices of the last full layer")
        self.cfg, self.read, self.rank, self.dev, self.maxctx = cfg, read, rank, device, maxctx
        self.L, self.dense = cfg["num_hidden_layers"], cfg["first_k_dense_replace"]
        self.full = [t == "full" for t in cfg["indexer_types"][:self.L]]
        self.build_dir = build_dir or os.path.expanduser("~/.cache/mirage/glm53_static")
        self.attn_w, self.ffn_w, self.instances, self.ready = [], [], None, False

    def load_weights(self):
        """The tensors of the attention (every layer), the dense MLP and the MoE (router, bias, the routed experts' bank and
        the shared expert) of this rank, from the checkpoint."""
        for l in range(self.L):
            w = dsa_weights.layer_tensors(self.read, l, tp=TP, rank=self.rank, device=self.dev, full=self.full[l])
            w.pop("input_norm")   # not a kernel tensor: the engine's norm, or the kernel before (norm_next below)
            self.attn_w.append(w)
            self.ffn_w.append(dense_mlp.dense_tensors(self.read, l, tp=TP, rank=self.rank, device=self.dev) if l < self.dense
                              else self._moe_tensors(l))
        if ALLREDUCE_IN_KERNEL:   # the kernels' norms: after the attention (post), before the next layer (input), final
            get = lambda name: self.read(name).to(self.dev, torch.bfloat16).contiguous()
            self.norm_post = [get(f"model.layers.{l}.post_attention_layernorm.weight") for l in range(self.L)]
            self.norm_next = [get(f"model.layers.{l + 1}.input_layernorm.weight") for l in range(self.L - 1)] + \
                [get("model.norm.weight")]
        torch.cuda.synchronize(self.dev)

    def _moe_tensors(self, l: int) -> dict:
        """Layer l's MoE tensors of this rank from the checkpoint (MOE_BANK: the shared expert in the bank)."""
        w = moe_weights.layer_tensors(self.read, l, tp=TP, rank=self.rank, device=self.dev)
        return moe_demo.shared_bank(w) if MOE_BANK else w

    # ---- setup: after the KV pool, before CUDA graph capture ----
    def setup(self, kv_pool, req_to_token: torch.Tensor, barrier=None):
        """kv_pool: the engine's DSA pool (get_key_buffer(l): [slots, 1, 576] bf16; get_index_k_with_scale_buffer(l):
        [pages, 64 x 132] uint8 for a full layer). req_to_token: [requests, ctx] int32 slots. barrier: a barrier over the
        ranks. Graphs of one GPU: rank 0 builds, then the barrier, then the other ranks load the builds. Graphs of the TP
        group: every rank loads at once, then the barrier (no rank launches a kernel before every rank holds its exchange
        regions)."""
        dev = self.dev
        maxp = self.maxctx // PAGE
        self.req_to_token = req_to_token
        self.seq_lens = torch.zeros(T, dtype=torch.int32, device=dev)
        self.block_table = torch.zeros(T, maxp, dtype=torch.int32, device=dev)
        self.cos_sin = dsa_demo.rope_table(self.maxctx, self.cfg["rope_parameters"]["rope_theta"]).to(dev)
        self.A, self.B = (torch.zeros(T, H, dtype=torch.bfloat16, device=dev) for _ in range(2))
        self.R = torch.zeros(T, H, dtype=torch.bfloat16, device=dev)   # the residual stream, updated in place by the kernels
        self.zero = torch.zeros(T, H, dtype=torch.bfloat16, device=dev)
        self.indices = torch.full((T, TOPK), -1, dtype=torch.int32, device=dev)
        K = 8 + MOE_BANK   # pairs per token: 8 routed (+ the shared expert, MOE_BANK)
        self.pairs = torch.zeros(T, K, dtype=torch.int64, device=dev)
        # the MoE's live rows (topk_route_live_rows): a row with live <= 0 (a padded row: seq_lens 0) streams no expert of
        # its own
        self.live = torch.zeros(T, dtype=torch.int32, device=dev)
        self.rows = torch.zeros(T * K, H, dtype=torch.float32, device=dev)
        st = {"seq_lens": self.seq_lens, "block_table": self.block_table, "cos_sin": self.cos_sin}
        attn_t, ffn_t = [], []
        for l in range(self.L):
            cache = kv_pool.get_key_buffer(l)
            assert cache.dtype == torch.bfloat16 and cache.shape[-1] == 576 and cache.shape[0] % PAGE == 0, \
                "the latent cache must be bf16 rows of 576 in whole pages (--kv-cache-dtype bfloat16, page size 64)"
            bufs = {"x": self.A, "cache": cache.view(-1, 576), "indices": self.indices, "residual": self.zero,
                    "attn_out": self.B}
            if ALLREDUCE_IN_KERNEL:
                bufs.update(residual=self.R, attn_out=self.R, norm_w=self.norm_post[l], x_next=self.B)
            if self.full[l]:
                bufs["index_cache"] = kv_pool.get_index_k_with_scale_buffer(l)
            attn_t.append(dsa_demo.kernel_tensors(self.attn_w[l], st, bufs))
            fused = {"residual": self.R, "norm_w": self.norm_next[l], "x_next": self.A} if ALLREDUCE_IN_KERNEL else {}
            if l < self.dense:
                ffn_t.append({**self.ffn_w[l], "x": self.B, "residual": self.zero, "mlp_out": self.A,
                              **fused, **({"mlp_out": self.R} if ALLREDUCE_IN_KERNEL else {})})
            else:
                moe = {**self.ffn_w[l], "x": self.B, "routing_pairs": self.pairs, "routed_rows": self.rows,
                       "moe_out": self.A, "residual": self.zero, "live": self.live}
                ffn_t.append({**moe, **fused, **({"moe_out": self.R} if ALLREDUCE_IN_KERNEL else {})})
        kinds = ["attn_full" if f else "attn_shared" for f in self.full] + \
                ["dense" if l < self.dense else "moe" for l in range(self.L)]
        tensors = attn_t + ffn_t
        first = {k: tensors[kinds.index(k)] for k in set(kinds)}
        plans = {k: self._plan(k, t) for k, t in first.items()}
        if self.gpus > 1:   # every instance's exchange region, in load order (the same on every rank)
            self.pool = ExchangePool(self.group, dev, [plans[k][0].exchange_region_bytes() for k in kinds])

        def load():
            return [static_instance(plans[k][0], plans[k][1], t,
                                    os.path.join(plans[k][2], f"r{self.rank}_{'a' if i < self.L else 'f'}{i % self.L}"),
                                    self.build_dir, dev.index, self.rank,
                                    self.pool.take(plans[k][0].exchange_region_bytes()) if self.gpus > 1 else None)
                    for i, (k, t) in enumerate(zip(kinds, tensors))]
        if self.rank == 0 or self.gpus > 1:
            # rank 0 compiles each kind once (the build cache), the other ranks then copy; the TP builds: every rank at once
            # (the exchange pool's rendezvous is collective; the build cache's lock serializes the compiles)
            self.instances = load()
        if barrier is not None:
            barrier()   # TP builds: every rank's kernels hold their exchange regions before any rank launches one
        if self.rank != 0 and self.gpus == 1:
            self.instances = load()
        self.ready = True

    def _graph(self, kind: str, t: dict):
        """The decode graph of `kind` on tensors t, for self.gpus GPUs (ALLREDUCE_IN_KERNEL: the norm inside)."""
        mpk = StaticMegakernel(num_gpus=self.gpus)
        # the norm's tensors are attached after the others (build_graph next_norm), so the plans' node indices stay valid
        nn = (t["norm_w"], t["x_next"], self.cfg["rms_norm_eps"]) if ALLREDUCE_IN_KERNEL else None
        live = t.get("live")   # attached last too, by this name (moe build_graph live): the instance binds it by name
        t = {k: v for k, v in t.items() if k not in ("norm_w", "x_next", "live")}
        if kind.startswith("attn"):
            full = kind == "attn_full"
            graph = ("x", "cache", "indices", "residual", "attn_out", "index_cache", "norm_w", "x_next")
            w = {k: v for k, v in t.items() if k not in ("seq_lens", "block_table", "cos_sin") + graph}
            st = {k: t[k] for k in ("seq_lens", "block_table", "cos_sin")}
            bufs = {k: t[k] for k in graph if k in t}
            cfg = self.cfg
            dsa_demo.build_graph(mpk, w, st, bufs, full, cfg, self.maxctx, 16, 16, cfg["rms_norm_eps"], key_scale="fp32",
                                 tail=ALLREDUCE_IN_KERNEL,   # else the engine all-reduces and adds: o_proj writes attn_out
                                 next_norm=nn)
        elif kind == "moe":
            grids = moe_demo.build_graph(mpk, t, next_norm=nn, live=live)
        else:
            dense_mlp.build_graph(mpk, t, next_norm=nn)
        return mpk, (grids if kind == "moe" else None)

    def _plan(self, kind: str, t: dict):
        """(graph, schedule paths, directory) of `kind` with the shapes of tensors t: a directory per kind and pool size."""
        mpk, rr_grids = self._graph(kind, t)
        tag = (f"{kind}_pages{t['cache'].shape[0] // PAGE}_ctx{self.maxctx}" if "cache" in t else kind) + f"_{_PLAN}_T{T}" + \
            ("_bank" if kind == "moe" and MOE_BANK else "")
        if ALLREDUCE_IN_KERNEL:
            tag += f"_allreduce{self.gpus}"
        out = os.path.join(self.build_dir, tag)
        if PLANS[kind] is None:   # round robin over this graph's nodes (row_rmsnorm included)
            plan = moe_demo.round_robin_plan(mpk, rr_grids or {})
            paths, _ = compiler.compile_plan(mpk, plan, os.path.join(out, f"schedules_r{self.rank}"))
            return mpk, paths, out
        plan = compiler.plan_from_file(PLANS[kind])
        if kind.startswith("attn"):   # a plan of the graph with its tail, on a graph without it (build_graph tail)
            plan = dsa_demo.drop_missing_nodes(mpk, plan)
        if len(plan["lists"]) == 1:   # a one-GPU plan file: every GPU of the graph gets the same lists
            plan["lists"] = plan["lists"] * mpk.num_gpus
        node = next((n for n in compiler.graph_nodes(mpk) if n.name == "row_rmsnorm"), None) if ALLREDUCE_IN_KERNEL else None
        if node is not None and all(e[0] != node.graph_idx for lists in plan["lists"] for sm in lists for e in sm):
            # a plan without the row_rmsnorm node: its tasks after the lists of SMs 0, 1, ... of every GPU (each waits
            # for all residual_add tasks, which end every list: no cycle)
            plan["lists"] = [[list(sm) for sm in lists] for lists in plan["lists"]]
            for lists in plan["lists"]:
                for i, (x, y) in enumerate((x, y) for y in range(node.grid[1]) for x in range(node.grid[0])):
                    lists[i].append([node.graph_idx, [x, y, 0]])
        paths, _ = compiler.compile_plan(mpk, plan, os.path.join(out, f"schedules_r{self.rank}"))
        return mpk, paths, out

    # ---- one decode step ----
    def decode(self, model, input_ids: torch.Tensor, forward_batch, all_reduce) -> torch.Tensor:
        """The hidden states after the final norm [bs, H] for the decode batch (bs <= T), a view of a buffer that the next
        step overwrites. model: the engine's model (its embedding and layer 0's input norm; with ALLREDUCE_IN_KERNEL=0 also
        its other norms). all_reduce: the engine's TP all-reduce (it may return a new tensor). Not used with
        ALLREDUCE_IN_KERNEL: the kernels do the all-reduce."""
        bs = input_ids.shape[0]
        assert bs <= T, f"decode batch {bs}: at most {T} requests (--max-running-requests)"
        rpi = forward_batch.req_pool_indices[:bs]
        # row 0 of the request pool is the padding of a CUDA graph's batch: no request. The rows past the batch get
        # seq_lens 0 too: a graph of a smaller batch must not see the requests of a larger one from an earlier step.
        self.seq_lens[bs:].zero_()
        self.block_table[bs:].zero_()
        self.seq_lens[:bs].copy_(torch.where(rpi != 0, forward_batch.seq_lens[:bs], 0))
        pages = self.req_to_token[rpi, ::PAGE][:, :self.block_table.shape[1]] // PAGE
        self.block_table[:bs, :pages.shape[1]].copy_(pages)
        self.live.copy_(self.seq_lens)   # the MoE routes only the rows with a request
        stream = torch.cuda.current_stream().cuda_stream
        layers, mm = model.model.layers, model.model
        h = mm.embed_tokens(input_ids)
        residual = h
        self.A[bs:].zero_()   # the rows past the batch: zero input (else the last step's output of a larger batch)
        self.A[:bs].copy_(layers[0].input_layernorm(h))
        if ALLREDUCE_IN_KERNEL:   # the kernels add into R and write the next sublayer's normed input (A <-> B)
            # the kernels add into R in place: the rows past the batch start each step from zero (else they keep adding
            # up over the steps and the layers, to inf, then NaN through the norms)
            self.R[bs:].zero_()
            self.R[:bs].copy_(h)
            for l in range(self.L):
                self.instances[l].launch_stream(stream, pdl=PDL)           # attention: A, R -> R, B
                self.instances[self.L + l].launch_stream(stream, pdl=PDL)  # FFN: B, R -> R, A (the last: the final norm)
            return self.A[:bs]
        for l in range(self.L):
            self.instances[l].launch_stream(stream, pdl=PDL)           # attention: A -> B
            o = all_reduce(self.B[:bs])
            x, residual = layers[l].post_attention_layernorm(o, residual)
            self.B[:bs].copy_(x)
            self.instances[self.L + l].launch_stream(stream, pdl=PDL)   # FFN: B -> A
            o = all_reduce(self.A[:bs])
            if l + 1 < self.L:
                x, residual = layers[l + 1].input_layernorm(o, residual)
                self.A[:bs].copy_(x)
            else:
                x, _ = mm.norm(o, residual)
        return x
