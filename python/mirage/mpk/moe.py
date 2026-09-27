"""The K3 MoE layer: the hand plan (the placement of the hand-written reference kernel) and the host code's arguments.
What each node's tasks are (slices, table rows, kernel lines) is stored by its layer method (static_megakernel.StaticMegakernel's layers)."""
import json
import os
import statistics
from collections import defaultdict
from typing import Dict, List, Optional

from .compiler import Node, graph_nodes, owned_row_blocks, tname

EPI_ROUTER, EPI_LATENT, EPI_SGU = 0, 1, 2                 # the GEMM node's params[0] (kind)
HAND_PARTS = {EPI_ROUTER: 14, EPI_LATENT: 8, EPI_SGU: 8}  # the hand plan's K parts of the three front GEMMs


def gemm_by_kind(nodes: List[Node]) -> Dict[int, Node]:
    return {n.params[0]: n for n in nodes if n.name == "gemm_tile"}


def one(nodes: List[Node], name: str) -> Node:
    return next(n for n in nodes if n.name == name)


def hand_plan(pk, num_gpus: int, num_sms: int) -> dict:
    """The hand plan (compiler.compile_plan's format). Grids (K parts in grid y): router (7, 14, 1), latent_down
    (28, 8, 1), shared gate_up (12, 8, 1). Per GPU the
    lists, in the reference kernel's order (8 tokens; `first` = the GPU's first latent block, GPU 0: blocks 0..2):
      latent tile (m, p) on SM 8 + 8 (m - first) + p          GPU 0: SMs 8..31 (8 = tokens = K parts)
      quant m right behind block m's last latent tile, on SM 8 + 8 (m - first) + 7
      route t alone on SM t                                   SMs 0..7
      router tiles one per SM from SM 40                      98 tiles: SMs 40..137
      shared gate_up tiles one at a time on the SM with the fewest bytes so far (lowest SM on ties); SMs 0..7 are kept free,
      the quant SMs and SMs 40..51 start with extra bytes so they get fewer
      sact j last on SM 40 + j"""
    nodes = graph_nodes(pk)
    gemm = gemm_by_kind(nodes)
    route, quant, sact = one(nodes, "route"), one(nodes, "quant"), one(nodes, "sact")
    router, latent, shared = gemm[EPI_ROUTER], gemm[EPI_LATENT], gemm[EPI_SGU]
    tokens = route.grid[0]
    row_blocks = lambda n: n.inputs[1].dim(0) // 128
    tile_bytes = lambda kind: 128 * (gemm[kind].inputs[1].dim(1) // 128 // HAND_PARTS[kind]) * 128 * 2

    all_lists = []
    for gpu in range(num_gpus):
        lists: Dict[int, list] = defaultdict(list)
        bytes_on_sm = [0.0] * num_sms
        first, last = owned_row_blocks(row_blocks(latent), gpu, num_gpus)
        parts = HAND_PARTS[EPI_LATENT]
        quant_sm = lambda m: tokens + parts * (m - first) + parts - 1
        for m in range(first, last):                       # latent tiles
            for p in range(parts):
                sm = tokens + parts * (m - first) + p
                lists[sm].append((latent.graph_idx, (m, p, 0)))
                bytes_on_sm[sm] += tile_bytes(EPI_LATENT)
        for m in range(first, last):                       # each quant behind its block's last latent tile
            lists[quant_sm(m)].append((quant.graph_idx, (m, 0, 0)))
        for t in range(tokens):                            # route t alone on SM t
            lists[t].append((route.graph_idx, (t, 0, 0)))
        sm = 40
        for m in range(row_blocks(router)):                # router tiles, one each on SMs 40, 41, ...
            for p in range(HAND_PARTS[EPI_ROUTER]):
                lists[sm].append((router.graph_idx, (m, p, 0)))
                bytes_on_sm[sm] += tile_bytes(EPI_ROUTER)
                sm += 1
        num_shared = row_blocks(shared)
        for t in range(tokens):                            # keep the route SMs, the quant SMs and the sact SMs free of shared tiles
            bytes_on_sm[t] += 1e9
        for m in range(first, last):
            bytes_on_sm[quant_sm(m)] += 100e3
        for j in range(num_shared):
            bytes_on_sm[40 + j] += 60e3
        for m in range(num_shared):                        # shared gate_up tiles: fewest bytes so far, lowest SM on ties
            for p in range(HAND_PARTS[EPI_SGU]):
                sm = min(range(num_sms), key=lambda k: (bytes_on_sm[k], k))
                lists[sm].append((shared.graph_idx, (m, p, 0)))
                bytes_on_sm[sm] += tile_bytes(EPI_SGU)
        for j in range(num_shared):                        # sact j last on SM 40 + j
            lists[40 + j].append((sact.graph_idx, (j, 0, 0)))
        all_lists.append([lists[k] for k in range(num_sms)])
    grid = lambda n, kind: (row_blocks(n), HAND_PARTS[kind], 1)
    return {"grids": {router.graph_idx: grid(router, EPI_ROUTER), latent.graph_idx: grid(latent, EPI_LATENT),
                      shared.graph_idx: grid(shared, EPI_SGU)}, "lists": all_lists}


def moe_host_args(pk, extra: Dict[str, str]) -> Dict[str, str]:
    """The strings moe_host_init reads: per buffer role the graph tensor's name (e.g. "y" -> "moe_out") and the router and
    latent_down K split counts (they size lpart / zpart) = the GEMM nodes' grid y when the layer is built (after compile_plan set the grids). `extra`: tensors the bodies
    read only through the tensor maps, which no node lists (a node has at most 8 inputs)."""
    nodes = graph_nodes(pk)
    gemm = gemm_by_kind(nodes)
    route, sact = one(nodes, "route"), one(nodes, "sact")
    queue, tail = one(nodes, "expert_queue"), one(nodes, "tail")
    name = lambda t: tname(pk, t)
    args = {
        "globals": name(gemm[EPI_ROUTER].inputs[2]), "maps": name(gemm[EPI_ROUTER].inputs[3]),
        "x": name(gemm[EPI_ROUTER].inputs[0]),
        "router_weight": name(gemm[EPI_ROUTER].inputs[1]), "latent_down_weight": name(gemm[EPI_LATENT].inputs[1]),
        "shared_gate_up_weight": name(gemm[EPI_SGU].inputs[1]),
        "score_correction_bias": name(route.inputs[1]), "routing_pairs": name(route.outputs[0]),
        "shared_gate_up": name(gemm[EPI_SGU].outputs[0]), "shared_act": name(sact.outputs[0]),
        "w13_blocks": name(queue.inputs[3]), "w2_blocks": name(queue.inputs[4]),
        "routed_sum": name(queue.outputs[0]), "shared_down_sum": name(queue.outputs[1]),
        "prefix": name(tail.inputs[2]), "gamma": name(tail.inputs[3]), "latent_up_weight": name(tail.inputs[4]),
        "y": name(tail.outputs[0]),
        "router_ksplit": str(gemm[EPI_ROUTER].grid[1]), "latent_ksplit": str(gemm[EPI_LATENT].grid[1]),
    }
    args.update(extra)
    return args


class MoeCosts:
    """The durations search.search_plan needs, from a cost file (the measured records of earlier runs; the median of the
    records with the same task type and key):
      gemm_tile     key {"kind": params[0], "pieces": K pieces of 128 per task}      e.g. router in 14 K parts: pieces 4
      route / quant / sact   key {"partials": K parts of the GEMM that writes its input}
      expert queue     pool work, key {"pool": "experts"};  tail: the time after the pool, key {"pool": "experts", "after": true}
    A key with no record: None (search leaves that choice out)."""

    def __init__(self, path: str):
        self.records = []
        if path and os.path.exists(path):
            with open(path) as f:
                self.records = json.load(f)["records"]

    def median(self, task_type: str, key: dict) -> Optional[float]:
        values = [r["median_us"] for r in self.records if r["task_type"] == task_type and r["key"] == key]
        return statistics.median(values) if values else None

    def task_us(self, node: Node, grid: tuple, producer_grids: list) -> Optional[float]:
        if node.name == "gemm_tile":
            return self.median(node.name, {"kind": node.params[0], "pieces": node.inputs[1].dim(1) // 128 // grid[1]})
        if node.name in ("route", "quant", "sact"):
            return self.median(node.name, {"partials": producer_grids[0][1]})
        return None

    def pool_us(self, node: Node) -> Optional[float]:
        return self.median("expert_queue", {"pool": "experts"}) if node.name == "expert_queue" else None

    def after_us(self, node: Node) -> float:
        return (self.median("expert_queue", {"pool": "experts", "after": True}) or 0.0) if node.name == "tail" else 0.0
