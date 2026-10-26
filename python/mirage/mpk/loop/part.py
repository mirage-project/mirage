"""A model part for the loop: demo/<model>_<part>/part.yaml and the hooks module it names.

part.yaml (YAML; the fields the loop reads):
  name          e.g. kimi_k3_moe
  model         e.g. Kimi K3
  module        the hooks module, a path relative to the part's directory (below)
  tp            GPUs the part is divided over
  machine       {gpu, cuda}: the machine the numbers below are from
  buckets       {tokens: {inputs: {input set name: directory}, bound_us, sglang_us}}: per token bucket the input sets (each a
                directory the hooks read; "~" allowed), the part's bound and SGLang's in-server time (null: not known yet)
  phases        {phase: [output tensor names]}: the nodes of each phase, by the name of their first output tensor; the model's
                error is tracked per phase (model.py)
  error_limits  {name: limit} and "derivation": how the limits were chosen (the hooks' check uses them)
  server        {before, after}: the kernel names just before and after the part in a server trace (the window's boundaries)
  flush_bytes   bytes read on every GPU before each launch (L2 emptied); default 512 MB
  graph_options {name: default}: the options the hooks' build takes (a candidate may set them)
  window        optional: the server window without the server (rule H2): {build, run} commands ({python}, {torchrun}, {input},
                {plan}, {out}, {flags}, {reps} filled in; run in the part's directory), {result} and {check} regexes on run's output
                (result: group 1 = the window median in us; check: groups 1 and 2 = launches passing and launches)
  sync_extra    optional: files outside the part's directory the box needs (driver.py sync list)
  promote_on    optional, "harness": a part with no server window yet promotes a candidate faster by >= noise in every
                harness round (round_job.py); without it, a part with no window is never promoted

The hooks module (Python, loaded from its path):
  build(input_dir, gpus, tokens, options) -> (ws, mpk)   every GPU's tensors (ws[g]: {name: tensor on GPU g}) and the
                                                         StaticMegakernel holding the part's graph (default grids)
  check(h, ws, input_dir, out_dir) -> bool               one launch (h: static_harness.Harness) and the part's output checks
  watch(ws) -> {name: tensor}                            optional: tensors expected bit-identical over the timed launches
  before_launch(ws, i)                                   optional: called before timed launch i (e.g. a part with state that
                                                         lives across launches: chained launches)
"""
import importlib.util
import os
from dataclasses import dataclass, field
from types import ModuleType
from typing import Dict, List, Optional

REQUIRED_FIELDS = ("name", "module", "tp", "buckets", "phases", "error_limits")
REQUIRED_HOOKS = ("build", "check")
DEFAULT_FLUSH_BYTES = 512 << 20


@dataclass
class Part:
    dir: str
    cfg: dict
    hooks: Optional[ModuleType] = field(default=None, repr=False)

    @property
    def name(self) -> str:
        return self.cfg["name"]

    @property
    def tp(self) -> int:
        return int(self.cfg["tp"])

    @property
    def flush_bytes(self) -> int:
        return int(self.cfg.get("flush_bytes", DEFAULT_FLUSH_BYTES))

    def buckets(self) -> List[int]:
        return sorted(int(b) for b in self.cfg["buckets"])

    def bucket(self, tokens: int) -> dict:
        for b, spec in self.cfg["buckets"].items():
            if int(b) == int(tokens):
                return spec or {}
        raise KeyError(f"part {self.name}: no bucket of {tokens} tokens (buckets {self.buckets()})")

    def input_dir(self, tokens: int, input_set: Optional[str] = None) -> str:
        """The directory of an input set of a bucket (the first set when input_set is None)."""
        inputs = self.bucket(tokens).get("inputs") or {}
        if not inputs:
            raise KeyError(f"part {self.name}: bucket {tokens} has no input sets")
        name = input_set or next(iter(inputs))
        if name not in inputs:
            raise KeyError(f"part {self.name}: bucket {tokens} has no input set {name} (sets {list(inputs)})")
        return os.path.expanduser(str(inputs[name]))

    def input_sets(self, tokens: int) -> List[str]:
        return list((self.bucket(tokens).get("inputs") or {}).keys())

    def phase_of(self) -> Dict[str, str]:
        """output tensor name -> phase."""
        return {t: ph for ph, names in self.cfg["phases"].items() for t in names}

    def graph_options(self, overrides: Optional[dict] = None) -> dict:
        opts = dict(self.cfg.get("graph_options") or {})
        for k, v in (overrides or {}).items():
            if k not in opts:
                raise KeyError(f"part {self.name}: no graph option {k} (options {sorted(opts)})")
            opts[k] = v
        return opts


def read_config(part_dir: str) -> dict:
    import yaml
    path = os.path.join(part_dir, "part.yaml")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    missing = [k for k in REQUIRED_FIELDS if k not in cfg]
    if missing:
        raise ValueError(f"{path}: missing fields {missing}")
    return cfg


def load_hooks(part_dir: str, cfg: dict) -> ModuleType:
    path = os.path.join(part_dir, cfg["module"])
    spec = importlib.util.spec_from_file_location(f"mk_part_{cfg['name']}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    missing = [h for h in REQUIRED_HOOKS if not callable(getattr(module, h, None))]
    if missing:
        raise ValueError(f"{path}: missing hooks {missing} (loop/part.py)")
    return module


def load(part_dir: str, with_hooks: bool = True) -> Part:
    """The part in part_dir; with_hooks False: only part.yaml (no torch / Mirage needed)."""
    part_dir = os.path.abspath(os.path.expanduser(part_dir))
    cfg = read_config(part_dir)
    return Part(part_dir, cfg, load_hooks(part_dir, cfg) if with_hooks else None)
