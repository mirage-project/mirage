"""Per-launch inputs and outputs of the layer demos' old / new A/B (glm53_dsa/demo.py, glm53_moe/demo.py, dense_demo.py).

--vary-x: launch n (0 = the first, untimed; 1 .. reps the timed ones) gets its own x when n > 0: bf16 N(0, 1), seed
1000 + n, the same on every GPU (launch 0 keeps the demo's x). A value that launch n - 2 left in the same exchange set
then differs from launch n's, so a read that does not wait for launch n's value (a missing re-arm, a wrong set) gives
a y that differs from the old build's launch n. Without --vary-x every launch has the same inputs and such a read
gives the right value by chance.
--residual random: the residual bf16 N(0, 1) (seed 3, the same on every GPU) instead of zeros: y = sum + residual
checks the residual's index; with --next-norm, x_next = RMSNorm(y) gamma (row_rmsnorm, gamma bf16 1 + N(0, 0.1),
seed 4) also checks the h32 scratch residual_add writes.
Each launch's outputs (y, and x_next with --next-norm) are kept on the host; save() writes them, compare() checks them
bit for bit against a saved run, launch by launch.
"""
import torch


def varied_x(n: int, shape) -> torch.Tensor:
    """x of launch n > 0 (--vary-x): bf16 N(0, 1), seed 1000 + n (CPU)."""
    return torch.randn(shape, generator=torch.Generator().manual_seed(1000 + n)).to(torch.bfloat16)


def residual(kind: str, shape) -> torch.Tensor:
    """--residual: "zero" (y = the layer's output) or "random" (bf16 N(0, 1), seed 3) (CPU)."""
    if kind == "zero":
        return torch.zeros(shape, dtype=torch.bfloat16)
    return torch.randn(shape, generator=torch.Generator().manual_seed(3)).to(torch.bfloat16)


def norm_gamma(h: int) -> torch.Tensor:
    """--next-norm: the RMSNorm weight, bf16 1 + N(0, 0.1), seed 4 (CPU)."""
    return (1 + 0.1 * torch.randn(h, generator=torch.Generator().manual_seed(4))).to(torch.bfloat16)


def set_x(dsts, x: torch.Tensor) -> None:
    """Copy x into every GPU's x tensor (dsts) and wait: the next launch reads it."""
    for d in dsts:
        d.copy_(x.to(d.device))
    for d in dsts:
        torch.cuda.synchronize(d.device)


def outputs(per_gpu: list, label: str) -> dict:
    """The outputs of a launch: per_gpu[g] = {name: GPU g's tensor}. Every GPU must hold the same bits (AssertionError).
    Returns GPU 0's on the host."""
    out = {k: v.cpu().clone() for k, v in per_gpu[0].items()}
    for g in range(1, len(per_gpu)):
        for k, v in per_gpu[g].items():
            assert torch.equal(v.cpu(), out[k]), f"{label}: {k} differs on GPU {g}"
    return out


def add_args(ap) -> None:
    ap.add_argument("--vary-x", action="store_true", help="each timed launch n gets its own x (seed 1000 + n; "
                    "ab_launches): --save-y / --compare-y then cover every launch (stale exchange values show)")
    ap.add_argument("--residual", choices=("zero", "random"), default="zero",
                    help="the residual: zero (default; y = the layer's output) or random (seed 3)")
    ap.add_argument("--next-norm", action="store_true", help="the kernel also writes x_next = RMSNorm(y) gamma "
                    "(row_rmsnorm, gamma seed 4, as the decoder's layers with the all-reduce inside)")


def _bits(v: torch.Tensor) -> torch.Tensor:
    return v.contiguous().view(torch.int16 if v.element_size() == 2 else torch.int32)


def save(path: str, launches: list) -> None:
    torch.save(launches, path)
    print(f"{len(launches)} launches' outputs ({', '.join(launches[0])}) saved to {path}")


def compare(path: str, launches: list, label: str) -> bool:
    """Every launch's outputs bit-equal to the saved run's (same launch count, same names). Prints one line:
    '<label>: per-launch outputs bit-equal to <path>: True|False (k of n launches ...)'."""
    base = torch.load(path)
    bad = []
    if len(base) != len(launches):
        bad.append(f"{len(launches)} launches, saved {len(base)}")
    for n, (a, b) in enumerate(zip(launches, base)):
        for k in a:
            if k not in b or b[k].shape != a[k].shape or not torch.equal(_bits(a[k]), _bits(b[k])):
                bad.append(f"launch {n} {k}")
    ok = not bad
    print(f"{label}: per-launch outputs bit-equal to {path}: {ok} ({len(launches)} launches, {', '.join(launches[0])}"
          f"{'; first differences: ' + ', '.join(bad[:4]) if bad else ''})")
    return ok
