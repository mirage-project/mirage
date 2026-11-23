"""Shape, paging, batching and numerical regressions for finite-mask attention."""

import argparse
import ctypes
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

import torch

import test_attention_softmax_hopper as common


POINTERS = ("qkv", "kc", "vc", "output", "q_scale", "k_scale",
            "cos", "sin", "qo", "pi", "pages", "last")
PROFILES = (
    (64, 1, 1, 8, 64),
    (64, 4, 8, 8, 80),
    (128, 1, 8, 8, 64),
    (128, 2, 8, 8, 64),
    (128, 4, 8, 8, 80),
    (128, 8, 4, 8, 64),
    (128, 16, 2, 8, 64),
    (256, 2, 4, 8, 64),
    (128, 2, 32, 8, 64),
    (128, 2, 8, 32, 64),
    (128, 2, 8, 64, 64),
)


def build(root, library, profile, finite_mask=True, rotary_dim=None):
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        raise RuntimeError("nvcc is required")
    fields = ("HEAD_DIM", "QO_PER_KV", "KV_HEADS", "MAX_TOKENS", "PAGE_STRIDE")
    command = [
        nvcc, str(Path(__file__).with_name("attention_finite_sizes_wrapper.cu")),
        "-o", str(library), "-shared", "-Xcompiler", "-fPIC", "-std=c++20",
        "-O3", "-use_fast_math", "-gencode", "arch=compute_90a,code=sm_90a",
        "-DMIRAGE_BACKEND_USE_CUDA", "-DMIRAGE_GRACE_HOPPER", "-DMPK_TARGET_CC=90",
        f"-I{root / 'include'}", f"-I{root / 'include/mirage/persistent_kernel'}",
        f"-I{common.ROOT / 'deps/cutlass/include'}", "--expt-relaxed-constexpr", "-lcuda",
        f"-DTEST_FINITE_MASK={int(finite_mask)}",
        f"-DTEST_ROTARY_DIM={profile[0] if rotary_dim is None else rotary_dim}",
        *[f"-DTEST_{name}={value}" for name, value in zip(fields, profile)],
    ]
    with library.with_suffix(".build.log").open("w") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    return command


def load(library):
    lib = ctypes.CDLL(str(library))
    lib.configure.restype = ctypes.c_int
    lib.enqueue.argtypes = [ctypes.c_void_p] * 12 + [ctypes.c_int] * 3 + [ctypes.c_void_p]
    lib.enqueue.restype = ctypes.c_int
    lib.resources.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    lib.resources.restype = ctypes.c_int
    assert lib.configure() == 0
    return lib


def norm_rotate(x, scale, cos, sin, normalize=True, rope=True):
    d = cos.shape[-1]
    unit = x
    if normalize:
        unit = (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True)
                                       + 1e-6)).to(x.dtype) * scale
    if not rope:
        return unit
    rotary = unit[..., :d]
    rotated = torch.cat((-rotary[..., d // 2:], rotary[..., :d // 2]), dim=-1)
    return torch.cat((rotary * cos + rotated * sin, unit[..., d:]), dim=-1)


def make_case(profile, active, context, mode, batch=1, rotary_dim=None):
    d, group, heads, capacity, stride = profile
    rotary_dim = d if rotary_dim is None else rotary_dim
    assert 0 < active <= capacity and context + active <= 64
    dtype, device = torch.bfloat16, "cuda"
    qkv = torch.randn((batch * capacity, heads, group + 2, d),
                      device=device, dtype=dtype) * 0.2
    kc = torch.randn((batch + 2, stride, heads, d), device=device, dtype=dtype) * 0.2
    vc = torch.randn_like(kc)
    q_scale = torch.ones(d, device=device, dtype=dtype)
    k_scale = torch.ones_like(q_scale)
    normalized = mode in ("normalized", "norm-only")
    rope = mode in ("normalized", "rope-only")
    if mode == "norm-only":
        q_scale = torch.randn_like(q_scale)
        k_scale = torch.randn_like(k_scale)
    if mode in ("large-values", "mixed-large-values"):
        qkv[:, :, :group + 1].zero_()
        kc.zero_()
        qkv[:, :, group + 1].fill_(2.0 ** 126)
        vc.fill_(2.0 ** 126)
        if mode == "mixed-large-values":
            vc[:, 1::2].neg_()
            for request in range(batch):
                qkv[request * active + (1 - context % 2):
                    (request + 1) * active:2, :, group + 1].neg_()
    frequency = 1 / (1e6 ** (torch.arange(0, rotary_dim, 2, device=device).float()
                            / rotary_dim))
    angle = torch.arange(64, device=device).float()[:, None] * frequency[None, :]
    angle = torch.cat((angle, angle), dim=-1)
    case = {
        "profile": profile, "active": active, "context": context, "mode": mode,
        "batch": batch, "normalized": normalized, "qkv": qkv, "kc": kc, "vc": vc,
        "operation": {"normalized": 1, "rope-only": 2, "norm-only": 3}.get(mode, 0),
        "rotary_dim": rotary_dim,
        "q_scale": q_scale, "k_scale": k_scale, "cos": angle.cos().to(dtype),
        "sin": angle.sin().to(dtype),
        "output": torch.full((batch * capacity, heads * group * d), float("nan"),
                             device=device, dtype=dtype),
        "qo": torch.arange(batch + 1, device=device, dtype=torch.int32) * active,
        "pi": torch.arange(batch + 1, device=device, dtype=torch.int32),
        "pages": torch.arange(batch, 0, -1, device=device, dtype=torch.int32),
        "last": torch.full((batch,), context + active, device=device, dtype=torch.int32),
    }
    expected, keys, values = [], [], []
    for request in range(batch):
        page = batch - request
        x = qkv[request * active:(request + 1) * active]
        q = x[:, :, :group].reshape(active, heads * group, d).transpose(0, 1)
        k = x[:, :, group].transpose(0, 1)
        v = x[:, :, group + 1].transpose(0, 1)
        if normalized or rope:
            cos = case["cos"][context:context + active]
            sin = case["sin"][context:context + active]
            q = norm_rotate(q, q_scale, cos, sin, normalized, rope)
            k = norm_rotate(k, k_scale, cos, sin, normalized, rope)
        keys.append(k.transpose(0, 1))
        values.append(v.transpose(0, 1))
        all_k = torch.cat((kc[page, :context].transpose(0, 1), k), dim=1)
        all_v = torch.cat((vc[page, :context].transpose(0, 1), v), dim=1)
        scores = (q @ all_k.repeat_interleave(group, 0).transpose(-1, -2)) * (d ** -0.5)
        future = (torch.arange(context + active, device=device)[None, :]
                  > context + torch.arange(active, device=device)[:, None])
        mask = torch.zeros_like(scores).masked_fill_(future, torch.finfo(dtype).min)
        p = (scores + mask).softmax(-1, dtype=torch.float32).to(dtype)
        expected.append((p @ all_v.repeat_interleave(group, 0)).transpose(0, 1)
                        .reshape(active, heads * group * d))
    case.update(reference=torch.cat(expected), keys=keys, values=values)
    return case


def enqueue(lib, case, poison=True):
    rc = lib.enqueue(*[case[name].data_ptr() for name in POINTERS], case["batch"],
                     case["operation"], int(poison),
                     torch.cuda.current_stream().cuda_stream)
    assert rc == 0, f"CUDA error {rc}"


def check_case(lib, case):
    active, context, batch = (case[key] for key in ("active", "context", "batch"))
    readonly = tuple(key for key in POINTERS if key not in ("kc", "vc", "output"))
    saved = {key: case[key].view(torch.uint8).clone() for key in readonly}
    old_k, old_v = case["kc"].clone(), case["vc"].clone()
    enqueue(lib, case)
    torch.cuda.synchronize()
    actual, expected = case["output"][:active * batch], case["reference"]
    def close(x, y):
        try:
            torch.testing.assert_close(x, y, rtol=0, atol=0.02, equal_nan=True)
            return True
        except AssertionError:
            return False
    finite = torch.isfinite(actual) & torch.isfinite(expected)
    error = ((actual[finite].float() - expected[finite].float()).abs().max().item()
             if finite.any() else 0.0)
    preserved = all(torch.equal(case[key].view(torch.uint8), value)
                    for key, value in saved.items())
    cache_correct = True
    for request in range(batch):
        page = batch - request
        cache_correct &= close(case["kc"][page, context:context + active],
                               case["keys"][request])
        cache_correct &= torch.equal(case["vc"][page, context:context + active],
                                     case["values"][request])
        old_k[page, context:context + active] = case["kc"][page, context:context + active]
        old_v[page, context:context + active] = case["vc"][page, context:context + active]
    preserved &= torch.equal(case["kc"].view(torch.uint8), old_k.view(torch.uint8))
    preserved &= torch.equal(case["vc"].view(torch.uint8), old_v.view(torch.uint8))
    preserved &= bool(torch.isnan(case["output"][active * batch:]).all())
    return {
        "profile": case["profile"], "active": active, "context": context,
        "mode": case["mode"], "batch": batch, "matches_reference": close(actual, expected),
        "rotary_dim": case["rotary_dim"],
        "max_finite_abs_error": error, "cache_correct": bool(cache_correct),
        "preserved": bool(preserved),
    }


def configurations(profile):
    capacity = profile[3]
    cases = [(active, length - active, mode, 1)
             for active in (1, 2, 4, 7, 8, 16, 31, 32, 33, 63, 64) if active <= capacity
             for length in (active, 15, 16, 31, 32, 63, 64) if length >= active
             for mode in ("random", "normalized")]
    cases += [(min(capacity, 8), 0, mode, batch)
              for mode in ("random", "normalized") for batch in (4, 16)]
    cases += [(min(capacity, 8), 64 - min(capacity, 8), mode, 1)
              for mode in ("large-values", "mixed-large-values")]
    cases += [(active, 64 - active, mode, 1)
              for active in (1, min(capacity, 8), capacity)
              for mode in ("rope-only", "norm-only")]
    return sorted(set(cases))


def passed(result):
    return all(result[name] for name in ("matches_reference", "cache_correct", "preserved"))


def run(output_dir, profiles):
    results = []
    for profile in profiles:
        library = output_dir / ("attention_" + "_".join(map(str, profile)) + ".so")
        build(common.ROOT, library, profile)
        lib = load(library)
        for config in configurations(profile):
            torch.manual_seed(793)
            try:
                result = check_case(lib, make_case(profile, *config))
            except RuntimeError as error:
                results.append({"profile": profile, "configuration": config,
                                "error": str(error)})
                (output_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
                raise
            results.append(result)
            if not passed(result):
                print(json.dumps(result), flush=True)
        (output_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        if profile in (PROFILES[0], PROFILES[3], PROFILES[7]):
            rotary_dim = profile[0] // 2
            partial = library.with_name(library.stem + "_partial.so")
            build(common.ROOT, partial, profile, rotary_dim=rotary_dim)
            lib = load(partial)
            for active in (1, 8):
                for mode in ("random", "normalized", "rope-only", "norm-only"):
                    torch.manual_seed(793)
                    result = check_case(lib, make_case(
                        profile, active, 64 - active, mode, rotary_dim=rotary_dim))
                    results.append(result)
                    if not passed(result):
                        print(json.dumps(result), flush=True)
            (output_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    failures = [item for item in results if not passed(item)]
    assert not failures, f"{len(failures)}/{len(results)} size cases failed"
    print(f"Passed {len(results)} finite-reference size cases.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--profile", type=int, action="append")
    args = parser.parse_args()
    if torch.cuda.get_device_capability() != (9, 0):
        raise SystemExit("This regression requires Hopper (sm90).")
    torch.set_num_threads(4)
    profiles = PROFILES if args.profile is None else [PROFILES[i] for i in args.profile]
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        run(args.output_dir, profiles)
    else:
        with tempfile.TemporaryDirectory(prefix="mirage-finite-sizes-") as directory:
            run(Path(directory), profiles)


if __name__ == "__main__":
    main()
