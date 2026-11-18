"""Experimental one-tile BF16 finite-mask reference compatibility."""

import ctypes
from pathlib import Path
import tempfile

import torch

import test_attention_softmax_hopper as common


def reference(case):
    active, context = case["active"], case["context"]
    q = case["qkv"][:active, :, :2].reshape(active, 16, 128).transpose(0, 1)
    k = case["qkv"][:active, :, 2].transpose(0, 1)
    if case["normalized"]:
        q = common.norm_rotate(q, case["q_scale"],
                               case["cos"][context:context + active],
                               case["sin"][context:context + active])
        k = common.norm_rotate(k, case["k_scale"],
                               case["cos"][context:context + active],
                               case["sin"][context:context + active])
    k = torch.cat((case["kc"][:context].transpose(0, 1), k), dim=1)
    v = torch.cat((case["vc"][:context].transpose(0, 1),
                   case["qkv"][:active, :, 3].transpose(0, 1)), dim=1)
    scores = (q @ k.repeat_interleave(2, 0).transpose(-1, -2)) * (128 ** -0.5)
    future = (torch.arange(context + active, device="cuda")[None, :]
              > context + torch.arange(active, device="cuda")[:, None])
    mask = torch.zeros_like(scores)
    mask.masked_fill_(future, torch.finfo(torch.bfloat16).min)
    probabilities = (scores + mask).softmax(-1, dtype=torch.float32).to(q.dtype)
    return (probabilities @ v.repeat_interleave(2, 0)).transpose(0, 1).reshape(active, 2048)


def make_case(active, context, mode):
    special = mode in ("future-overflow", "future-overflow-control", "future-mass")
    base_mode = "normalized" if special else mode
    single_overflow = mode in ("single-positive-overflow", "single-negative-overflow",
                              "single-rounding-overflow")
    if mode in ("large-values", "mixed-large-values", "finite-future-overflow") or single_overflow:
        base_mode = "random"
    case = common.make_case(active, context, base_mode, False)
    assert active + context <= 64
    if special:
        case["qkv"][:active].fill_(1)
        case["qkv"][0, :, 2].zero_()
        case["qkv"][1:active, :, 3].fill_(2)
        if mode == "future-overflow":
            case["q_scale"].fill_(float(2 ** 80))
            case["k_scale"].fill_(float(2 ** 80))
        elif mode == "future-mass":
            case["normalized"] = False
            case["qkv"][0, :, 2].fill_(-float("inf"))
            case["qkv"][1:active, :, 2].zero_()
    elif mode in ("large-values", "mixed-large-values"):
        case["qkv"][:active, :, :3].zero_()
        case["kc"][:context].zero_()
        case["qkv"][:active, :, 3].fill_(2.0 ** 126)
        case["vc"][:context].fill_(2.0 ** 126)
        if mode == "mixed-large-values":
            case["vc"][1:context:2].neg_()
            case["qkv"][(1 - context % 2):active:2, :, 3].neg_()
    elif mode == "finite-future-overflow":
        case["qkv"].zero_()
        case["qkv"][:active, :, :2, 0].fill_(2.0 ** 64)
        case["qkv"][1, :, 2, 0].fill_(2.0 ** 64)
        case["qkv"][:active, :, 3].fill_(1)
    elif single_overflow:
        case["qkv"].zero_()
        case["qkv"][:active, :, :2, 0].fill_(2.0 ** 64)
        case["qkv"][:active, :, 2, 0].fill_(2.0 ** 64)
        case["qkv"][:active, :, 3].fill_(1)
        if mode == "single-negative-overflow":
            case["qkv"][:active, :, :2, 0].neg_()
        elif mode == "single-rounding-overflow":
            case["qkv"][:active, :, :2, 0].fill_(torch.finfo(torch.bfloat16).max)
            case["qkv"][:active, :, :2, 1].fill_(2.0 ** 119)
            case["qkv"][:active, :, 2, :2].fill_(1)
    case["mode"] = mode
    return case


def configurations():
    cases = [(active, context, mode) for active in (1, 7, 8)
             for context in (0, 39, 56) for mode in ("random", "normalized")]
    cases += [(active, 0, mode) for active in (1, 7)
              for mode in ("sentinel", "below", "deep-negative", "normalized-below",
                           "nan", "positive-infinity", "negative-infinity")]
    cases += [(7, 0, mode) for mode in
              ("future-overflow", "future-overflow-control", "future-mass")]
    cases += [(active, 0, "large-values") for active in (7, 8)]
    cases += [(active, context, mode) for active, context in ((1, 63), (8, 56))
              for mode in ("large-values", "mixed-large-values")]
    cases += [(7, 0, "mixed-large-values")]
    cases += [(7, 0, "finite-future-overflow")]
    cases += [(1, 0, mode) for mode in ("single-positive-overflow",
                                       "single-negative-overflow",
                                       "single-rounding-overflow")]
    cases += [(0, 0, "random")]
    return cases


def check_case(lib, active, context, mode):
    case = make_case(active, context, mode)
    expected = reference(case)
    readonly = ("qkv", "q_scale", "k_scale", "cos", "sin", "qo", "pi", "pages", "last")
    saved = {name: case[name].view(torch.uint8).clone() for name in readonly}
    old_k, old_v = case["kc"].clone(), case["vc"].clone()
    common.enqueue(lib, case)
    torch.cuda.synchronize()
    actual = case["output"][:active]
    nan_match = torch.equal(torch.isnan(actual), torch.isnan(expected))
    both_finite = torch.isfinite(actual) & torch.isfinite(expected)
    error = ((actual[both_finite].float() - expected[both_finite].float()).abs().max().item()
             if both_finite.any() else 0.0)
    try:
        torch.testing.assert_close(actual, expected, rtol=0, atol=0.02, equal_nan=True)
        matches = True
    except AssertionError:
        matches = False
    preserved = all(torch.equal(case[name].view(torch.uint8), saved[name])
                    for name in readonly)
    for name, old in (("kc", old_k), ("vc", old_v)):
        preserved &= torch.equal(case[name][:context], old[:context])
        preserved &= torch.equal(case[name][context + active:], old[context + active:])
    preserved &= bool(torch.isnan(case["output"][active:]).all())
    return {
        "active": active, "context": context, "mode": mode,
        "matches_finite_reference": matches, "nan_mask_matches": nan_match,
        "max_finite_abs_error": error, "preserved": bool(preserved),
        "inputs_all_finite": all(bool(torch.isfinite(case[name]).all())
                                 for name in readonly),
        "output_finite_count": int(torch.isfinite(actual).sum()),
        "reference_finite_count": int(torch.isfinite(expected).sum()),
        "first_row_all_nan": bool(torch.isnan(actual[0]).all()) if active else None,
        "first_row_value": actual[0, 0].item() if active and torch.isfinite(actual[0, 0]) else None,
    }


def check_score_transform(lib):
    lib.finite_scores.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
    ]
    lib.finite_scores.restype = ctypes.c_int
    encodings = torch.arange(65536, device="cuda", dtype=torch.int32)
    # Cover every BF16 encoding and both sides of each FP32-to-BF16 rounding tie.
    inputs = torch.cat([
        ((encodings << 16) | tail).view(torch.float32)
        for tail in (0, 0x7fff, 0x8000, 0x8001)
    ]).repeat_interleave(2)
    outputs = torch.empty_like(inputs)
    for masks in range(4):
        mask = torch.tensor([torch.finfo(torch.bfloat16).min if masks & (1 << lane)
                             else 0 for lane in range(2)],
                            device="cuda", dtype=torch.bfloat16)
        expected = ((inputs.bfloat16() * (128 ** -0.5)).reshape(-1, 2)
                    + mask).flatten().float()
        rc = lib.finite_scores(inputs.data_ptr(), outputs.data_ptr(), inputs.numel(),
                               masks, torch.cuda.current_stream().cuda_stream)
        assert rc == 0
        torch.testing.assert_close(outputs, expected, rtol=0, atol=0, equal_nan=True)
        non_nan = ~torch.isnan(expected)
        assert torch.equal(outputs.view(torch.int32)[non_nan],
                           expected.view(torch.int32)[non_nan])
    return {"encodings": 65536, "rounding_tails": 4,
            "lane_mask_combinations": 4, "non_nan_bitwise_match": True, "passed": True}


def main():
    if torch.cuda.get_device_capability() != (9, 0):
        raise SystemExit("This prototype requires Hopper (sm90).")
    torch.set_num_threads(4)
    torch.manual_seed(793)
    with tempfile.TemporaryDirectory(prefix="mirage-finite-mask-") as directory:
        library = Path(directory) / "finite_mask.so"
        common.build(common.ROOT, library, finite_mask=True)
        lib = common.load(library)
        check_score_transform(lib)
        results = [check_case(lib, *config) for config in configurations()]
    failed = [result for result in results
              if not result["matches_finite_reference"] or not result["preserved"]]
    assert not failed, failed
    print(f"Passed {len(results)} finite-additive-mask cases.")


if __name__ == "__main__":
    main()
