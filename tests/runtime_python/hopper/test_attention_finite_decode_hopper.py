"""Single-query batches, mixed request lengths, and repeated cache updates."""

import argparse
import json
from pathlib import Path
import tempfile

import torch

import test_attention_finite_mask_sizes_hopper as sizes


def check_step(lib, case, reference_k, reference_v):
    d, group, heads, capacity, stride = case["profile"]
    offsets = case["qo"].tolist()
    lengths = case["last"].tolist()
    pages = case["pages"].tolist()
    expected, writes = [], []
    for request in range(case["batch"]):
        begin, end = offsets[request:request + 2]
        active = end - begin
        if not active:
            continue
        context = lengths[request] - active
        page = pages[request]
        x = case["qkv"][begin:end]
        q = x[:, :, :group].reshape(active, heads * group, d).transpose(0, 1)
        k = x[:, :, group].transpose(0, 1)
        v = x[:, :, group + 1].transpose(0, 1)
        normalize = case["operation"] in (1, 3)
        rope = case["operation"] in (1, 2)
        cos, sin = (case[name][context:context + active] for name in ("cos", "sin"))
        q = sizes.norm_rotate(q, case["q_scale"], cos, sin, normalize, rope)
        k = sizes.norm_rotate(k, case["k_scale"], cos, sin, normalize, rope)
        keys = torch.cat((reference_k[page, :context].transpose(0, 1), k), dim=1)
        values = torch.cat((reference_v[page, :context].transpose(0, 1), v), dim=1)
        scores = (q @ keys.repeat_interleave(group, 0).transpose(-1, -2)) * (d ** -0.5)
        future = (torch.arange(context + active, device="cuda")[None, :]
                  > context + torch.arange(active, device="cuda")[:, None])
        mask = torch.zeros_like(scores).masked_fill_(future, torch.finfo(q.dtype).min)
        probabilities = (scores + mask).softmax(-1, dtype=torch.float32).to(q.dtype)
        expected.append((probabilities @ values.repeat_interleave(group, 0))
                        .transpose(0, 1).reshape(active, heads * group * d))
        reference_k[page, context:context + active] = k.transpose(0, 1)
        reference_v[page, context:context + active] = v.transpose(0, 1)
        writes.append((page, context, context + active))
    expected = torch.cat(expected)
    readonly = {name: case[name].view(torch.uint8).clone()
                for name in sizes.POINTERS if name not in ("kc", "vc", "output")}
    old_k, old_v = case["kc"].clone(), case["vc"].clone()
    sizes.enqueue(lib, case)
    torch.cuda.synchronize()
    torch.testing.assert_close(case["output"][:offsets[-1]], expected,
                               rtol=0, atol=0.02, equal_nan=True)
    torch.testing.assert_close(case["kc"], reference_k, rtol=0, atol=0.02, equal_nan=True)
    assert torch.equal(case["vc"].view(torch.uint8), reference_v.view(torch.uint8))
    for page, begin, end in writes:
        old_k[page, begin:end] = case["kc"][page, begin:end]
        old_v[page, begin:end] = case["vc"][page, begin:end]
    assert torch.equal(case["kc"].view(torch.uint8), old_k.view(torch.uint8))
    assert torch.equal(case["vc"].view(torch.uint8), old_v.view(torch.uint8))
    assert all(torch.equal(case[name].view(torch.uint8), saved)
               for name, saved in readonly.items())
    assert bool(torch.isnan(case["output"][offsets[-1]:]).all())


def run(libraries, output):
    results = []
    for index in (1, 3, 6, 7, 10):
        profile = sizes.PROFILES[index]
        library = libraries / ("attention_" + "_".join(map(str, profile)) + ".so")
        if not library.exists():
            sizes.build(sizes.common.ROOT, library, profile)
        lib = sizes.load(library)
        for batch in (1, 4, 16):
            for mode in ("random", "normalized"):
                torch.manual_seed(793 + len(results))
                case = sizes.make_case(profile, 1, 0, mode, batch)
                reference_k, reference_v = case["kc"].clone(), case["vc"].clone()
                for context in range(64):
                    case["qkv"].normal_().mul_(0.2)
                    case["last"].fill_(context + 1)
                    case["output"].fill_(float("nan"))
                    check_step(lib, case, reference_k, reference_v)
                results.append({"profile": profile, "batch": batch, "mode": mode,
                                "generation_steps": 64, "passed": True})
        for mode in ("random", "normalized", "rope-only", "norm-only"):
            torch.manual_seed(971 + len(results))
            case = sizes.make_case(profile, 1, 0, mode, 4)
            case["qo"].copy_(torch.tensor([0, 1, 8, 8, 9], device="cuda",
                                         dtype=torch.int32))
            case["last"].copy_(torch.tensor([64, 7, 0, 24], device="cuda",
                                           dtype=torch.int32))
            check_step(lib, case, case["kc"].clone(), case["vc"].clone())
            results.append({"profile": profile, "mode": mode, "passed": True,
                            "request_lengths": [1, 7, 0, 1],
                            "contexts": [63, 0, 0, 23]})
        output.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Passed {sum(r.get('generation_steps', 1) for r in results)} "
          "batched generation/mixed-request steps.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--libraries", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if torch.cuda.get_device_capability() != (9, 0):
        raise SystemExit("This regression requires Hopper (sm90).")
    torch.set_num_threads(4)
    with tempfile.TemporaryDirectory(prefix="mirage-finite-decode-") as directory:
        run(args.libraries or Path(directory), args.output or Path(directory) / "results.json")


if __name__ == "__main__":
    main()
