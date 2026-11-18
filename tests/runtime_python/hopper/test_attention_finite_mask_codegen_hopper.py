"""Exercise finite-mask selection through Mirage's normal graph/codegen path."""

import argparse
import os
from pathlib import Path
import tempfile

import torch

from mirage.mpk.persistent_kernel import PersistentKernel
from mirage.kernel import get_key_paths

import test_attention_finite_mask_hopper as finite
import test_attention_finite_mask_sizes_hopper as sizes


def run_shapes(output_dir, active, context=0):
    output_dir = output_dir.resolve()
    params = PersistentKernel.get_default_init_parameters()
    params.update(
        mode="online_notoken" if context else "offline",
        test_mode=True,
        max_seq_length=64,
        max_num_batched_requests=1,
        max_num_batched_tokens=active,
        max_num_pages=1,
        page_size=64,
        meta_tensors={
            "prompt_lengths": torch.tensor([active], dtype=torch.int32, device="cuda"),
        },
    )
    pk = PersistentKernel(**params)
    profiles = (sizes.PROFILES[1], sizes.PROFILES[6], sizes.PROFILES[7])
    if active == 32:
        profiles = (sizes.PROFILES[9],)
    elif active == 1:
        profiles = tuple(sizes.PROFILES[i] for i in (1, 2, 3, 6, 7, 10))
    cases = []
    try:
        for index, profile in enumerate(profiles):
            d, group, heads, capacity, stride = profile
            for mode in ("random", "normalized"):
                case = sizes.make_case(profile, active, context, mode)
                if not case["normalized"]:
                    case["cos"].fill_(1)
                    case["sin"].zero_()
                tag = f"shape_{index}_{mode}"

                def attach(tensor, name):
                    return pk.attach_input(tensor, name=f"{tag}_{name}")

                k_cache = case["kc"][1:2, :64]
                v_cache = case["vc"][1:2, :64]
                pk.paged_attention_layer(
                    input=attach(case["qkv"].reshape(capacity, -1), "qkv"),
                    k_cache=attach(k_cache, "kc"),
                    v_cache=attach(v_cache, "vc"),
                    q_norm=attach(case["q_scale"], "qs"),
                    k_norm=attach(case["k_scale"], "ks"),
                    cos_pos_embed=attach(case["cos"], "cos"),
                    sin_pos_embed=attach(case["sin"], "sin"),
                    output=attach(case["output"], "output"),
                    grid_dim=(1, heads, 1),
                    block_dim=(256, 1, 1),
                    enable_qk_norm=case["normalized"],
                    finite_causal_mask=True,
                )
                cases.append((case, k_cache, v_cache, k_cache.clone(), v_cache.clone()))
        with tempfile.TemporaryDirectory(prefix="mirage-shape-codegen-") as directory:
            original_cwd = Path.cwd()
            try:
                os.chdir(directory)
                pk.compile(output_dir=str(output_dir))
            finally:
                os.chdir(original_cwd)
        if context:
            # Offline scheduling rebuilds metadata for a fresh prefill; use the
            # externally supplied batch mode to test an existing cache instead.
            pk.meta_tensors["qo_indptr_buffer"][1] = active
            pk.meta_tensors["paged_kv_indptr_buffer_0"][1] = 1
            pk.meta_tensors["paged_kv_last_page_len_buffer_0"].fill_(context + active)
        pk()
        torch.cuda.synchronize()
        for case, kc, vc, old_k, old_v in cases:
            torch.testing.assert_close(case["output"][:active], case["reference"],
                                       rtol=0, atol=0.02, equal_nan=True)
            torch.testing.assert_close(kc[0, context:context + active], case["keys"][0],
                                       rtol=0, atol=0.02, equal_nan=True)
            assert torch.equal(vc[0, context:context + active], case["values"][0])
            assert torch.equal(kc[:, :context], old_k[:, :context])
            assert torch.equal(vc[:, :context], old_v[:, :context])
            assert torch.equal(kc[:, context + active:], old_k[:, context + active:])
            assert torch.equal(vc[:, context + active:], old_v[:, context + active:])
    finally:
        pk.finalize()
    print(f"Passed {len(cases)} generated shape cases with {active} active tokens "
          f"and {context} cached tokens.")


def run(output_dir):
    torch.manual_seed(793)
    params = PersistentKernel.get_default_init_parameters()
    params.update(
        test_mode=True,
        max_seq_length=64,
        max_num_batched_requests=1,
        max_num_batched_tokens=8,
        max_num_pages=1,
        page_size=512,
        meta_tensors={
            "prompt_lengths": torch.tensor([7], dtype=torch.int32, device="cuda"),
        },
    )
    pk = PersistentKernel(**params)
    cases = []
    for enabled in (False, True):
        for mode in ("random", "normalized", "finite-future-overflow", "large-values"):
            case = finite.make_case(7, 0, mode)
            expected = finite.reference(case)
            tag = f"case_{int(enabled)}_{mode.replace('-', '_')}"

            def attach(tensor, name):
                return pk.attach_input(tensor, name=f"{tag}_{name}")

            arguments = dict(
                input=attach(case["qkv"].reshape(8, 4096), "qkv"),
                k_cache=attach(case["kc"].unsqueeze(0), "kc"),
                v_cache=attach(case["vc"].unsqueeze(0), "vc"),
                q_norm=attach(case["q_scale"], "q_scale"),
                k_norm=attach(case["k_scale"], "k_scale"),
                cos_pos_embed=attach(case["cos"], "cos"),
                sin_pos_embed=attach(case["sin"], "sin"),
                output=attach(case["output"], "output"),
                grid_dim=(1, 8, 1),
                block_dim=(256, 1, 1),
                enable_qk_norm=case["normalized"],
            )
            # Use identity rotary factors when testing unnormalized Q/K.
            if not case["normalized"]:
                case["cos"].fill_(1)
                case["sin"].zero_()
            pk.paged_attention_layer(**arguments, finite_causal_mask=enabled)
            readonly = ("qkv", "q_scale", "k_scale", "cos", "sin")
            snapshots = {name: case[name].view(torch.uint8).clone() for name in readonly}
            old_k, old_v = case["kc"].clone(), case["vc"].clone()
            cases.append((enabled, mode, case, expected, snapshots, old_k, old_v))

    original = pk.target_cc, pk.max_seq_length
    try:
        for cc, length in ((80, 64), (100, 64), (90, 65)):
            pk.target_cc, pk.max_seq_length = cc, length
            try:
                pk.paged_attention_layer(**arguments, finite_causal_mask=True)
            except ValueError:
                pass
            else:
                raise AssertionError(f"Unsupported finite-mask configuration: {cc}, {length}")
    finally:
        pk.target_cc, pk.max_seq_length = original

    try:
        pk.compile(output_dir=str(output_dir))
        pk()
        torch.cuda.synchronize()
        for enabled, mode, case, expected, snapshots, old_k, old_v in cases:
            actual = case["output"][:7]
            if enabled or mode in ("random", "normalized"):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0.02,
                                           equal_nan=True)
            elif mode == "finite-future-overflow":
                assert bool((actual[0] == 1).all())
                assert bool(torch.isnan(expected[0]).all())
            elif mode == "large-values":
                assert bool(torch.isinf(actual[4:]).all())
                assert bool(torch.isfinite(expected).all())
            for name, saved in snapshots.items():
                assert torch.equal(case[name].view(torch.uint8), saved)
            assert torch.equal(case["kc"][7:], old_k[7:])
            assert torch.equal(case["vc"][7:], old_v[7:])
            assert bool(torch.isnan(case["output"][7:]).all())
    finally:
        pk.finalize()
    print(f"Passed {len(cases)} generated attention cases and 3 option guards.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if Path(get_key_paths()[1]).resolve() != (sizes.common.ROOT / "include").resolve():
        raise RuntimeError("Import Mirage from this checkout so codegen uses its headers.")
    if torch.cuda.get_device_capability() != (9, 0):
        raise SystemExit("This regression requires Hopper (sm90).")
    torch.set_num_threads(4)
    if args.output_dir:
        run(args.output_dir)
        run_shapes(args.output_dir / "shapes_8", 8)
        run_shapes(args.output_dir / "shapes_32", 32)
        run_shapes(args.output_dir / "single_query", 1, 63)
    else:
        with tempfile.TemporaryDirectory(prefix="mirage-finite-codegen-") as directory:
            run(Path(directory))
            run_shapes(Path(directory) / "shapes_8", 8)
            run_shapes(Path(directory) / "shapes_32", 32)
            run_shapes(Path(directory) / "single_query", 1, 63)


if __name__ == "__main__":
    main()
