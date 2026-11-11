"""Which GPU toolchain the static megakernel is built for: "cuda" (NVIDIA, nvcc) or "hip" (AMD, hipcc).

The backend is the installed torch's (torch.version.hip set: "hip"), or MK_BACKEND=cuda|hip (e.g. to generate code on a machine
without a GPU). What differs per backend:
  template_dir    the kernel templates under include/mirage/: static_megakernel (CUDA) or static_megakernel_hip (HIP); the
                  same file names (core.cuh, host.cuh, runtime.cuh, config.cuh, tasks/<name>.cuh)
  wave            threads per warp / wavefront: 32 (CUDA) or 64 (AMD CDNA)
  clusters        CTA clusters exist (CUDA) or not (AMD): compiler.compile_plan keeps the plain launch on HIP
  smem_limit_kb   dynamic shared memory (LDS) per CTA
  ns_per_tick     the device clock the time stamps read: globaltimer is in ns (CUDA); s_memrealtime runs at 100 MHz on
                  gfx942 / gfx950 (10 ns per tick; host.cuh checks hipDeviceAttributeWallClockRate at init)
"""
import os

BACKENDS = ("cuda", "hip")


def backend() -> str:
    forced = os.environ.get("MK_BACKEND")
    if forced:
        if forced not in BACKENDS:
            raise ValueError(f"MK_BACKEND={forced}: one of {BACKENDS}")
        return forced
    import torch
    return "hip" if getattr(torch.version, "hip", None) else "cuda"


def template_dir(b: str = None) -> str:
    return {"cuda": "static_megakernel", "hip": "static_megakernel_hip"}[b or backend()]


def wave(b: str = None) -> int:
    return {"cuda": 32, "hip": 64}[b or backend()]


def has_clusters(b: str = None) -> bool:
    return (b or backend()) == "cuda"


def smem_limit_kb(b: str = None) -> int:
    """Dynamic shared memory (LDS) one CTA may use: 227 KB on B300 (sm_100), 160 KB on gfx950 (CDNA4 ISA guide)."""
    return {"cuda": 227, "hip": 160}[b or backend()]


def ns_per_tick(b: str = None) -> int:
    return {"cuda": 1, "hip": 10}[b or backend()]


def offload_arch() -> str:
    """The AMD GPU's target name, e.g. gfx950 (torch's gcnArchName without the feature suffixes like :sramecc+:xnack-)."""
    import torch
    return torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
