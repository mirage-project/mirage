"""Streaming weight-loading utilities — vLLM-style.

The primary entry point is :func:`safetensors_weights_iterator`. It yields
``(name, tensor)`` tuples one at a time from one or more safetensors files,
backed by ``safetensors.safe_open`` which mmaps each file. The tensor view
is zero-copy: the actual disk I/O happens when the consumer (a layer's
``weight_loader`` callback) reads bytes during ``copy_``. See
``/raid/user_data/zepengz/projects/vllm/WEIGHT_LOADING_NOTES.md`` §2 for the
details verified against safetensors==0.7.0.

The companion :func:`find_safetensors_files` resolves either a single
``model.safetensors`` file, an ``index.json``-style sharded checkpoint, or a
glob of ``model-*.safetensors`` files into a sorted list of paths.
"""

import glob
import json
import os
from typing import Iterator, List, Tuple

import torch
from safetensors import safe_open


def safetensors_weights_iterator(
    files: List[str],
) -> Iterator[Tuple[str, torch.Tensor]]:
    """Yield ``(name, mmap-view tensor)`` tuples in file-then-key order.

    Each yielded tensor is a zero-copy view into the file's mmap region.
    The bytes are not read until the caller reads them (typically via
    ``.narrow(...).copy_(...)`` into the target Parameter). This keeps peak
    CPU RSS bounded by file metadata + the size of whichever single tensor
    the caller is currently processing.
    """
    for path in files:
        with safe_open(path, framework="pt") as fh:
            for name in fh.keys():
                yield name, fh.get_tensor(name)


def find_safetensors_files(model_path: str) -> List[str]:
    """Resolve ``model_path`` to a sorted list of safetensors files.

    Handles three on-disk shapes:
      1. ``<model_path>/model.safetensors.index.json`` — sharded checkpoint
         with a JSON index; files listed in the index, deduplicated.
      2. ``<model_path>/model-*.safetensors`` — sharded without an index;
         lexicographic sort.
      3. ``<model_path>/model.safetensors`` — single-file checkpoint.

    Raises FileNotFoundError if no safetensors file is found.
    """
    if not os.path.isdir(model_path):
        raise FileNotFoundError(
            f"find_safetensors_files: not a directory: {model_path}"
        )
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        with open(index_path) as f:
            index = json.load(f)
        weight_map = index.get("weight_map", {})
        files = sorted({os.path.join(model_path, p) for p in weight_map.values()})
        if files:
            return files
    sharded = sorted(glob.glob(os.path.join(model_path, "model-*.safetensors")))
    if sharded:
        return sharded
    single = os.path.join(model_path, "model.safetensors")
    if os.path.isfile(single):
        return [single]
    raise FileNotFoundError(
        f"find_safetensors_files: no safetensors files under {model_path}"
    )
