"""Request metadata and KV pool sizing for one-call Qwen hybrid offline serving."""

import torch

from mirage import get_configurations_from_gpu
from mirage.mpk.kv_planner import resolve_pool_size
from mirage.mpk.persistent_kernel import PersistentKernel

from .builder import plan_qwen38_kv_cache
from .batch import validate_offline_batch_size


def create_offline_runtime(config, prompt_rows, new_tokens, *, ignore_eos=True):
    rows = [list(row) for row in prompt_rows]
    batch = len(rows)
    validate_offline_batch_size(batch)
    if batch not in (1, 8) or not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("Offline Qwen requires B1 or B8 equal-length nonempty prompts")
    if isinstance(new_tokens, bool) or not isinstance(new_tokens, int) or new_tokens < 1:
        raise ValueError("new_tokens must be positive")
    if batch == 8 and not ignore_eos:
        raise ValueError("B8 offline requires ignore_eos so request slots remain stable")
    prompt_length = len(rows[0])
    seq_length = prompt_length + new_tokens
    if seq_length > 1024:
        raise ValueError("Prompt plus generated tokens exceeds the 1024-token attention cache")
    if any(type(token) is not int or not 0 <= token < config.vocab_size
           for row in rows for token in row):
        raise ValueError("Expected in-vocabulary integer token IDs")

    if hasattr(config, "num_experts"):
        from .moe_builder import plan_qwen36_kv_cache
        plan = plan_qwen36_kv_cache(config)
    else:
        plan = plan_qwen38_kv_cache(config)
    resolve_pool_size(plan, max_num_pages=batch, max_seq_length=seq_length,
                      max_num_batched_requests=batch, max_num_batched_tokens=batch)
    tokens = torch.zeros((batch, seq_length), dtype=torch.int64, device="cuda")
    tokens[:, :prompt_length] = torch.tensor(rows, dtype=torch.int64, device="cuda")
    metadata = {
        "step": torch.zeros(batch, dtype=torch.int32, device="cuda"),
        "tokens": tokens,
        "input_tokens": torch.zeros(batch, dtype=torch.int64, device="cuda"),
        "output_tokens": torch.zeros(batch, dtype=torch.int64, device="cuda"),
        "num_new_tokens": torch.ones(batch, dtype=torch.int32, device="cuda"),
        "prompt_lengths": torch.full((batch,), prompt_length, dtype=torch.int32, device="cuda"),
        "qo_indptr_buffer": torch.zeros(batch + 1, dtype=torch.int32, device="cuda"),
        **plan.build_meta_tensors(max_seq_length=seq_length,
                                  max_num_batched_requests=batch),
    }
    workers, schedulers = get_configurations_from_gpu(0)
    # The graph has one token row per request. Without this limit the offline
    # scheduler fills its entire B8 token budget from request zero's prompt.
    mpk = PersistentKernel(
        mode="offline", world_size=1, mpi_rank=0,
        num_workers=workers, num_local_schedulers=schedulers, num_remote_schedulers=0,
        max_seq_length=seq_length, max_num_batched_requests=batch,
        max_num_batched_tokens=batch, max_tokens_per_request=1,
        max_num_pages=batch,
        kv_groups=plan.group_specs(), eos_token_id=-1 if ignore_eos else config.eos_token_id,
        meta_tensors=metadata, profiler_tensor=None, trace_name="",
        spec_decode_config=None, use_cutlass_kernel=False, use_fast_math=False)
    return mpk, plan, metadata


def offline_generated_ids(metadata, prompt_length, new_tokens, *, require_full=True):
    """Return all rows only after the offline call has completed."""
    steps = metadata["step"].tolist()
    tokens = metadata["tokens"]
    result = []
    for row, step in enumerate(steps):
        end = min(step + 1, tokens.shape[1])
        generated = tokens[row, prompt_length:end].tolist()
        if require_full and len(generated) != new_tokens:
            raise RuntimeError(f"Offline request {row} produced {len(generated)} of {new_tokens} tokens")
        result.append(generated)
    return result
