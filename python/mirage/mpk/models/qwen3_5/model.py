"""BF16 Qwen hybrid text graphs and preserved fixed-step numerical probes.

The offline graph takes token and step metadata from MPK and receives its
attention cache pages from the KV planner. The fixed-step probe retains its
private state for independent numerical comparison.
"""
from numbers import Integral
import os

import torch

from .batch import BatchRows, linear_grid, validate_batch_size, validate_offline_batch_size


def create_fixed_batch_kernel(batch_size=1):
    validate_batch_size(batch_size)
    if batch_size > 1 and torch.cuda.get_device_capability() != (10, 0):
        raise ValueError("Multiple requests currently require SM100 (B200)")
    from mirage import get_configurations_from_gpu
    from mirage.mpk.persistent_kernel import PersistentKernel

    workers, schedulers = get_configurations_from_gpu(0)
    args = PersistentKernel.get_default_init_parameters()
    args.update(test_mode=True, num_workers=workers, num_local_schedulers=schedulers,
                mpi_rank=0, world_size=1, max_num_batched_tokens=batch_size,
                max_num_batched_requests=batch_size, max_seq_length=1024,
                page_size=1024, max_num_pages=batch_size, use_fast_math=False)
    return PersistentKernel(**args)


def create_single_request_kernel():
    return create_fixed_batch_kernel(1)


class Qwen35Model:
    """Qwen3.8-27B dense or Qwen3.6-35B-A3B MoE text graph.

    ``offline=True`` binds token, position and KV metadata to an owning MPK
    offline runtime. Build through ``Qwen38Builder``, compile that runtime and
    call it once. The methods below retain the fixed-step numerical probe.

    ``step`` returns a borrowed BF16 logits buffer, valid until the next step.
    ``prefill`` defaults to whole-prompt CUDA execution, sharing weights and
    caches with MPK decode. ``attn_implementation`` selects matching eager
    (default) or SDPA attention in both paths. ``mode="stream"`` uses MPK for
    every input token. Compile with ``prefill_chunk_size=64`` and select
    ``mode="mpk"`` for batched prefill inside MPK on Hopper. Call ``reset``
    before a different request. All logits buffers are borrowed.

    On B200, ``batch_size=2..32`` compiles one fixed-batch decode graph. Pass
    rectangular host token lists [batch, length] to prefill/generate and one
    token per row to step; logits have shape [batch, vocab]. Prompt lengths
    must match and each request has private caches. Block prefill is vectorized
    PyTorch CUDA; stream prefill uses only MPK. Batch size 1 keeps its flat API.
    EOS stops each returned continuation independently; completed slots keep
    executing privately until the whole batch finishes. No padded prompts,
    request insertion/removal or multi-request MPK chunk prefill is supported.
    """
    max_seq_length = 1024
    batch_size = 1

    def __init__(self, config, weights, *, pk=None, attn_implementation="eager", batch_size=1,
                 offline=False, kv_cache_pairs=None):
        from .decoder import Qwen35DecoderProbe
        from .attention import validate_attention_implementation, validate_sdpa_runtime

        validate_attention_implementation(attn_implementation)
        validate_batch_size(batch_size)
        if offline:
            validate_offline_batch_size(batch_size)
        if offline and attn_implementation != "eager":
            raise ValueError("The Qwen offline graph currently supports eager attention only")

        moe = hasattr(config, "num_experts")
        geometry = (config.num_hidden_layers, config.hidden_size,
                    config.intermediate_size, config.vocab_size)
        if geometry != ((40, 2048, 512, 248320) if moe else (64, 5120, 17408, 248320)):
            raise ValueError("The complete graph supports Qwen3.8-27B and Qwen3.6-35B-A3B text only")
        if offline and (batch_size not in (1, 8) or pk is None or pk.test_mode or pk.mode != "offline"
                        or pk.world_size != 1
                        or pk.max_num_batched_tokens != batch_size or pk.max_num_batched_requests != batch_size
                        or pk.max_tokens_per_request != 1
                        or not 2 <= pk.max_seq_length <= 1024 or pk.max_num_pages != batch_size
                        or pk.page_size != 1024 or kv_cache_pairs is None):
            raise ValueError("Qwen offline requires B1 or B8, one token per request and iteration, "
                             "one 1024-token KV page per request, and an external offline kernel")
        if offline and set(kv_cache_pairs) != set(config.attention_layers):
            raise ValueError("Every offline attention layer must receive a planner cache page")
        if torch.cuda.device_count() != 1:
            raise ValueError("Expose exactly one allocated SM90/SM100 GPU")
        if torch.cuda.get_device_capability() not in ((9, 0), (10, 0)):
            raise ValueError("The Qwen hybrid graph requires SM90 or SM100")
        if batch_size > 1 and torch.cuda.get_device_capability() != (10, 0):
            raise ValueError("Multiple requests currently require SM100 (B200)")
        if moe and torch.cuda.get_device_capability() != (10, 0):
            raise ValueError("Qwen3.6 MoE routing currently requires SM100 (B200)")
        if attn_implementation == "sdpa":
            validate_sdpa_runtime()
        if pk is not None and not offline and (not pk.test_mode or pk.world_size != 1
                or pk.max_num_batched_tokens != batch_size or pk.max_num_batched_requests != batch_size
                or pk.max_seq_length != 1024):
            raise ValueError("A supplied kernel must use matching fixed-batch test_mode and capacity 1024")
        self.config = config
        self.batch_size = batch_size
        self.attn_implementation = attn_implementation
        self.offline = offline
        self.pk = pk = pk if pk is not None else create_fixed_batch_kernel(batch_size)
        # Recurrent FP32 differences can cross BF16 rounding boundaries.
        # Preserve CUDA exp/div precision in this correctness-first graph.
        pk.use_fast_math = False
        self.position = 0
        self._compiled = False
        self._closed = False
        self._prefill_graph = None
        self.tensors, self.dt = {}, {}
        self.batch = BatchRows(pk, batch_size, self.tensors, self.dt, "qwen35_model_")
        self.layers = []
        optimized_batch = (offline and batch_size == 8) or (
            not offline and not moe and batch_size in (8, 16) and
            os.environ.get(f"MPK_QWEN38_B{batch_size}_PARTITION", "1") == "1") or (
            moe and batch_size in (8, 16) and
            os.environ.get(f"MPK_QWEN36_B{batch_size}_PARTITION", "1") == "1")
        pk.qwen_gdn_optimized = bool(not offline and optimized_batch)
        # The B16 MoE graph can occasionally observe a nonfinite SM100 TMA
        # output even when the accumulator registers are finite. Guard each
        # ordinary projection before publishing its task-completion event.
        pk.qwen_linear_output_guard = bool(moe and batch_size == 16)

        def attach(name, tensor):
            tensor = tensor.detach().to("cuda").contiguous()
            self.tensors[name] = tensor
            self.dt[name] = pk.attach_input(tensor, name="qwen35_model_" + name)

        if offline:
            for name, source in (("token", "input_tokens"), ("next_token", "output_tokens")):
                tensor = pk.meta_tensors[source]
                if tensor.shape != (batch_size,) or tensor.dtype != torch.int64:
                    raise ValueError(f"Expected {batch_size} INT64 {source} elements")
                self.tensors[name] = tensor
                self.dt[name] = pk.attach_input(tensor, name="qwen38_offline_" + name)
                self.batch.alias(name, (1,))
            step = pk.meta_tensors["step"]
            if step.shape != (batch_size,) or step.dtype != torch.int32:
                raise ValueError(f"Expected {batch_size} INT32 runtime steps")
            self.tensors["runtime_step"] = step
            self.dt["runtime_step"] = pk.attach_input(step, name="qwen38_runtime_step")
            from .attention_reference import text_rope
            cos, sin = text_rope(config, torch.arange(1024))
            attach("runtime_rope_table", torch.stack((cos[0], sin[0]), dim=1))
            self.batch.empty("runtime_position", (1,), torch.int64)
            self.batch.empty("runtime_rope", (2, 64))
            self._task("qwen38_runtime_position", ["runtime_step", "runtime_rope_table",
                                                    "runtime_position", "runtime_rope"])
            position_pair = self.tensors["runtime_position"], self.dt["runtime_position"]
            rope_pair = self.tensors["runtime_rope"], self.dt["runtime_rope"]
        else:
            self.batch.empty("token", (1,), torch.int64)
            position_pair = rope_pair = None
        embed = weights["embed_tokens.weight"]
        head = weights["lm_head.weight"]
        for w in (embed, head):
            if w.shape != (config.vocab_size, config.hidden_size) or w.dtype != torch.bfloat16:
                raise ValueError("Expected BF16 embedding/LM head matching the configured vocabulary and width")
        attach("embedding", embed)
        self.batch.empty("hidden", (1, config.hidden_size))
        self._task("qwen35_embed", ["token", "embedding", "hidden"])
        pair = self.tensors["hidden"], self.dt["hidden"]
        for i in range(config.num_hidden_layers):
            layer = Qwen35DecoderProbe(config, weights, layer=i, pk=pk,
                                       input_pair=pair, name_prefix=f"qwen35_layer_{i}_",
                                       attn_implementation=attn_implementation, batch_size=batch_size,
                                       position_pair=position_pair, rope_pair=rope_pair,
                                       kv_cache_pair=None if kv_cache_pairs is None else kv_cache_pairs.get(i),
                                       optimized_b8=optimized_batch)
            self.layers.append(layer)
            pair = layer.tensors["decoder_output"], layer.dt["decoder_output"]
        self.tensors["last_hidden"], self.dt["last_hidden"] = pair
        self.batch.alias("last_hidden", (1, config.hidden_size))
        attach("norm_weight", weights["norm.weight"].float())
        self.batch.empty("normalized", (1, config.hidden_size))
        self._task("qwen35_norm", ["last_hidden", "norm_weight", "normalized"])
        if config.tie_word_embeddings:
            # The loader validates equality; share storage and the graph input.
            self.tensors["lm_head"] = self.tensors["embedding"]
            self.dt["lm_head"] = self.dt["embedding"]
        else:
            attach("lm_head", head)
        self.batch.empty("logits", (1, config.vocab_size))
        pk.linear_layer(self.dt["normalized"], self.dt["lm_head"], self.dt["logits"],
                        linear_grid(config.vocab_size, batch_size,
                                    optimized_b8=optimized_batch), (256, 1, 1))
        if not offline:
            self.batch.empty("next_token", (1,), torch.int64)
        self._task("qwen35_argmax", ["logits", "next_token"])

    @classmethod
    def from_pretrained(cls, model_path, *, weight_policy="hf_bf16", attn_implementation="eager",
                        shard_roots=(), batch_size=1):
        """Load native HF BF16 weights; choose eager or pinned H100 SDPA attention.

        The default remains eager for compatibility with existing references.
        SDPA decode runs a fused MPK CUDA task. Prefill defaults to PyTorch;
        the optional 64-token MPK graph is enabled at compile time.
        """
        from .attention import validate_attention_implementation
        from .weight_loader import Qwen35WeightLoader
        validate_attention_implementation(attn_implementation)
        validate_batch_size(batch_size)
        loader = Qwen35WeightLoader(model_path, shard_roots=shard_roots)
        return cls(loader.config, loader.load(weight_policy=weight_policy),
                   attn_implementation=attn_implementation, batch_size=batch_size)

    def _task(self, name, names):
        self.batch.task(name, names, [-1] * len(names))

    def compile(self, output_dir, *, prefill_chunk_size=None):
        """Compile decode, optionally a separate 64-token Hopper prefill graph.

        The graphs share weights. Prefill uses private scratch and shadow
        caches, committing to decode state only after all tiles complete.
        Compilation failure closes the model; create a new one to retry.
        """
        if self.offline:
            raise RuntimeError("Compile the owning offline MPK, then invoke it once")
        if self._closed or self._compiled:
            raise RuntimeError("Compile exactly once before closing")
        if prefill_chunk_size is not None and (isinstance(prefill_chunk_size, bool)
                or not isinstance(prefill_chunk_size, Integral) or prefill_chunk_size != 64):
            raise ValueError("MPK prefill_chunk_size must be 64")
        if prefill_chunk_size is not None and self.batch_size != 1:
            raise ValueError("MPK chunk prefill currently requires batch_size=1; use block or stream")
        if prefill_chunk_size is not None and hasattr(self.config, "num_experts"):
            raise ValueError("Qwen3.6 MoE supports stream prefill only")
        if prefill_chunk_size is not None and torch.cuda.get_device_capability() != (9, 0):
            raise ValueError("MPK chunk prefill currently requires an SM90 GPU")
        try:
            self.pk.compile(output_dir=str(output_dir))
            if prefill_chunk_size is not None:
                from pathlib import Path
                from .mpk_prefill import Qwen35ChunkPrefill
                self._prefill_graph = Qwen35ChunkPrefill(self)
                self._prefill_graph.compile(Path(output_dir) / "prefill")
        except Exception:
            self.close()
            raise
        self._compiled = True

    def _validate_tokens(self, tokens):
        tokens = list(tokens)
        if not tokens or any(isinstance(t, bool) or not isinstance(t, Integral) or
                             not 0 <= t < self.config.vocab_size for t in tokens):
            raise ValueError("Expected nonempty in-vocabulary integer token IDs")
        if self.position + len(tokens) > self.max_seq_length:
            raise ValueError("Sequence capacity exceeded (1024 tokens)")
        return tokens

    def _validate_batch_tokens(self, tokens, *, step=False):
        try:
            rows = list(tokens)
            if len(rows) != self.batch_size:
                raise ValueError("Expected exactly batch_size token rows")
            if step:
                rows = [[t] for t in rows]
            rows = [self._validate_tokens(row) for row in rows]
        except TypeError as exc:
            raise ValueError("Expected rectangular integer token rows [batch, length]") from exc
        if len({len(row) for row in rows}) != 1:
            raise ValueError("Fixed batches require equal prompt lengths; padding is unsupported")
        return [row[0] for row in rows] if step else rows

    @torch.inference_mode()
    def step(self, token):
        if self.offline:
            raise RuntimeError("The offline runtime owns token steps; call the MPK once")
        if not self._compiled or self._closed:
            raise RuntimeError("Compile the model before stepping; it must remain open")
        if self.batch_size == 1:
            token = self._validate_tokens([token])[0]
            self.tensors["token"].fill_(token)
        else:
            rows = self._validate_batch_tokens(token, step=True)
            self.tensors["token"].copy_(torch.tensor(rows, dtype=torch.int64,
                                                    device=self.tensors["token"].device))
        for layer in self.layers:
            mixer = layer.mixer
            mixer.tensors["position"].fill_(self.position)
            if "rope" in mixer.tensors:
                mixer.tensors["rope"].copy_(mixer.rope_table[self.position])
            if "split_config" in mixer.tensors:
                mixer.tensors["split_config"].copy_(mixer.split_table[self.position])
                mixer.tensors["attention_positions"].fill_(self.position)
        self.pk.init_request_func()
        self.pk()
        torch.cuda.synchronize()
        self.position += 1
        for layer in self.layers:
            layer.mixer.position = self.position
        return self.tensors["logits"]

    @staticmethod
    def _validate_prefill_mode(mode):
        if not isinstance(mode, str) or mode not in ("block", "stream", "mpk"):
            raise ValueError("prefill mode must be 'block', 'stream', or 'mpk'")

    def _prefill_block(self, tokens):
        if hasattr(self.config, "num_experts"):
            # The dense PyTorch block path has no MoE implementation. Reuse
            # the exact MPK one-token graph until a MoE chunk graph is added.
            for column in zip(*tokens) if self.batch_size > 1 else tokens:
                logits = self.step(column)
            return logits
        from .prefill import block_prefill
        return block_prefill(self, tokens)

    @torch.inference_mode()
    def prefill(self, tokens, *, mode="block"):
        """Append contiguous tokens; return the borrowed last-token logits.

        ``block`` uses PyTorch CUDA batch operations; ``mpk`` uses the compiled
        64-token MPK graph, including a partial final tile. Both use chunked
        GDN. ``stream`` uses only MPK one-token launches. Their floating-point
        paths may differ. One-token calls use MPK decode in every mode.
        """
        self._validate_prefill_mode(mode)
        if self.batch_size > 1:
            tokens = self._validate_batch_tokens(tokens)
            if not self._compiled or self._closed:
                raise RuntimeError("Compile the model before prefill; it must remain open")
            if mode == "mpk":
                raise ValueError("MPK chunk prefill requires batch_size=1; use block or stream")
            if mode == "block" and len(tokens[0]) > 1:
                return self._prefill_block(tokens)
            for column in zip(*tokens):
                logits = self.step(column)
            return logits
        tokens = self._validate_tokens(tokens)  # Validate all before mutation.
        if not self._compiled or self._closed:
            raise RuntimeError("Compile the model before prefill; it must remain open")
        if mode == "mpk" and len(tokens) > 1:
            if self._prefill_graph is None:
                raise RuntimeError("Compile with prefill_chunk_size=64 to enable MPK prefill")
            return self._prefill_graph.run(tokens)
        if mode == "block" and len(tokens) > 1:
            return self._prefill_block(tokens)
        for token in tokens:
            logits = self.step(token)
        return logits

    @torch.inference_mode()
    def generate(self, prompt_ids, max_new_tokens=16, eos_token_ids=None, *, prefill_mode="block"):
        """Greedy continuation, including the generated EOS token when reached.

        ``None`` uses the checkpoint's ``config.eos_token_id``. An integer or
        iterable of integers replaces that default for this request; an empty
        iterable disables EOS stopping for fixed-length measurements. The final
        returned token is not fed back into the recurrent/KV cache. Prefill
        defaults to whole-prompt CUDA with the selected attention backend; use
        ``prefill_mode="mpk"`` for the compiled chunk graph, or ``"stream"``
        for token-at-a-time MPK experiments.
        """
        if self.position:
            raise ValueError("Reset before generating a new request")
        self._validate_prefill_mode(prefill_mode)
        prompt_ids = (self._validate_tokens(prompt_ids) if self.batch_size == 1
                      else self._validate_batch_tokens(prompt_ids))
        length = len(prompt_ids) if self.batch_size == 1 else len(prompt_ids[0])
        if self.batch_size > 1 and prefill_mode == "mpk":
            raise ValueError("MPK chunk prefill requires batch_size=1; use block or stream")
        if (isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, Integral)
                or max_new_tokens < 0 or length + max(0, max_new_tokens - 1) > 1024):
            raise ValueError("Invalid generation length or sequence capacity exceeded")
        if eos_token_ids is None:
            eos_token_ids = self.config.eos_token_id
        try:
            eos = ((eos_token_ids,) if isinstance(eos_token_ids, Integral)
                   else tuple(eos_token_ids))
        except TypeError as exc:
            raise ValueError("EOS IDs must be in-vocabulary integers") from exc
        if any(isinstance(t, bool) or not isinstance(t, Integral) or
               not 0 <= t < self.config.vocab_size for t in eos):
            raise ValueError("EOS IDs must be in-vocabulary integers")
        eos = set(eos)
        if max_new_tokens == 0:
            return [] if self.batch_size == 1 else [[] for _ in range(self.batch_size)]
        self.prefill(prompt_ids, mode=prefill_mode)
        if self.batch_size > 1:
            result = [[] for _ in range(self.batch_size)]
            finished = [False] * self.batch_size
            for i in range(max_new_tokens):
                tokens = self.tensors["next_token"].tolist()
                for row, token in enumerate(tokens):
                    if not finished[row]:
                        result[row].append(token)
                        finished[row] = token in eos
                if all(finished) or i + 1 == max_new_tokens:
                    break
                # Finished slots continue privately until the batch completes;
                # their later tokens are discarded, with no cross-row state.
                self.step(tokens)
            return result
        result = []
        for i in range(max_new_tokens):
            token = int(self.tensors["next_token"].item())
            result.append(token)
            if token in eos or i + 1 == max_new_tokens:
                break
            self.step(token)
        return result

    @torch.inference_mode()
    def reset(self):
        if self._closed:
            raise RuntimeError("Model is closed")
        for layer in self.layers:
            layer.reset()
        self.position = 0

    def close(self):
        if not self._closed:
            if self._prefill_graph is not None:
                self._prefill_graph.close()
            self.pk.finalize()
            self._closed = True


class Qwen36MoeModel(Qwen35Model):
    """Qwen3.6-35B-A3B BF16 text model using the shared hybrid decoder."""

    @classmethod
    def from_pretrained(cls, model_path, *, weight_policy="hf_bf16", shard_roots=(),
                        batch_size=1):
        from .weight_loader import Qwen36WeightLoader
        validate_batch_size(batch_size)
        loader = Qwen36WeightLoader(model_path, shard_roots=shard_roots)
        return cls(loader.config, loader.load(weight_policy=weight_policy),
                   batch_size=batch_size)
