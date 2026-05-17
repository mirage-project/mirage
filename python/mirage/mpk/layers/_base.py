"""Base class for the MPK layer catalog.

Every catalog module inherits from :class:`MPKModule` (which is itself a
``torch.nn.Module``). The class defines the contract Phase-2 subagents
implement, the model-author audience reads, and Phase-3 composite
modules orchestrate.

The contract has three parts:

1. ``forward(self, ...)`` — a faithful PyTorch reference. The module
   owns its weights as ``nn.Parameter`` so ``state_dict()`` and
   ``load_state_dict()`` work the standard way. ``forward()`` runs in
   eager PyTorch with no MPK installed and no compile scope; it is the
   correctness oracle for the compiled path.

2. ``compile(self, ..., *, grid_dim=None, block_dim=None)`` — registers
   one or more MPK tasks into the active :class:`PersistentKernel` (read
   via :func:`mirage.mpk.context.current_pk` inside the body). Returns
   the output ``DTensor`` (or tuple of DTensors, mirroring the
   ``forward()`` signature). ``grid_dim`` and ``block_dim`` are
   keyword-only overrides; when omitted the leaf falls back to
   :meth:`auto_grid_dim` and :meth:`default_block_dim`.

3. ``auto_grid_dim(self, input_dt)`` — per-layer heuristic that returns
   a ``(x, y, z)`` tuple with product at most
   ``current_pk().num_workers``, respecting any kernel-specific
   alignment constraints (e.g., MMA-M=128 for MoE GEMMs). Each leaf
   overrides this; the base class only declares the signature.

``prefix`` is the vLLM/transformers convention for state_dict key
resolution. Setting ``prefix="model.layers.3.self_attn."`` makes the
module's weight load from ``state_dict["model.layers.3.self_attn.q_proj.weight"]``
without any custom loader plumbing.
"""

from typing import Iterable, Set, Tuple

import torch
import torch.nn as nn


GridDim = Tuple[int, int, int]
BlockDim = Tuple[int, int, int]


class MPKModule(nn.Module):
    """Base class for every layer in ``mirage.mpk.layers``.

    Subclasses MUST override:
        * ``forward`` — the PyTorch reference.
        * ``compile`` — the MPK task-registration path.
        * ``auto_grid_dim`` — per-layer grid heuristic.
    """

    def __init__(self, *, prefix: str = "") -> None:
        super().__init__()
        # ``prefix`` is a string path used both as the HF state_dict-key
        # prefix and as a unique name component for tensors attached to
        # MPK (see ``pk.attach_input(weight, name=f"{prefix}weight")``).
        # The empty-string default lets unit tests instantiate modules
        # standalone without picking a prefix.
        self.prefix = prefix

    def compile(self, *args, **kwargs):
        raise NotImplementedError(
            f"{type(self).__name__}.compile() is not implemented. "
            "Every MPKModule subclass must implement compile()."
        )

    def auto_grid_dim(self, *args, **kwargs) -> GridDim:
        raise NotImplementedError(
            f"{type(self).__name__}.auto_grid_dim() is not implemented. "
            "Either implement it on the subclass or pass grid_dim "
            "explicitly to compile()."
        )

    def default_block_dim(self) -> BlockDim:
        """Architecture-conditioned default block dim.

        128 threads per worker on Ampere (CC<90), 256 on Hopper/Blackwell —
        matches the values in ``include/mirage/persistent_kernel/`` headers.
        Subclasses with kernel-specific requirements (e.g., one warp per
        lane) override this.
        """
        from .. import context as _ctx
        pk = _ctx.current_pk()
        return (128, 1, 1) if pk.target_cc < 90 else (256, 1, 1)

    # ------------------------------------------------------------------
    # Weight loading (vLLM-style streaming)
    # ------------------------------------------------------------------
    def load_weights(
        self,
        weights: Iterable[Tuple[str, torch.Tensor]],
    ) -> Set[str]:
        """Default routing by HF state_dict key paths.

        ``weights`` is an iterable of ``(name, tensor)`` where ``name`` is
        the key path RELATIVE to ``self`` — for the top-level model these
        are the safetensors / HF state_dict keys; recursive calls receive
        names with the module's own dotted path already stripped.

        Routing rule: for each ``(name, tensor)`` we find the descendant
        :class:`MPKModule` whose dotted path (from
        ``self.named_modules()``) is the longest matching strict prefix of
        ``name``. The weight is then dispatched to that descendant's
        ``load_weights`` with the dotted path stripped (so the descendant
        sees a name relative to itself). When the deepest match is ``self``
        — i.e., the parameter belongs directly to this module — we look up
        ``name`` in ``self._parameters`` and invoke either the parameter's
        ``weight_loader`` callback (TP-aware leaves) or a plain ``copy_``.

        Note: routing uses the dotted path from ``named_modules()``, NOT
        ``self.prefix``. ``self.prefix`` is reserved for MPK kernel-tensor
        naming and may contain underscore separators distinct from HF's
        dotted keys.

        Override on composite modules that need fused-key handling (e.g.,
        ``Qwen3Attention`` mapping HF ``q_proj.weight`` →
        ``qkv_proj.weight`` with ``shard_id="q"``); the override applies a
        name-remap table, then delegates the rest to
        ``super().load_weights``.

        Returns the set of names (relative to ``self``) that were consumed;
        the top-level caller compares against the iterator's full set to
        detect missing/extra keys.
        """
        weights_list = list(weights)
        consumed: Set[str] = set()

        # Build a routing table from named_modules(): the dotted path
        # produced by named_modules is precisely the HF state_dict key
        # prefix (sans trailing dot). Sort by descending prefix length so
        # the first match is the deepest.
        routing = [(self, "")]
        for name, mod in self.named_modules():
            if isinstance(mod, MPKModule) and mod is not self and name:
                routing.append((mod, name + "."))
        routing.sort(key=lambda mp: -len(mp[1]))

        # Group weights by destination module.
        # value layout: (module, hf_prefix, list-of-(name, tensor))
        groups = {}
        for name, tensor in weights_list:
            for mod, prefix in routing:
                if prefix == "" or name.startswith(prefix):
                    if id(mod) not in groups:
                        groups[id(mod)] = (mod, prefix, [])
                    groups[id(mod)][2].append((name, tensor))
                    break

        for mod, prefix, group in groups.values():
            if mod is self:
                # Local parameters: looked up by relative name.
                for name, tensor in group:
                    if name in self._parameters and self._parameters[name] is not None:
                        param = self._parameters[name]
                        loader = getattr(param, "weight_loader", None)
                        if loader is not None:
                            loader(param, tensor)
                        else:
                            param.data.copy_(tensor)
                        consumed.add(name)
            else:
                # Recurse with prefix stripped so the descendant sees names
                # relative to itself.
                stripped = [(n[len(prefix):], t) for n, t in group]
                sub = mod.load_weights(stripped)
                for s in sub:
                    consumed.add(prefix + s)

        return consumed

    def process_weights(self) -> None:
        """Hook for post-load weight transforms.

        Runs AFTER all weights of this module's subtree have been loaded by
        :meth:`load_weights`. Default: recursively call ``process_weights``
        on every child :class:`MPKModule`. The default is a no-op at leaves.

        Override on modules that need post-load tensor transforms such as
        KV absorption (DeepSeek MLA), FP8 weight-scale TMA repack, or any
        cross-parameter fusion that depends on multiple children already
        being loaded.

        Overrides SHOULD call ``super().process_weights()`` first so leaf
        transforms run before composite-level transforms — this matches the
        natural order (a composite that consumes children's loaded params
        wants those leaves already in their post-load state).
        """
        for child in self.children():
            if isinstance(child, MPKModule):
                child.process_weights()
