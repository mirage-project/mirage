"""DeepSeek V3 PyTorch reference (built on official inference code).

Uses /home/muhengl/DeepSeek-V3/inference/model.py (clone of
github.com/deepseek-ai/DeepSeek-V3) as the ground-truth math
implementation. We add: (a) selective-layer construction so we can
test a subset, (b) per-layer hidden-state forward hooks, (c)
HF-checkpoint loader.

Entry point: `runner.run_reference(model_path, prompt, ...)`.
"""
