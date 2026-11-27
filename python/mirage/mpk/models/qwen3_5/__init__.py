"""Qwen3.8 dense and Qwen3.6 MoE text checkpoint helpers.

The package can also be imported with python/mirage/mpk/models on PYTHONPATH
for CPU checkpoint inspection. MPK discovers builder.py separately.
"""

from .config import Qwen35TextConfig
from .weight_loader import Qwen35WeightLoader
from .moe_config import Qwen36MoeTextConfig
from .weight_loader import Qwen36WeightLoader

__all__ = ["Qwen35TextConfig", "Qwen35WeightLoader", "Qwen36MoeTextConfig", "Qwen36WeightLoader"]
