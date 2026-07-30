from typing import Dict, Type, Iterable, Union

_MODEL_BUILDERS: Dict[str, Type] = {}

def register_model_builder(*names: Union[str, Iterable[str]]):
    """
    Decorator to register a model builder under one or multiple names.
    Usage:
      @register_model_builder("qwen3", "Qwen3", "qwen")
      class Qwen3Builder: ...
    """
    # Flatten names in case a single iterable was passed
    if len(names) == 1 and isinstance(names[0], (list, tuple, set)):
        reg_names = list(names[0])
    else:
        reg_names = list(names)

    def decorator(cls):
        for n in reg_names:
            _MODEL_BUILDERS[n] = cls
        return cls

    return decorator

def get_builder(name: str):
    builder = _MODEL_BUILDERS.get(name)
    if builder is None:
        raise ValueError(f"Unknown builder: {name}. Registered: {list(_MODEL_BUILDERS.keys())}")
    return builder