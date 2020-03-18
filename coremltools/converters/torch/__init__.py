from ..._deps import HAS_TORCH as _HAS_TORCH

if _HAS_TORCH:
    from ._torch_converter import convert
