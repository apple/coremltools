from ..._deps import _HAS_TORCH

if _HAS_TORCH:
    from ._torch_converter import convert
