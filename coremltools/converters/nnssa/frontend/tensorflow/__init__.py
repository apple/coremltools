from coremltools._deps import HAS_TF as _HAS_TF

if _HAS_TF:
    from .load import load
