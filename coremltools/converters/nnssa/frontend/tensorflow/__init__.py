from coremltools._deps import _HAS_TF

if _HAS_TF:
    from .load import load
