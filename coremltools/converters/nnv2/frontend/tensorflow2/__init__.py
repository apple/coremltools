from ....._deps import HAS_TF_2 as _HAS_TF_2

if _HAS_TF_2:
    from .load import load
