from ....._deps import HAS_TF_1 as _HAS_TF_1

if _HAS_TF_1:
    from .dialect_ops import * # register tf extension ops
    from .load import load
