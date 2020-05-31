# Copyright (c) 2020, Apple Inc. All rights reserved.
SPACES = "  "

from .block import curr_block, Block, Function
from .input_type import *
from .operation import *
from .program import *
from .var import *

from .builder import Builder
from .ops.defs._op_reqs import register_op
