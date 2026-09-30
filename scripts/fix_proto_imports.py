#!/usr/bin/env python
#
# Copyright (c) 2026, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

# Make the imports between protoc-generated ``*_pb2.py`` files relative, so that
# they work from inside the ``coremltools.proto`` package:
#
#     import FeatureTypes_pb2 as FeatureTypes__pb2
#     from FeatureTypes_pb2 import *
#
# become
#
#     from . import FeatureTypes_pb2 as FeatureTypes__pb2
#     from .FeatureTypes_pb2 import *
#
# This used to be done with ``python -m lib2to3 -f import``, but lib2to3 was
# removed in Python 3.13.

import re
import sys


def main():
    for path in sys.argv[1:]:
        with open(path, encoding="utf-8") as f:
            source = f.read()
        source = re.sub(r"^import (\w+_pb2) as ", r"from . import \1 as ", source, flags=re.M)
        source = re.sub(r"^from (\w+_pb2) import ", r"from .\1 import ", source, flags=re.M)
        with open(path, "w", encoding="utf-8") as f:
            f.write(source)


if __name__ == "__main__":
    main()
