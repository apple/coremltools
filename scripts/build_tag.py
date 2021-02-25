#!/usr/bin/env python
#
# Copyright (c) 2021, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

# Construct a valid wheel build tag with ``git describe``.
#
# We do this with python because it is available and much more straightforward
# than doing it in the shell.

import subprocess
import os.path


def main():
    # ensure that no matter where this script is called, it reports git info
    # for the working directory where it has been checked out
    git_dir = os.path.dirname(__file__)
    version_str = subprocess.check_output(
        ["git", "-C", git_dir, "describe", "--tags"], text=True
    )
    build_tag = version_str.split("-", maxsplit=1)[1].replace("-", "_").strip()
    print(build_tag)


if __name__ == "__main__":
    main()
