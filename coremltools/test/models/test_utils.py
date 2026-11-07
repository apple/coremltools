#  Copyright (c) 2025, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import sys

from coremltools.models.utils import _python_version


def test_python_version():
    """
    _python_version should return exactly three integer components, so that it
    works on pre-release interpreters such as "3.15.0rc2" as well.
    """
    version = _python_version()
    assert len(version) == 3
    assert all(isinstance(v, int) for v in version)
    assert version == (
        sys.version_info.major,
        sys.version_info.minor,
        sys.version_info.micro,
    )
