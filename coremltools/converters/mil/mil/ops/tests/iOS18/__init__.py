#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import coremltools as ct
from coremltools.converters.mil.testing_reqs import backends_internal, clean_up_backends

backends = clean_up_backends(backends_internal, ct.target.iOS18)
