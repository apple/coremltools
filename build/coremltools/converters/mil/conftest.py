#  Copyright (c) 2021, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

def pytest_make_parametrize_id(config, val, argname):
    if isinstance(val, (tuple, list)) and all(isinstance(elem, str) for elem in val):
        return "-".join(val)
    if isinstance(val, (int,bool,float)):
        return "[{}={}]".format(argname, val)
    if isinstance(val, (tuple, list)) and all(isinstance(elem, int) for elem in val):
        return "[{}=({})]".format(argname, ",".join([str(i) for i in val]))
    return None
