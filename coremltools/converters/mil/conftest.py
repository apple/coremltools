#  Copyright (c) 2021, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause


def pytest_make_parametrize_id(config, val, argname):
    '''
    This function is a hook into pytest. It generates a user friendly string
    representation of the parameterized values.
    '''
    return "{}={}".format(argname, str(val))
