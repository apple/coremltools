#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from coremltools.converters.mil.mil.passes.name_sanitization_utils import NameSanitizer as _NameSanitizer

class TestNameSanitizer:

    def test_name_sanitizer(self):
        input_and_expected_strings = [("1", "_1"),
                                      ("abc", "abc"),
                                      ("*asdf", "_asdf"),
                                      ("*asd*f", "_asd_f"),
                                      ("0abc2", "_0abc2"),
                                      ("is8174 + 16", "is8174___16"),
                                      ("a:abc", "a_abc"),
                                      ("a.abc", "a_abc"),
                                      ("dense_2_1/BiasAdd", "dense_2_1_BiasAdd"),
                                      ("dense_2_1-BiasAdd", "dense_2_1_BiasAdd"),
                                      ("key:0", "key_0"),
                                    ]

        for i, in_and_out_str in enumerate(input_and_expected_strings):
            out = _NameSanitizer().sanitize_name(in_and_out_str[0])
            assert out == in_and_out_str[1]

