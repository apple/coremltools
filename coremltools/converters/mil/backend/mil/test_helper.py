#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from coremltools.converters.mil.mil.passes.defs.preprocess import NameSanitizer as _NameSanitizer


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

    def test_name_sanitizer_collides_with_already_valid_name(self):
        # "a/b" gets sanitized into "a_b", so a subsequent (already valid) "a_b"
        # must not be handed back unchanged, otherwise two different vars end up
        # sharing a name.
        sanitizer = _NameSanitizer()
        assert sanitizer.sanitize_name("a/b") == "a_b"
        assert sanitizer.sanitize_name("a_b") == "a_b_0"

    def test_name_sanitizer_unique_suffix(self):
        # Names that all sanitize into the same string get "_0", "_1", ... appended,
        # as documented, instead of an ever growing suffix chain.
        sanitizer = _NameSanitizer()
        sanitized = [sanitizer.sanitize_name(name) for name in ("x/0", "x-0", "x.0", "x:0")]
        assert sanitized == ["x_0", "x_0_0", "x_0_1", "x_0_2"]
        assert len(set(sanitized)) == len(sanitized)
