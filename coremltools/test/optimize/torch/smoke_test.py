#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import torch


class TestSmokeTest:
    def test_coremltools_optimize_torch_import(self):
        import coremltools.optimize.torch

    def test_model_optimizations(self):
        from coremltools.optimize.torch.palettization import DKMPalettizer, DKMPalettizerConfig
        from coremltools.optimize.torch.pruning import MagnitudePruner, MagnitudePrunerConfig
        from coremltools.optimize.torch.quantization import LinearQuantizer, LinearQuantizerConfig

        for OptCls, OptConfig, args in [
            (MagnitudePruner, MagnitudePrunerConfig, None),
            (DKMPalettizer, DKMPalettizerConfig, None),
            (LinearQuantizer, LinearQuantizerConfig, torch.randn(100)),
        ]:
            obj = OptCls(torch.nn.Identity(), OptConfig())
            obj.prepare(args)
            obj.finalize()

    def test_model_conversion(self, mnist_model, mnist_example_input):
        import coremltools.test.optimize.torch.conversion.conversion_utils as util

        converted_model = util.get_converted_model(mnist_model, mnist_example_input)
        assert converted_model is not None
