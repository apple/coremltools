#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

def is_pruner_prepared(optimizer):
    """
    Check if any layer to be compressed with `optimizer` has already
    been prepared for pruning by addition of pruning forward pre hooks
    """
    for name, submodule in optimizer._model.named_modules(remove_duplicate=True):
        config = optimizer._config.get_module_config(name, submodule)
        if config and hasattr(submodule, "pruning_method"):
            return True
    return False


def is_quant_prepared(optimizer):
    """
    Check if any layer to be compressed with `optimizer` has already
    been prepared for quantization or palettization by inserting fake quant layers
    """
    for name, submodule in optimizer._model.named_modules(remove_duplicate=True):
        config = optimizer._config.get_module_config(name, submodule)
        if config and hasattr(submodule, "weight_fake_quant"):
            return True
    return False


def is_quantized_module(module):
    """
    Check if a module has been quantized by inserting torch.ao.quantization.FakeQuantize layers
    """
    return hasattr(module, "weight_fake_quant") and not hasattr(
        module.weight_fake_quant, "fake_palett_enabled"
    )


def is_palettized_module(module):
    """
    Check if a module has been palettized by inserting coremltools.optimize.torch.palettization.FakePalettize layers
    """
    return hasattr(module, "weight_fake_quant") and hasattr(
        module.weight_fake_quant, "fake_palett_enabled"
    )
