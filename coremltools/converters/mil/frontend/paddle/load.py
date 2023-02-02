#  Copyright (c) 2020, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import os.path as _os_path

import paddle as _paddle
import paddle.fluid as _fluid

from coremltools import _logger as logger

from .converter import PaddleConverter, paddle_to_mil_types


def load(model_spec, inputs, specification_version,
         debug=False, outputs=None, cut_at_symbols=None,
         **kwargs):
    """
    Convert PyPaddle model to mil CoreML format.

    Parameters
    ----------
    model_spec: String path to .pt file, or a PaddleScript object representing
        the model to convert.
    inputs: Can be a singular element or list of elements of the following form
        1. Any subclass of InputType
        2. paddle.Tensor (only shape and dtype will be used)
        3. list of (1. or 2.)
        Inputs are parsed in the flattened order that the model accepts them.
        If names are not specified: input keys for calling predict on the converted model
        will be internal symbols of the input to the graph.
        User can specify a subset of names.
    debug: bool, optional. Defaults to False.
        This flag should generally be False except for debugging purposes
        for diagnosing conversion errors. Setting this flag to True will
        print the list of supported and unsupported ops found in the model
        if conversion fails due to an unsupported op.
    outputs (optional): list[ct.InputType] or None
        list of either ct.TensorTypes or ct.ImageTypes (both of which are child classes of InputType)
        This is the value of the "outputs" argument, passed on by the user in "coremltools.convert" API.
    cut_at_symbols (optional): List of internal symbol name strings. Graph conversion will
        terminate once these symbols have been generated. For debugging use
        only.
    """
    [paddle_program, feed_var_names, fetch_vars]  = _pdmodel_from_model(model_spec)

    # if hasattr(pdmodel, 'training') and pdmodel.training:
    #     logger.warning("Model is not in eval mode. "
    #                      "Consider calling '.eval()' on your model prior to conversion")
    # if type(pdmodel) == _paddle.jit._script.RecursiveScriptModule:
    #     logger.warning("Support for converting Paddle Script Models is experimental. "
    #                      "If possible you should use a traced model for conversion.")

    # inputs = _convert_to_inputtype(inputs)
    converter = PaddleConverter(paddle_program, inputs, outputs, cut_at_symbols, specification_version)
    return _perform_paddle_convert(converter, debug)


def _pdmodel_from_model(model_spec):
    if isinstance(model_spec, str):
        _paddle.enable_static()
        model_dir = _os_path.abspath(model_spec)
        exe = _fluid.Executor(_fluid.CPUPlace())
        return _fluid.io.load_inference_model(
            model_dir, exe, model_filename="inference.pdmodel", params_filename="inference.pdiparams")
    # elif isinstance(model_spec, _paddle.jit.ScriptModule):
    #     return model_spec
    else:
        raise TypeError(
            "@model must the path of Paddle inference model, received: {}".format(
                type(model_spec)
            )
        )


def _perform_paddle_convert(converter, debug):
    try:
        prog = converter.convert()
    except RuntimeError as e:
        if debug and "convert function" in str(e):
            implemented, missing = converter.check_ops()
            print("the following model ops are IMPLEMENTED:")
            print("\n".join(["  " + str(x) for x in sorted(implemented)]))
            print("the following model ops are MISSING:")
            print("\n".join(["  " + str(x) for x in sorted(missing)]))
        raise e

    return prog
