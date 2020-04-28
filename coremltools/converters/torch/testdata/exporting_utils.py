import argparse
import os

import torch

"""
Convenience class for storing the tracing data shape and model constructor.
"""


class ModelParams:
    def __init__(self, input_shape, ctor, data_generator=None):
        self.input_shape = input_shape
        self.ctor = ctor
        self.data_generator = (
            data_generator if data_generator is not None else torch.rand
        )


def exporter_parser(description, model_names):
    parser = argparse.ArgumentParser(allow_abbrev=False, description=description)
    for name in model_names:
        parser.add_argument("--{}".format(name), action="store_true")
    parser.add_argument(
        "-a", "--all", action="store_true", help="export all available models"
    )
    parser.add_argument(
        "-p", "--path", default="./", help="path to save exported models"
    )
    return parser


def trace_and_save_model(name, params, path, datestamp):
    """ Construct, trace, and save a PyTorch model. Will be saved to
        "<path>/<name>_<datestamp>.pt"

        Arguments
            name: string name of the model
            params: ModelParams object with tracing shape and model constructor
            path: path to place the saved model
            datestamp: string appended to model name for versioning
    """

    model = params.ctor()
    if model is None:
        print("***Couldn't export {}.".format(name))
        return

    model.eval()
    test_data = params.data_generator(params.input_shape)
    trace = torch.jit.trace(model, test_data)
    filename = os.path.join(path, "{}_{}.pt".format(name, datestamp))
    trace.save(filename)
    print(">>>Saved {} to {}".format(name, filename))
