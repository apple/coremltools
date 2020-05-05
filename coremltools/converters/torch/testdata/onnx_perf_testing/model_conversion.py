import torchvision.models as models
import torch
from torch import nn
import os
import tqdm
import logging
import coremltools
import tempfile
import numpy as np
import time

from datetime import date
from onnx_coreml import convert as onnx_convert

# NOTE: Useful function below to append to sys path to get imports from
# elsewhere on your machine.
# import sys
# sys.path.append("/Users/paul_curry/onnx_conversion/deeplab-pytorch")
try:
    from libs.models import DeepLabV2, DeepLabV3
    from pytorch.pretrainedmodels.models import xception
    from pytorch_tiramisu.models.tiramisu import FCDenseNet
    from CoreMLConversion_tests.perf_extension import TestRecipeGenerator
except:
    print(
        """please clone the following repos and make sure they are in your python path:\n
        https://github.com/kazuto1011/deeplab-pytorch\n
        https://github.com/Cadene/pretrained-models.pytorch\n
        https://github.com/bfortuner/pytorch_tiramisu\n
        """
    )


def DeepLabV2DefaultInstantiation():
    return DeepLabV2(n_classes=21, n_blocks=[3, 4, 23, 3], atrous_rates=[6, 12, 18, 24])


MODEL_TO_INPUT = {
    DeepLabV2DefaultInstantiation: torch.rand(1, 3, 224, 224),
    xception: torch.rand(1, 3, 224, 224),
    models.mobilenet_v2: torch.rand(1, 3, 224, 224),
    models.mnasnet1_0: torch.rand(1, 3, 224, 224),
    models.alexnet: torch.rand(1, 3, 224, 224),
    models.densenet161: torch.rand(1, 3, 224, 224),
    models.googlenet: torch.rand(1, 3, 224, 224),
    models.inception_v3: torch.rand(1, 3, 229, 229),
    models.resnet18: torch.rand(1, 3, 224, 224),
    models.shufflenet_v2_x1_0: torch.rand(1, 3, 224, 224),
    models.squeezenet1_0: torch.rand(1, 3, 224, 224),
    models.vgg19_bn: torch.rand(1, 3, 224, 224),
    models.vgg16: torch.rand(1, 3, 224, 224),
}

SAVE_PATH = "converted_models"
GENERATOR = TestRecipeGenerator(SAVE_PATH, "X")


def get_onnx_model(model, input_data):
    print("\tConverting via onnx pathway")
    example_outputs = model(input_data)

    start = time.time()
    try:
        with tempfile.TemporaryDirectory() as name:
            torch.onnx.export(
                model, input_data, name + "test.onnx", example_outputs=example_outputs,
            )
            onnx_mlmodel = onnx_convert(
                model=name + "test.onnx", minimum_ios_deployment_target="13"
            )
    except Exception as e:
        logging.error(
            "Exception getting onnx model with input size: {}".format(input_data.shape)
        )
        raise e
    end = time.time()
    print("\tTime to convert via onnx: {}s".format(end - start))
    return onnx_mlmodel


def get_name(model):
    today = date.today()
    stripped_classname = str(type(model))[8:-2].replace(".", "_")

    # NOTE: Edge case: We need to disciminate between the two VGGs
    if stripped_classname == "torchvision_models_vgg_VGG":
        num_modules = len([x for x in model.modules()])
        if num_modules == 64:
            stripped_classname = stripped_classname + "19"
        else:
            stripped_classname = stripped_classname + "16"

    model_name = "{}-onnx".format(stripped_classname)
    return model_name


def generate_recipe_and_save(model_name, mlmodel, input_data):
    coreml_inputs = {
        str(x): inp.numpy() for x, inp in zip(mlmodel.input_description, [input_data])
    }
    coreml_outputs = [str(x) for x in mlmodel.output_description]
    coreml_result = mlmodel.predict(coreml_inputs, useCPUOnly=True)
    coreml_result = coreml_result[coreml_outputs[0]]
    artifact_outputs = {
        str(x): inp for x, inp in zip(mlmodel.output_description, [coreml_result])
    }
    GENERATOR.generate_artifacts(
        mlmodel, model_name, coreml_inputs, artifact_outputs, benchmark_model=True,
    )


if __name__ == "__main__":
    for model_func, input_data in tqdm.tqdm(MODEL_TO_INPUT.items(), unit="model(s)"):
        print(
            " ================== Converting model: {} ==================".format(
                model_func
            )
        )
        model = model_func().eval()
        model_name = get_name(model)
        onnx_model = get_onnx_model(model, input_data)
        generate_recipe_and_save(model_name, onnx_model, input_data)
