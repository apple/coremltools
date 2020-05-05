from datetime import datetime
import sys

import torch

from exporting_utils import (
    ModelParams,
    exporter_parser,
    trace_and_save_model,
)


def _print_import_error(repo):
    print("Repo not found. Please run the following command to clone it:")
    print("\tgit clone {}".format(repo))


def deeplabv2_ctor():
    try:
        from deeplab_pytorch.libs.models import deeplabv2
    except ModuleNotFoundError:
        repo = "https://github.com/kazuto1011/deeplab-pytorch deeplab_pytorch"
        _print_import_error(repo)
        return None

    return deeplabv2.DeepLabV2(
        n_classes=21, n_blocks=[3, 4, 23, 3], atrous_rates=[6, 12, 18, 24]
    )


def deeplabv3_ctor():
    try:
        from deeplab_pytorch.libs.models import deeplabv3
    except ModuleNotFoundError:
        repo = "https://github.com/kazuto1011/deeplab-pytorch deeplab_pytorch"
        _print_import_error(repo)
        return None

    return deeplabv3.DeepLabV3(
        n_classes=21,
        n_blocks=[3, 4, 23, 3],
        atrous_rates=[6, 12, 18],
        multi_grids=[1, 2, 4],
        output_stride=8,
    )


def xception_ctor():
    try:
        from pretrained_models.pretrainedmodels.models import xception
    except ModuleNotFoundError:
        repo = "https://github.com/Cadene/pretrained-models.pytorch pretrained_models"
        _print_import_error(repo)
        return None

    return xception()


def tiramisu_ctor():
    try:
        from pytorch_tiramisu.models import tiramisu
    except ModuleNotFoundError:
        repo = "https://github.com/bfortuner/pytorch_tiramisu pytorch_tiramisu"
        _print_import_error(repo)
        return None

    return tiramisu.FCDenseNet67(n_classes=12)


def bert_ctor():
    try:
        from transformers import BertModel
    except ModuleNotFoundError:
        repo = "https://github.com/huggingface/transformers transformers"
        _print_import_error(repo)
        return None

    return BertModel.from_pretrained("bert-base-uncased", torchscript=True)


def gpt2lm_ctor():
    try:
        from transformers import GPT2LMHeadModel
    except ModuleNotFoundError:
        repo = "https://github.com/huggingface/transformers transformers"
        _print_import_error(repo)
        return None

    return GPT2LMHeadModel.from_pretrained("gpt2", torchscript=True)


"""
List of open source models that are available for export.
"""
MODELS = {
    "tiramisu": ModelParams((1, 3, 256, 256), tiramisu_ctor),
    "deeplabv2": ModelParams((1, 3, 256, 256), deeplabv2_ctor),
    "deeplabv3": ModelParams((1, 3, 512, 512), deeplabv3_ctor),
    "xception": ModelParams((1, 3, 229, 229), xception_ctor),
    "bert": ModelParams(
        (1, 11), bert_ctor, data_generator=lambda shape: torch.randint(1000, shape)
    ),
    "gpt2lm": ModelParams(
        (1, 15), gpt2lm_ctor, data_generator=lambda shape: torch.randint(1000, shape)
    ),
}


if __name__ == "__main__":
    parser = exporter_parser(
        description="Export open source end-to-end conversion test models.",
        model_names=list(MODELS.keys()),
    )
    args = parser.parse_args(sys.argv[1:])

    datestamp = datetime.now().strftime("%m-%d-%Y")

    for name in MODELS.keys():
        if args.all or getattr(args, name):
            trace_and_save_model(name, MODELS[name], args.path, datestamp)

    print("--FINISHED--")
