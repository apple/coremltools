import coremltools
import sys, os
from _infer_shapes_nn_mlmodel import _infer_shapes
import operator


def inspect(model_path, output_txt_file):
    spec = coremltools.utils.load_spec(model_path)
    sys.stdout = open(os.devnull, "w")
    shape_dict = _infer_shapes(model_path)
    sys.stdout = sys.__stdout__

    types_dict = {}

    sys.stdout = open(output_txt_file, "w")

    nn = spec.neuralNetwork
    if spec.WhichOneof("Type") == "neuralNetwork":
        nn = spec.neuralNetwork
    elif spec.WhichOneof("Type") == "neuralNetworkRegressor":
        nn = spec.neuralNetworkRegressor
    elif spec.WhichOneof("Type") == "neuralNetworkClassifier":
        nn = spec.neuralNetworkClassifier
    else:
        raise ValueError("Only neural network model Type is supported")

    for i, layer in enumerate(nn.layers):
        print(
            "---------------------------------------------------------------------------------------------------------------------------------------------"
        )
        print(
            "{}: layer name = {}, layer type = ( {} ), \n inputs = \n {}, \n input shapes = {}, \n outputs = \n {}, \n output shapes = {} ".format(
                i,
                layer.name,
                layer.WhichOneof("layer"),
                ", ".join([x for x in layer.input]),
                ", ".join([str(shape_dict[x]) for x in layer.input]),
                ", ".join([x for x in layer.output]),
                ", ".join([str(shape_dict[x]) for x in layer.output]),
            )
        )

        layer_type = layer.WhichOneof("layer")
        if layer_type == "convolution" and layer.convolution.isDeconvolution:
            layer_type = "deconvolution"
        if layer_type in types_dict:
            types_dict[layer_type] += 1
        else:
            types_dict[layer_type] = 1

    print(
        "---------------------------------------------------------------------------------------------------------------------------------------------"
    )
    sorted_types_count = sorted(types_dict.items(), key=operator.itemgetter(1))
    print("Layer Type counts:")
    for i in sorted_types_count:
        print("{} : {}".format(i[0], i[1]))


if __name__ == "__main__":
    """
    Write a summary of the CoreML model to a text file.
    Summary includes layer name, type, input and output names and shapes. 

    Arguments
    ----------
    - path to the CoreML .mlmodel file
    - path to the output .txt file where the summary is written
    
    python inspect_mlmodel.py model.mlmodel text_file.txt

    """
    if len(sys.argv) != 3:
        raise ValueError(
            "Script expects two arguments. "
            + "Usage: python inspect_mlmodel.py /path/to/the/coreml/model.mlmodel /path/to/the/output/text/file.txt"
        )
    inspect(sys.argv[1], sys.argv[2])
