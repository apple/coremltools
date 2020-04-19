import tensorflow as tf
from tensorflow.core.framework import graph_pb2
import time
import operator
import sys


def inspect(model_pb, output_txt_file):
    graph_def = graph_pb2.GraphDef()
    with open(model_pb, "rb") as f:
        graph_def.ParseFromString(f.read())

    tf.import_graph_def(graph_def)

    sess = tf.Session()
    OPS = sess.graph.get_operations()

    ops_dict = {}

    sys.stdout = open(output_txt_file, "w")
    for i, op in enumerate(OPS):
        print(
            "---------------------------------------------------------------------------------------------------------------------------------------------"
        )
        print(
            "{}: op name = {}, op type = ( {} ), inputs = {}, outputs = {}".format(
                i,
                op.name,
                op.type,
                ", ".join([x.name for x in op.inputs]),
                ", ".join([x.name for x in op.outputs]),
            )
        )
        print("@input shapes:")
        for x in op.inputs:
            print("name = {} : {}".format(x.name, x.get_shape()))
        print("@output shapes:")
        for x in op.outputs:
            print("name = {} : {}".format(x.name, x.get_shape()))
        if op.type in ops_dict:
            ops_dict[op.type] += 1
        else:
            ops_dict[op.type] = 1

    print(
        "---------------------------------------------------------------------------------------------------------------------------------------------"
    )
    sorted_ops_count = sorted(ops_dict.items(), key=operator.itemgetter(1))
    print("OPS counts:")
    for i in sorted_ops_count:
        print("{} : {}".format(i[0], i[1]))


if __name__ == "__main__":
    """
    Write a summary of the frozen TF graph to a text file.
    Summary includes op name, type, input and output names and shapes. 
    
    Arguments
    ----------
    - path to the frozen .pb graph
    - path to the output .txt file where the summary is written
    
    Usage
    ----------
    python inspect_pb.py frozen.pb text_file.txt
    
    """
    if len(sys.argv) != 3:
        raise ValueError(
            "Script expects two arguments. "
            + "Usage: python inspect_pb.py /path/to/the/frozen.pb /path/to/the/output/text/file.txt"
        )
    inspect(sys.argv[1], sys.argv[2])
