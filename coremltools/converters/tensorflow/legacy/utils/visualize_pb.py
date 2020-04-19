import tensorflow as tf
from tensorflow.python.platform import gfile
import os
import shutil
import sys


def visualize(model_filename, log_dir):

    with tf.Session() as sess:
        with gfile.FastGFile(model_filename, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            g_in = tf.import_graph_def(graph_def)
    train_writer = tf.summary.FileWriter(log_dir)
    train_writer.add_graph(sess.graph)


if __name__ == "__main__":
    """
    Visualize the frozen TF graph using tensorboard. 

    Arguments
    ----------
    - path to the frozen .pb graph
    - path to a log directory for writing graph summary for visualization

    Usage
    ----------
    python visualize_pb.py frozen.pb /tmp/pb
    
    
    To kill a previous tensorboard process, use the following commands in the terminal
    ps aux | grep tensorboard
    kill PID
    """

    if len(sys.argv) != 3:
        raise ValueError(
            "Usage: python visualize_pb.py /path/to/frozen.pb /path/to/log/directory"
        )
    # load file
    visualize(sys.argv[1], sys.argv[2])
    os.system("tensorboard --logdir=" + sys.argv[2])
