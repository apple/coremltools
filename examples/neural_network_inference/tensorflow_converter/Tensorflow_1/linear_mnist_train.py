from __future__ import print_function
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data # Import MINST data

def linear_model(x):
  # x is the image input
  # mnist data image of shape 28*28=784

  # Set model weights
  W = tf.Variable(tf.zeros([784, 10]))
  b = tf.Variable(tf.zeros([10]))

  # Construct model
  pred = tf.nn.softmax(tf.matmul(x, W) + b)

  # Return the last op
  return pred


def train():
  mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
  # instantiate the model in the default graph
  x = tf.placeholder(tf.float32, [None, 784])

  print('image_input: ', x)  
  #print 'image_input: ', x
  pred = linear_model(x)
  #print 'pred output:', pred

  print('pred output:', pred)

  # Add training components to it
  # 0-9 digits recognition => 10 classes
  y = tf.placeholder(tf.float32, [None, 10])

  # Define training hyper-parameters
  learning_rate = 0.01
  training_epochs = 25
  batch_size = 100
  display_step = 1

  # Define Cross Entropy loss
  cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
  # Use Gradient Descent
  optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

  # Initialize the variables (i.e. assign their default value)
  init = tf.global_variables_initializer()

  # Use a saver to save checkpoints
  saver = tf.train.Saver()
  # Training starts here
  with tf.Session() as sess:
    sess.run(init)
    # Training cycle
    for epoch in range(training_epochs):
      avg_cost = 0.
      total_batch = int(mnist.train.num_examples/batch_size)
      # Loop over all batches
      for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        # Fit training using batch data
        _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                       y: batch_ys})
        # Compute average loss
        avg_cost += c / total_batch
      # Display logs per epoch step
      if (epoch+1) % display_step == 0:
        print(("Epoch: {:04d} , cost= {:.9f}").format(epoch+1,avg_cost))
        #print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost)
    print('Training Done. Now save the checkpoint...')
    #print 'Training Done. Now save the checkpoint...'
    save_dir = './checkpoints'
    save_path = os.path.join(save_dir, 'model.ckpt')
    if not os.path.exists(save_dir):
      os.mkdir(save_dir)
    save_path = saver.save(sess, save_path)
    tf.train.write_graph(sess.graph, './', 'model.pbtxt')


if __name__ == '__main__':

  # Read the data
  train()

