#!/usr/bin/env python3

import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import datetime

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    meta_graph = tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    input_tensor = sess.graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob_tensor = sess.graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out_tensor = sess.graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out_tensor = sess.graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out_tensor = sess.graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return input_tensor, keep_prob_tensor, layer3_out_tensor, layer4_out_tensor, layer7_out_tensor
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function

    debug_ops = []

    debug_ops.append(tf.Print(vgg_layer3_out, [tf.shape(vgg_layer3_out)], message="vgg_layer3_out: ", summarize=10, first_n=1))
    debug_ops.append(tf.Print(vgg_layer4_out, [tf.shape(vgg_layer4_out)], message="vgg_layer4_out: ", summarize=10, first_n=1))
    debug_ops.append(tf.Print(vgg_layer7_out, [tf.shape(vgg_layer7_out)], message="vgg_layer7_out: ", summarize=10, first_n=1))

    x = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, strides=(1,1), padding='SAME', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    debug_ops.append(tf.Print(x, [tf.shape(x)], message="post-1x1-convolution: ", summarize=10, first_n=1))

    x = tf.layers.conv2d_transpose(x, num_classes, 4, strides=(2, 2), padding='SAME', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    debug_ops.append(tf.Print(x, [tf.shape(x)], message="transpose1: ", summarize=10, first_n=1))

    skip_4 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, strides=(1,1), padding='SAME', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    x = tf.add(x, skip_4)
    x = tf.layers.batch_normalization(x)
    x = tf.layers.conv2d_transpose(x, num_classes, 4, strides=(2, 2), padding='SAME', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    debug_ops.append(tf.Print(x, [tf.shape(x)], message="transpose2: ", summarize=10, first_n=1))

    skip_3 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, strides=(1,1), padding='SAME', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    x = tf.add(x, skip_3)
    x = tf.layers.batch_normalization(x)
    x = tf.layers.conv2d_transpose(x, num_classes, 16, strides=(8, 8), padding='SAME', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    debug_ops.append(tf.Print(x, [tf.shape(x)], message="transpose3: ", summarize=10, first_n=1))


    return x, debug_ops
# tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    tf.summary.image("prediction", tf.expand_dims(255*tf.nn.softmax(nn_last_layer[:,:,:, 1]), 3))
    tf.summary.image("label", tf.expand_dims(255*tf.nn.softmax(correct_label[:,:,:, 1]), 3))

    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    correct_label = tf.reshape(correct_label, (-1, num_classes))
    adam = tf.train.AdamOptimizer(learning_rate)
    with tf.name_scope('loss'):
        cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label))
        tf.summary.scalar('cross_entropy_loss', tf.reduce_max(cross_entropy_loss))
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(correct_label, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        variable_summaries(correct_prediction)
    train_op = adam.minimize(cross_entropy_loss)

    return logits, train_op, cross_entropy_loss
# tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, get_batches_fn_test, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate, debug_ops, merged, train_writer, test_writer):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param get_batches_fn_test: Function to get batches of testing data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    :param debug_ops: Ops to run for debugging
    :param merged: Merged variable_summaries op
    :param train_writer: Train logs writer
    :param test_writer: Test logs writer
    """

    # TODO: Implement function
    step = 1
    for epoch in range(epochs):
        batch = 1
        test_batches = get_batches_fn_test(batch_size)
        for image, label in get_batches_fn(batch_size):
            #training
            summary, loss, _, _ = sess.run([merged, cross_entropy_loss, train_op, debug_ops], feed_dict={
                input_image: image,
                correct_label: label,
                keep_prob: 0.5,
            })
            print("step{}: epoch {}, batch {}, batch-loss: {}".format(step, epoch+1, batch, loss))
            train_writer.add_summary(summary, step)


            if False and batch % 10 == 0:
                try:
                    test_image, test_label = next(test_batches)
                except StopIteration:
                    test_batches = get_batches_fn_test(batch_size)
                    test_image, test_label = next(test_batches)

                    summary, loss = sess.run([merged, cross_entropy_loss], feed_dict={
                        input_image: test_image,
                        correct_label: test_label,
                        keep_prob: 1.0,
                    })
                test_writer.add_summary(summary, step)
                print("  test-batch-loss: {}".format(loss))

            batch += 1
            step += 1
# tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    learning_rate = 0.001
    epochs = 6 # 6 was enough for him
    labels_tensor = tf.placeholder(tf.float32, [None, None, None, num_classes])
    batch_size = 10

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)
        get_batches_fn_test = helper.gen_batch_function(os.path.join(data_dir, 'data_road/testing'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        input_tensor, keep_prob_tensor, layer3_out_tensor, layer4_out_tensor, layer7_out_tensor = load_vgg(sess, vgg_path)
        output_layer, debug_ops = layers(layer3_out_tensor, layer4_out_tensor, layer7_out_tensor, num_classes)
        logits, train_op, cross_entropy_loss = optimize(output_layer, labels_tensor, learning_rate, num_classes)

        # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('logs/{:%Y-%m-%d--%H-%M-%S}-train'.format(datetime.datetime.now()), sess.graph)
        test_writer = tf.summary.FileWriter('logs/{:%Y-%m-%d--%H-%M-%S}-test'.format(datetime.datetime.now()))

        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        # TODO: Train NN using the train_nn function
        train_nn(sess, epochs, batch_size, get_batches_fn, get_batches_fn_test,
                     train_op, cross_entropy_loss, input_tensor,
                     labels_tensor, keep_prob_tensor, learning_rate, debug_ops, merged, train_writer, test_writer)

        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob_tensor, input_tensor)

        # OPTIONAL: Apply the trained model to a video

# tf.reset_default_graph()
# run()
if __name__ == '__main__':
    run()
