#!/usr/bin/env python3

import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import datetime
import mobilenet_v1
from tensorflow.python.framework import graph_util

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


def layers(end_points, num_classes, debug_ops=[]):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the mobilenet layers.
    :param end_points: dictionary of names to tensors for various activations in the graph
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """

    x = tf.layers.conv2d(end_points["Conv2d_13_pointwise"], num_classes, 1, strides=(1,1), padding='SAME',
                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                         kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    x = tf.layers.batch_normalization(x)
    debug_ops.append(tf.Print(x, [tf.shape(x)], message="post-1x1-convolution: ", summarize=10, first_n=1))

    x = tf.layers.conv2d_transpose(x, num_classes, 3, strides=(2, 2), padding='SAME',
                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                         kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    x = tf.layers.batch_normalization(x)
    debug_ops.append(tf.Print(x, [tf.shape(x)], message="transpose1: ", summarize=10, first_n=1))

    skip_11 = tf.layers.conv2d(end_points["Conv2d_11_pointwise"], num_classes, 1, strides=(1,1), padding='SAME',
                       kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                       kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    x = tf.add(x, skip_11)
    x = tf.layers.batch_normalization(x)
    x = tf.layers.conv2d_transpose(x, num_classes, 3, strides=(2, 2), padding='SAME',
                       kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                       kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    debug_ops.append(tf.Print(x, [tf.shape(x)], message="transpose2: ", summarize=10, first_n=1))

    skip_4 = tf.layers.conv2d(end_points["Conv2d_4_pointwise"], num_classes, 1, strides=(1,1), padding='SAME',
                       kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                       kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    x = tf.add(x, skip_4)
    x = tf.layers.batch_normalization(x)
    x = tf.layers.conv2d_transpose(x, num_classes, 3, strides=(2, 2), padding='SAME',
                       kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                       kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    debug_ops.append(tf.Print(x, [tf.shape(x)], message="transpose3: ", summarize=10, first_n=1))

    x = tf.layers.conv2d_transpose(x, num_classes, 6, strides=(4, 4), padding='SAME',
                       kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                       kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    return x, debug_ops


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    tf.summary.image("prediction", tf.expand_dims(255*tf.nn.softmax(nn_last_layer)[:,:,:, 1], 3))
    tf.summary.image("label", tf.expand_dims(255*tf.nn.softmax(correct_label)[:,:,:, 1], 3))

    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    correct_label = tf.reshape(correct_label, (-1, num_classes))
    adam = tf.train.AdamOptimizer(learning_rate)
    with tf.name_scope('loss'):
        cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label))
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_constant = 0.01
        cross_entropy_loss = cross_entropy_loss + reg_constant * sum(reg_losses)
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


    cropped_image_op, cropped_label_op = helper.random_crop_and_pad_image_and_labels_batch(input_image, correct_label, 160, 160)

    step = 1
    for epoch in range(epochs):
        batch = 1
        test_batches = get_batches_fn_test(batch_size)
        for image, label in get_batches_fn(batch_size):
            # image, label = sess.run([cropped_image_op, cropped_label_op], feed_dict={
            #     input_image: image,
            #     correct_label: label,
            # })

            # print(image, label)
            #training
            # _ = sess.run([debug_ops], feed_dict={
            #     input_image: image,
            #     correct_label: label,
            #     # keep_prob: 0.5,
            # })
            summary, loss, _, _ = sess.run([merged, cross_entropy_loss, train_op, debug_ops], feed_dict={
                input_image: image,
                correct_label: label,
                # keep_prob: 0.5,
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
                        # keep_prob: 1.0,
                    })
                test_writer.add_summary(summary, step)
                print("  test-batch-loss: {}".format(loss))

            batch += 1
            step += 1
# tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    image_true_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    # helper.maybe_download_pretrained_mobilenet(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    learning_rate = 0.001
    epochs = 20
    labels_tensor = tf.placeholder(tf.float32, [None, None, None, num_classes])
    batch_size = 20

    img_size = 160
    factor = 0.5
    weight_decay = 0.0
    arg_scope = mobilenet_v1.mobilenet_v1_arg_scope(weight_decay=weight_decay)
    input_image = tf.placeholder(tf.float32, shape=(None, None, None, 3), name="input")

    with tf.contrib.slim.arg_scope(arg_scope):
        _, end_points = mobilenet_v1.mobilenet_v1(input_image,
                    num_classes=1001,
                    is_training=False,
                    depth_multiplier=factor)

    for name in end_points:
        print(name, end_points[name])


    with tf.Session() as sess:
        # Path to vgg model

        mobilenet_path = os.path.join(data_dir, 'mobilenet_v1_0.50_160_2017_06_14/mobilenet_v1_0.50_160.ckpt')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)
        get_batches_fn_test = helper.gen_batch_function(os.path.join(data_dir, 'data_road/testing'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        rest_var = tf.contrib.slim.get_variables_to_restore()

        saver = tf.train.Saver(rest_var)
        saver.restore(sess, mobilenet_path)

        # Build NN using load_vgg, layers, and optimize function
        # input_tensor, layer4_tensor, layer11_tensor, layer13_tensor, debug_ops = load_mobilenet(sess, mobilenet_path)
        output_layer, debug_ops = layers(end_points, num_classes)
        logits, train_op, cross_entropy_loss = optimize(output_layer, labels_tensor, learning_rate, num_classes)

        # # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('logs/{:%Y-%m-%d--%H-%M-%S}-train'.format(datetime.datetime.now()), sess.graph)
        test_writer = tf.summary.FileWriter('logs/{:%Y-%m-%d--%H-%M-%S}-test'.format(datetime.datetime.now()))

        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        # # Train NN using the train_nn function
        train_nn(sess, epochs, batch_size, get_batches_fn, get_batches_fn_test,
                     train_op, cross_entropy_loss, input_image,
                     labels_tensor, None, learning_rate, debug_ops, merged, train_writer, test_writer)


        predictions = tf.contrib.layers.softmax(logits)
        output = tf.identity(predictions, name='output')

        graph = tf.get_default_graph()
        input_graph_def = graph.as_graph_def()

        # We use a built-in TF helper to export variables to constant
        output_graph_def = graph_util.convert_variables_to_constants(
            sess, # The session is used to retrieve the weights
            input_graph_def, # The graph_def is used to retrieve the nodes
            # The output node names are used to select the useful nodes
            ['output']
            )

        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile('mobilenet_saved_model.pb', "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("{} ops in the final graph.".format(len(output_graph_def.node)))

        # # Save inference data using helper.save_inference_samples
        keep_prob = tf.placeholder(tf.float32)
        helper.save_inference_samples_mobilenet(runs_dir, data_dir, sess, image_shape, image_true_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video

# tf.reset_default_graph()
# run()
if __name__ == '__main__':
    run()
