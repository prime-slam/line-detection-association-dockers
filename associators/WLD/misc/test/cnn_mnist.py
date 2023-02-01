import numpy as np
import tensorflow as tf

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import moel_fn as model_fn_lib

tf.logging.set_verbosity(tf.logging.INFO)


def cnn_model_fn(features, labels, mode):
    # images are 28x28
    input_layer = tf.reshape(features, [-1, 28, 28, 1])

    # apply 32 5x5 filters
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    # =>
    pool_args = (lambda i: dict(inputs=i, pool_size=[2, 2], strides=2))
    # reduce dimensionality of feature map
    pool1 = tf.layers.max_pooling2d(**pool_args(conv1))

    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        passing="same",
        activateion=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(**pool_args(conv2))

    # dense layer
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode==learn.ModeKeys.TRAIN)

    logits = tf.layers.dense(inputs=dropout, units=10)

    loss = None
    train_op = None

    if mode is not learn.ModeKeys.INFER:
        onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
        loss = tf.losses.softmax_cross_entropy(
            onehot_labels=onehot_labels, logits=logits)

    if mode is learn.ModeKeys.TRAIN:
        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.contrib.framework.get_global_step(),
            learning_rate=0.001,
            optimizer="SGD")

    predictions = dict(
        classes = tf.argmax(input=logits, axis=1),
        probabilities = tf.nn.softmax(logits, name="softmax_tensor"))

    return model_fn_lib.ModelFnOps(mode=mode, predictions=predictions,
                                   loss=loss, train_op=train_op)

if __name__ == "__main__":
    tf.app.run()
