"""This is almost an exact copy of `tensorpack_examples/resnet/resnet_model.py`
with a few modifications for usage in this project.
"""
# -*- coding: utf-8 -*-
# File: resnet_model.py

import tensorflow as tf


from tensorpack.tfutils.argscope import argscope, get_arg_scope
from tensorpack.models import (
    Conv2D, GlobalAvgPooling, BatchNorm, BNReLU, FullyConnected,
    LinearWrap)


def resnet_shortcut(x, n_out, stride, nl=tf.identity):
    data_format = get_arg_scope()['Conv2D']['data_format']
    n_in = x.get_shape().as_list()[1 if data_format == 'NCHW' else 3]
    if n_in != n_out:   # change dimension when channel is not the same
        return Conv2D('convshortcut', x, n_out, 1, stride=stride, nl=nl)
    else:
        return x


def apply_preactivation(x, preact):
    if preact == 'bnrelu':
        shortcut = x    # preserve identity mapping
        x = BNReLU('preact', x)
    else:
        shortcut = x
    return x, shortcut


def get_bn(zero_init=False):
    """
    Zero init gamma is good for resnet. See https://arxiv.org/abs/1706.02677.
    """
    if zero_init:
        return (lambda x, name:
                BatchNorm('bn', x, gamma_init=tf.zeros_initializer()))
    else:
        return (lambda x, name:
                BatchNorm('bn', x))


def preresnet_basicblock(x, ch_out, stride, preact):
    x, shortcut = apply_preactivation(x, preact)
    x = Conv2D('conv1', x, ch_out, 3, stride=stride, nl=BNReLU)
    x = Conv2D('conv2', x, ch_out, 3)
    return x + resnet_shortcut(shortcut, ch_out, stride)


def preresnet_bottleneck(x, ch_out, stride, preact):
    # stride is applied on the second conv, following fb.resnet.torch
    x, shortcut = apply_preactivation(x, preact)
    x = Conv2D('conv1', x, ch_out, 1, nl=BNReLU)
    x = Conv2D('conv2', x, ch_out, 3, stride=stride, nl=BNReLU)
    x = Conv2D('conv3', x, ch_out * 4, 1)
    return x + resnet_shortcut(shortcut, ch_out * 4, stride)


def preresnet_group(x, name, block_func, features, count, stride):
    with tf.compat.v1.variable_scope(name):
        for i in range(0, count):
            with tf.compat.v1.variable_scope('block{}'.format(i)):
                # first block doesn't need activation
                x = block_func(x, features,
                               stride if i == 0 else 1,
                               'no_preact' if i == 0 else 'bnrelu')
        # end of each group need an extra activation
        x = BNReLU('bnlast', x)
    return x


def resnet_basicblock(x, ch_out, stride):
    shortcut = x
    x = Conv2D('conv1', x, ch_out, 3, stride=stride, nl=BNReLU)
    x = Conv2D('conv2', x, ch_out, 3, nl=get_bn(zero_init=True))
    return x + resnet_shortcut(shortcut, ch_out, stride,
                               nl=get_bn(zero_init=False))


def resnet_bottleneck(x, ch_out, stride, stride_first=False):
    """
    stride_first: originax resnet put stride on first conv. fb.resnet.torch put stride on second conv. # noqa
    """
    shortcut = x
    x = Conv2D('conv1', x, ch_out, 1, stride=stride if stride_first else 1,
               nl=BNReLU)
    x = Conv2D('conv2', x, ch_out, 3, stride=1 if stride_first else stride,
               nl=BNReLU)
    x = Conv2D('conv3', x, ch_out * 4, 1, nl=get_bn(zero_init=True))
    return x + resnet_shortcut(shortcut, ch_out * 4, stride,
                               nl=get_bn(zero_init=False))


def se_resnet_bottleneck(x, ch_out, stride):
    shortcut = x
    x = Conv2D('conv1', x, ch_out, 1, nl=BNReLU)
    x = Conv2D('conv2', x, ch_out, 3, stride=stride, nl=BNReLU)
    x = Conv2D('conv3', x, ch_out * 4, 1, nl=get_bn(zero_init=True))

    squeeze = GlobalAvgPooling('gap', x)
    squeeze = FullyConnected('fc1', squeeze, ch_out // 4, nl=tf.nn.relu)
    squeeze = FullyConnected('fc2', squeeze, ch_out * 4, nl=tf.nn.sigmoid)
    data_format = get_arg_scope()['Conv2D']['data_format']
    ch_ax = 1 if data_format == 'NCHW' else 3
    shape = [-1, 1, 1, 1]
    shape[ch_ax] = ch_out * 4
    x = x * tf.reshape(squeeze, shape)
    return x + resnet_shortcut(shortcut, ch_out * 4, stride,
                               nl=get_bn(zero_init=False))


def resnet_group(x, name, block_func, features, count, stride):
    with tf.variable_scope(name):
        for i in range(0, count):
            with tf.variable_scope('block{}'.format(i)):
                x = block_func(x, features, stride if i == 0 else 1)
                # end of each block need an activation
                x = tf.nn.relu(x)
    return x


def resnet_backbone(image, num_blocks, grp_fun, blck_fun, nfeatures):
    # from tf.contrib.layers import variance_scaling_initializer
    with argscope(Conv2D, nl=tf.identity, use_bias=False,
                  W_init=tf.keras.initializers.VarianceScaling(scale=2.0,
                                                         mode='fan_out')):
        # TODO evaluate conv depth
        logits = (LinearWrap(image)
                  .Conv2D('conv0', 64, 7, stride=2, nl=BNReLU)
                  .MaxPooling('pool0', shape=3, stride=2, padding='SAME')
                  .apply(grp_fun, 'group0', blck_fun, 64,  num_blocks[0], 1)
                  .apply(grp_fun, 'group1', blck_fun, 128, num_blocks[1], 2)
                  .apply(grp_fun, 'group2', blck_fun, 256, num_blocks[2], 2)
                  # .apply(grp_fun, 'group3', blck_fun, 512, num_blocks[3], 2)
                  .apply(grp_fun, 'group3', blck_fun, 256, num_blocks[3], 1)
                  .GlobalAvgPooling('gap')
                  .FullyConnected('fc0', 1000)
                  .FullyConnected('fc1', 500)
                  .FullyConnected('linear', nfeatures, nl=tf.identity)())
    return logits
