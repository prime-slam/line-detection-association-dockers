"""This is almost an exact copy of `tensorpack_examples/resnet/resnet_model.py`
with a few modifications for usage in this project.
"""
# -*- coding: utf-8 -*-
# File: resnet_model.py

import tensorflow as tf
import tensorflow_addons as tfa

from tensorpack.tfutils.argscope import argscope, get_arg_scope
from tensorpack.models import (
    Conv2D, GlobalAvgPooling, BatchNorm, BNReLU, FullyConnected,
    LinearWrap, MaxPooling)


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


def preresnet_basicblock(x, ch_out, stride, preact, bn=True):
    act = BNReLU if bn else tf.nn.relu
    preact = preact if bn else 'no_preact'
    x, shortcut = apply_preactivation(x, preact)
    x = Conv2D('conv1', x, ch_out, 3, stride=stride, nl=act)
    x = Conv2D('conv2', x, ch_out, 3)
    return x + resnet_shortcut(shortcut, ch_out, stride)


def preresnet_bottleneck(x, ch_out, stride, preact):
    # stride is applied on the second conv, following fb.resnet.torch
    x, shortcut = apply_preactivation(x, preact)
    x = Conv2D('conv1', x, ch_out, 1, nl=BNReLU)
    x = Conv2D('conv2', x, ch_out, 3, stride=stride, nl=BNReLU)
    x = Conv2D('conv3', x, ch_out * 4, 1)
    return x + resnet_shortcut(shortcut, ch_out * 4, stride)


def preresnet_group(x, name, block_func, features, count, stride, bn=True):
    with tf.compat.v1.variable_scope(name):
        for i in range(0, count):
            with tf.compat.v1.variable_scope('block{}'.format(i)):
                # first block doesn't need activation
                x = block_func(x, features,
                               stride if i == 0 else 1,
                               'no_preact' if i == 0 else 'bnrelu',
                               bn=bn)
        # end of each group need an extra activation
        if bn:
            x = BNReLU('bnlast', x)
        else:
            x = tf.nn.relu(x, 'relulast')
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
            with tf.compat.v1.variable_scope('block{}'.format(i)):
                x = block_func(x, features, stride if i == 0 else 1)
                # end of each block need an activation
                x = tf.nn.relu(x)
    return x


def resnet_backbone(images, num_blocks, grp_fun, blck_fun, nfeatures, bn=True):
    # from tf.contrib.layers import variance_scaling_initializer
    with argscope(Conv2D, nl=tf.identity, use_bias=False,
                  W_init=tf.keras.initializers.VarianceScaling(scale=2.0,
                                                         mode='fan_out')):
        first_input = images[0]
        second_input = images[1]
        act = BNReLU if bn else tf.nn.relu
        x = Conv2D('conv0', first_input, 64, 7, stride=2, nl=act)
        y = Conv2D('conv1', second_input, 64, 7, stride=2, nl=act)
        # stack second_input into channel-dimension of conv0 output
        x = tf.concat([x, y], axis=3, name='stack_second_input')
        x = MaxPooling('pool0', x, shape=3, stride=2, padding='SAME')
        x = grp_fun(x, 'group0', blck_fun, 64 , num_blocks[0], 1, bn=bn)
        x = grp_fun(x, 'group1', blck_fun, 128, num_blocks[1], 2, bn=bn)
        x = grp_fun(x, 'group2', blck_fun, 256, num_blocks[2], 2, bn=bn)
        x = grp_fun(x, 'group3', blck_fun, 256, num_blocks[3], 1, bn=bn)
        x = GlobalAvgPooling('gap', x)
        x = FullyConnected('fc0', x, 1000)  # NOTE linear activations gewollt ?
        x = FullyConnected('fc1', x, 500)  # NOTE linear activations gewollt ?
        x = FullyConnected('linear', x, nfeatures,
                           nl=tf.identity)  # NOTE sieht aus als ging Fabi von non-linear act. als default aus
        # NOTE die letzten 3 FC layers werden linear aktiviert (siehe Graph in TB) d.h. ein einzelnes FC layer sollte ausreichen (evtl. bessere Laufzeit)
    return x
