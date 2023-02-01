def create_dense_net(num_blocks, image, nfeatures):
    import tensorpack as tp
    import tensorflow as tf
    from cnn import config

    def conv(name, l, channel, stride, kern_size=3):
        import numpy as np
        return tp.Conv2D(name, l, channel, kern_size, stride=stride,
                         nl=tf.identity, use_bias=False,
                         W_init=tf.random_normal_initializer(
                             stddev=np.sqrt(2.0 / 9 / channel)))

    def add_layer(name, l, kern_size=3, gr_factor=1):
        with tf.variable_scope(name):
            c = tp.BatchNorm('bn', l)
            c = tf.nn.relu(c)
            c = conv('conv', c, gr_factor * config.growth_rate,
                     1, kern_size)
        return c

    def add_transition(name, layer, theta):
        import numpy as np

        shape = layer.get_shape().as_list()
        in_channel = np.floor(theta * shape[3])
        with tf.variable_scope(name):
            layer = tp.BatchNorm('bn', layer)
            layer = tf.nn.relu(layer)
            layer = tp.Conv2D('conv', layer, in_channel, 1, stride=1,
                              use_bias=False, nl=tf.nn.relu)
            layer = tp.AvgPooling('pool', layer, 2)
        return layer

    def dense_net(name, num_blocks, init_channels):
        layer = conv('conv0', image, init_channels, 1)

        for j in range(len(num_blocks)):
            with tf.variable_scope('block{}'.format(j + 1)):

                for i in range(num_blocks[j]):
                    c = layer
                    if (config.dense_net_BC):
                        # add 1x1 bottleneck layer
                        c = add_layer('bottleneck.{}'.format(i), c, 1, 4)
                    c = add_layer('dense_layer.{}'.format(i), c)
                    layer = tf.concat([c, layer], 3)

                if j < (len(num_blocks) - 1):
                    layer = add_transition('transition',
                                           layer,
                                           config.theta)

        layer = tp.BatchNorm('bnlast', layer)
        layer = tf.nn.relu(layer)
        layer = tp.GlobalAvgPooling('gap', layer)
        logits = tp.FullyConnected('linear', layer, out_dim=nfeatures,
                                   nl=tf.identity)

        return logits

    init_channels = 16
    if config.dense_net_BC:
        init_channels = 2 * config.growth_rate

    logits = dense_net("dense_net", num_blocks, init_channels)

    prob = tf.nn.softmax(logits, name='output')

    return prob
