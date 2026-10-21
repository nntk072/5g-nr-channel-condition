import tensorflow as tf
# import tensorflow_addons as tfa
import numpy as np

####################################################
# Operational Layers.


class Oper2D(tf.keras.Model):
    def __init__(self, filters, kernel_size, activation=None, q=1, padding='valid', use_bias=True, strides=1, apply_initializer=False):
        super(Oper2D, self).__init__(name='')

        self.activation = activation
        self.q = q
        self.all_layers = []
        initializer = tf.random_normal_initializer(0., 0.02)
        for i in range(0, q):  # q convolutional layers.
            if not apply_initializer:
                self.all_layers.append(tf.keras.layers.Conv2D(filters,
                                                              (kernel_size,
                                                               kernel_size),
                                                              padding=padding,
                                                              use_bias=use_bias,
                                                              strides=strides,
                                                              activation=activation))
            else:
                self.all_layers.append(tf.keras.layers.Conv2D(filters,
                                                              (kernel_size,
                                                               kernel_size),
                                                              padding=padding,
                                                              use_bias=use_bias,
                                                              strides=strides,
                                                              activation=activation,
                                                              kernel_initializer=initializer))

    @tf.function
    def call(self, input_tensor, training=False):

        x = self.all_layers[0](input_tensor)  # First convolutional layer.

        if self.q > 1:
            for i in range(1, self.q):
                x += self.all_layers[i](tf.math.pow(input_tensor, i + 1))

        if self.activation is not None:
            return eval('tf.nn.' + self.activation + '(x)')
        else:
            return x

####################################################
# Transposed Operational Layers.


class Oper2DTranspose(tf.keras.Model):
    def __init__(self, filters, kernel_size, activation=None, q=1, padding='valid', use_bias=True, strides=1, apply_initializer=False):
        super(Oper2DTranspose, self).__init__(name='')

        self.activation = activation
        self.q = q
        self.all_layers = []
        initializer = tf.random_normal_initializer(0., 0.02)
        for i in range(0, q):  # q convolutional layers.
            if not apply_initializer:
                self.all_layers.append(tf.keras.layers.Conv2DTranspose(filters,
                                                                       kernel_size,
                                                                       padding=padding,
                                                                       use_bias=use_bias,
                                                                       strides=strides,
                                                                       activation=activation))
            else:
                self.all_layers.append(tf.keras.layers.Conv2DTranspose(filters,
                                                                       kernel_size,
                                                                       padding=padding,
                                                                       use_bias=use_bias,
                                                                       strides=strides,
                                                                       activation=activation,
                                                                       kernel_initializer=initializer))

    @tf.function
    def call(self, input_tensor, training=False):

        x = self.all_layers[0](input_tensor)  # First convolutional layer.

        if self.q > 1:
            for i in range(1, self.q):
                x += self.all_layers[i](tf.math.pow(input_tensor, i + 1))

        if self.activation is not None:
            return eval('tf.nn.' + self.activation + '(x)')
        else:
            return x


def downsample(filters, size, norm_type='batch_norm', apply_norm=True):
    """Downsamples an input.

    Conv2D => instance_norm => LeakyRelu

    Args:
      filters: number of filters
      size: filter size
      norm_type: Normalization type; either 'batch_norm' or 'instance_norm'.
      apply_norm: If True, adds the batch_norm layer

    Returns:
      Downsample Sequential Model
    """
    # initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                               # kernel_initializer=initializer,
                               use_bias=False))
    if apply_norm:
        if norm_type.lower() == 'batch_norm':
            result.add(tf.keras.layers.batch_normalization())
        elif norm_type.lower() == 'instance_norm':
            result.add(tfa.layers.InstanceNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result


def upsample(filters, size, norm_type='instance_norm', apply_dropout=False):
    """Upsamples an input.

    Conv2DTranspose => instance_norm => Dropout => Relu

    Args:
      filters: number of filters
      size: filter size
      norm_type: Normalization type; either 'batch_norm' or 'instance_norm'.
      apply_dropout: If True, adds the dropout layer

    Returns:
      Upsample Sequential Model
    """

    # initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                        padding='same',
                                        # kernel_initializer=initializer,
                                        use_bias=False))

    if norm_type.lower() == 'batch_norm':
        result.add(tf.keras.layers.batch_normalization())
    elif norm_type.lower() == 'instance_norm':
        result.add(tfa.layers.InstanceNormalization())

    if apply_dropout:
        # result.add(tf.keras.layers.Dropout(0.5))
        result.add(tf.keras.layers.Dropout(0.2))

    result.add(tf.keras.layers.ReLU())

    return result


def integrate_oper_upsample(filters, size, norm_type='instance_norm', apply_dropout=False, q=1):
    """Upsamples an input.

    Oper2DTranspose => instance_norm => Dropout => Relu

    Args:
      filters: number of filters
      size: filter size
      norm_type: Normalization type; either 'batch_norm' or 'instance_norm'.
      apply_dropout: If True, adds the dropout layer

    Returns:
      Upsample Sequential Model
    """
    result = tf.keras.Sequential()
    result.add(
        Oper2DTranspose(filters, size, strides=2, q=q, padding='same', use_bias=False, apply_initializer=False))

    if norm_type.lower() == 'batch_norm':
        result.add(tf.keras.layers.batch_normalization())
    elif norm_type.lower() == 'instance_norm':
        result.add(tfa.layers.InstanceNormalization())

    if apply_dropout:
        # result.add(tf.keras.layers.Dropout(0.5))
        result.add(tf.keras.layers.Dropout(0.2))

    # result.add(tf.keras.layers.ReLU())
    # Using tanh instead
    result.add(tf.keras.layers.Activation('tanh'))

    return result


def integrate_oper_downsample(filters, size, norm_type='batch_norm', apply_norm=True, q=1):
    """Downsamples an input.

    Oper2D => instance_norm => LeakyRelu

    Args:
      filters: number of filters
      size: filter size
      norm_type: Normalization type; either 'batch_norm' or 'instance_norm'.
      apply_norm: If True, adds the batch_norm layer

    Returns:
      Downsample Sequential Model
    """
    result = tf.keras.Sequential()
    result.add(
        Oper2D(filters, size, strides=2, q=q, padding='same', use_bias=False, apply_initializer=False))

    if apply_norm:
        if norm_type.lower() == 'batch_norm':
            result.add(tf.keras.layers.batch_normalization())
        elif norm_type.lower() == 'instance_norm':
            result.add(tfa.layers.InstanceNormalization())

    result.add(tf.keras.layers.Activation('tanh'))

    return result


def EncoderMiniBlock(inputs, n_filters=32, dropout_prob=0.3, max_pooling=True):
    """
    This block uses multiple convolution layers, max pool, relu activation to create an architecture for learning. 
    Dropout can be added for regularization to prevent overfitting. 
    The block returns the activation values for next layer along with a skip connection which will be used in the decoder
    """
    # Add 2 Conv Layers with relu activation and HeNormal initialization using TensorFlow
    # Proper initialization prevents from the problem of exploding and vanishing gradients
    # 'Same' padding will pad the input to conv layer such that the output has the same height and width (hence, is not reduced in size)
    conv = tf.keras.layers.Conv2D(n_filters,
                                  3,   # Kernel size
                                  #   activation='relu',
                                  padding='same',
                                  #   kernel_initializer='HeNormal'
                                  )(inputs)
    conv = tfa.layers.InstanceNormalization()(conv)
    conv = tf.keras.layers.Activation('tanh')(conv)
    conv = tf.keras.layers.Conv2D(n_filters,
                                  3,   # Kernel size
                                  #   activation='relu',
                                  padding='same',
                                  #   kernel_initializer='HeNormal'
                                  )(conv)

    # # Batch Normalization will normalize the output of the last layer based on the batch's mean and standard deviation
    # conv = BatchNormalization()(conv, training=False)
    conv = tfa.layers.InstanceNormalization()(conv)
    conv = tf.keras.layers.Activation('tanh')(conv)

    # In case of overfitting, dropout will regularize the loss and gradient computation to shrink the influence of weights on output
    if dropout_prob > 0:
        conv = tf.keras.layers.Dropout(dropout_prob)(conv)

    # Pooling reduces the size of the image while keeping the number of channels same
    # Pooling has been kept as optional as the last encoder layer does not use pooling (hence, makes the encoder block flexible to use)
    # Below, Max pooling considers the maximum of the input slice for output computation and uses stride of 2 to traverse across input image
    if max_pooling:
        next_layer = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
        next_layer = tfa.layers.InstanceNormalization()(next_layer)
        next_layer = tf.keras.layers.Activation('tanh')(next_layer)
    else:
        next_layer = conv

    # skip connection (without max pooling) will be input to the decoder layer to prevent information loss during transpose convolutions
    skip_connection = conv

    return next_layer, skip_connection


def DecoderMiniBlock(prev_layer_input, skip_layer_input, n_filters=32):
    """
    Decoder Block first uses transpose convolution to upscale the image to a bigger size and then,
    merges the result with skip layer results from encoder block
    Adding 2 convolutions with 'same' padding helps further increase the depth of the network for better predictions
    The function returns the decoded layer output
    """
    # Start with a transpose convolution layer to first increase the size of the image
    up = tf.keras.layers.Conv2DTranspose(
        n_filters,
        (3, 3),    # Kernel size
        strides=(2, 2),
        padding='same')(prev_layer_input)
    up = tfa.layers.InstanceNormalization()(up)
    up = tf.keras.layers.Activation('tanh')(up)
    # Merge the skip connection from previous block to prevent information loss
    merge = tf.keras.layers.concatenate([up, skip_layer_input], axis=3)

    # Add 2 Conv Layers with relu activation and HeNormal initialization for further processing
    # The parameters for the function are similar to encoder
    conv = tf.keras.layers.Conv2D(n_filters,
                                  3,     # Kernel size
                                  #  activation='relu',
                                  padding='same',
                                  #  kernel_initializer='HeNormal'
                                  )(merge)
    conv = tfa.layers.InstanceNormalization()(conv)
    conv = tf.keras.layers.Activation('tanh')(conv)
    conv = tf.keras.layers.Conv2D(n_filters,
                                  3,   # Kernel size
                                  #  activation='relu',
                                  padding='same',
                                  #  kernel_initializer='HeNormal'
                                  )(conv)
    conv = tfa.layers.InstanceNormalization()(conv)
    conv = tf.keras.layers.Activation('tanh')(conv)
    return conv
