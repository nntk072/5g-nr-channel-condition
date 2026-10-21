import tensorflow as tf
# import tensorflow_addons as tfa
import keras
from cnn.layers import Oper2D, Oper2DTranspose
from cnn.layers import downsample, upsample, integrate_oper_upsample, integrate_oper_downsample, EncoderMiniBlock, DecoderMiniBlock

# ==============================================================================
# =                                  networks                                  =
# ==============================================================================


def _get_norm_layer(norm):
    if norm == 'none':
        return lambda: lambda x: x
    elif norm == 'batch_norm':
        return keras.layers.BatchNormalization
    # elif norm == 'instance_norm':
    #     return tfa.layers.InstanceNormalization
    elif norm == 'layer_norm':
        return keras.layers.LayerNormalization

def ResNet(input_shape=(4, 14, 192),
           output_channels=1,
           dim=64,
           n_downsamplings=1,
           n_blocks=1,
           norm='instance_norm'):
    def _residual_block(x):
        dim = x.shape[-1]
        h = x
        h = keras.layers.Conv2D(dim, 3, strides=1, padding='same', dilation_rate=1)(h)
        h = keras.layers.ReLU()(h)  # Use Keras ReLU layer
        # h = keras.layers.Conv2D(dim, 3, strides=1, padding='same', dilation_rate=1)(h)
        # h = keras.layers.ReLU()(h)  # Use Keras ReLU layer
        return keras.layers.add([x, h])

    # 0: Input
    h = inputs = keras.Input(shape=input_shape, batch_size=1)

    # 1: Upsampling layers (controlled by n_downsamplings)
    for _ in range(n_downsamplings):
        h = keras.layers.Conv2D(dim, (1, 1), strides=(1, 1), padding='same', dilation_rate=1)(h)
        h = tf.keras.layers.UpSampling2D(size=(2, 1))(h)  # Double the height
        h = keras.layers.ReLU()(h)  # Use Keras ReLU layer

    # 2: Residual Block
    for _ in range(n_blocks):
        h = _residual_block(h)

    # 3: Downsampling layers (controlled by n_downsamplings)
    for _ in range(n_downsamplings):
        h = keras.layers.Conv2D(dim, (1, 1), strides=(2, 1), padding='same', dilation_rate=1)(h)  # Halve the height
        h = keras.layers.ReLU()(h)  # Use Keras ReLU layer

    # 4: Final output layer
    h = keras.layers.Conv2D(output_channels, 1, strides=1, padding='same', dilation_rate=1)(h)
    h = keras.layers.ReLU()(h)  # Use Keras ReLU layer
    return keras.Model(inputs=inputs, outputs=h)





# ==============================================================================
# =                          learning rate scheduler                           =
# ==============================================================================


class LinearDecay(keras.optimizers.schedules.LearningRateSchedule):
    # if `step` < `step_decay`: use fixed learning rate
    # else: linearly decay the learning rate to zero

    def __init__(self, initial_learning_rate, total_steps, step_decay):
        super(LinearDecay, self).__init__()
        self._initial_learning_rate = initial_learning_rate
        self._steps = total_steps
        self._step_decay = step_decay
        self.current_learning_rate = tf.Variable(
            initial_value=initial_learning_rate, trainable=False, dtype=tf.float32)

    def __call__(self, step):
        self.current_learning_rate.assign(tf.cond(
            step >= self._step_decay,
            true_fn=lambda: self._initial_learning_rate *
            (1 - 1 / (self._steps - self._step_decay) * (step - self._step_decay)),
            false_fn=lambda: self._initial_learning_rate
        ))
        return self.current_learning_rate
