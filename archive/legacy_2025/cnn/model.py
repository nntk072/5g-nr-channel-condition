
import tensorflow as tf
# from cnn.module import ResNet
import keras


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

# Modify ResNet for classification
def ResNetClassifier(input_shape, num_classes, dim, n_downsamplings, n_blocks, norm):
    base_model = ResNet(
        input_shape=input_shape,
        output_channels=dim,  # Intermediate feature output
        dim=dim,
        n_downsamplings=n_downsamplings,
        n_blocks=n_blocks,
        norm=norm
    )

    # Add classification head
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)  # Global pooling
    x = tf.keras.layers.Dense(32, activation='relu')(x)  # Fully connected layer

    if num_classes > 2:
        output = tf.keras.layers.Dense(num_classes, activation='softmax')(x)  # Output layer
    else:
        output = tf.keras.layers.Dense(1, activation='sigmoid')(x)  # Binary classification
    return tf.keras.Model(inputs=base_model.input, outputs=output)
