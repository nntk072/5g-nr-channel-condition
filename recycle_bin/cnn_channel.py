# %%
import os
import numpy as np
from datetime import datetime
import tf2onnx
import keras
from keras.metrics import BinaryAccuracy, SparseCategoricalAccuracy
from keras.models import load_model
from tqdm import tqdm  # For dynamic progress updates
from cnn.model import ResNetClassifier
from utils import load_features_from_hdf5
from data_preprocessing import (
    extract_and_concatenate_with_grouping,
    extract_and_concatenate_with_grouping_channel,
    extract_and_concatenate_with_grouping_tdl_cdl,
    extract_and_concatenate_with_grouping_channel_tdl_cdl,
)

from cnn.layers import (
    downsample,
    upsample,
    integrate_oper_upsample,
    integrate_oper_downsample,
    EncoderMiniBlock,
    DecoderMiniBlock,
    Oper2D,
    Oper2DTranspose,
)
from cnn.model import ResNetClassifier
from cnn.module import ResNet
from evaluation import print_metrics, print_metrics_multiclass
from visualization import (
    plot_confusion_matrix,
    plot_confusion_matrix_multiclass,
    plot_roc_curve,
    plot_roc_curve_multiclass,
)
import onnxruntime as ort  # ONNX Runtime for inference
if os.getenv("CUDA_VISIBLE_DEVICES") is None:
    gpu_num = 0  # Use "" to use the CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
    print(f"Using GPU {gpu_num}")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)
# Avoid warnings from TensorFlow
tf.get_logger().setLevel("ERROR")

# %%
# Load features from HDF5 files

# mimo_los_test = load_features_from_hdf5('dataset_sionna_mimo_los_test.h5')
# mimo_los_train = load_features_from_hdf5('dataset_sionna_mimo_los.h5')
# mimo_nlos_test = load_features_from_hdf5('dataset_sionna_mimo_nlos_test.h5')
# mimo_nlos_train = load_features_from_hdf5('dataset_sionna_mimo_nlos.h5')
# siso_los_test = load_features_from_hdf5('dataset_sionna_siso_los_test.h5')
# siso_los_train = load_features_from_hdf5('dataset_sionna_siso_los.h5')
# siso_nlos_test = load_features_from_hdf5('dataset_sionna_siso_nlos_test.h5')
# siso_nlos_train = load_features_from_hdf5('dataset_sionna_siso_nlos.h5')
# tdl_cdl_test = load_features_from_hdf5('dataset_tdl_cdl_test.h5')
# tdl_cdl_train = load_features_from_hdf5('dataset_tdl_cdl.h5')

# mimo_los_test = load_features_from_hdf5('/home/l2nguyen/long_nguyen_thesis/complete_dataset_without_scenario/dataset_sionna_mimo_los_test.h5')
# mimo_los_train = load_features_from_hdf5('/home/l2nguyen/long_nguyen_thesis/complete_dataset_without_scenario/dataset_sionna_mimo_los.h5')
# mimo_nlos_test = load_features_from_hdf5('/home/l2nguyen/long_nguyen_thesis/complete_dataset_without_scenario/dataset_sionna_mimo_nlos_test.h5')
# mimo_nlos_train = load_features_from_hdf5('/home/l2nguyen/long_nguyen_thesis/complete_dataset_without_scenario/dataset_sionna_mimo_nlos.h5')
# siso_los_test = load_features_from_hdf5('/home/l2nguyen/long_nguyen_thesis/complete_dataset_without_scenario/dataset_sionna_siso_los_test.h5')
# siso_los_train = load_features_from_hdf5('/home/l2nguyen/long_nguyen_thesis/complete_dataset_without_scenario/dataset_sionna_siso_los.h5')
# siso_nlos_test = load_features_from_hdf5('/home/l2nguyen/long_nguyen_thesis/complete_dataset_without_scenario/dataset_sionna_siso_nlos_test.h5')
# siso_nlos_train = load_features_from_hdf5('/home/l2nguyen/long_nguyen_thesis/complete_dataset_without_scenario/dataset_sionna_siso_nlos.h5')
tdl_cdl_test = load_features_from_hdf5('dataset_tdl_cdl_test.h5')
tdl_cdl_train = load_features_from_hdf5('dataset_tdl_cdl.h5')
# 
# Extract and concatenate for MIMO datasets
# train_mimo_los, train_mimo_los_labels = extract_and_concatenate_with_grouping(mimo_los_train)
# train_mimo_nlos, train_mimo_nlos_labels = extract_and_concatenate_with_grouping(mimo_nlos_train)
# train_siso_los, train_siso_los_labels = extract_and_concatenate_with_grouping(siso_los_train)
# train_siso_nlos, train_siso_nlos_labels = extract_and_concatenate_with_grouping(siso_nlos_train)

# test_mimo_los, test_mimo_los_labels = extract_and_concatenate_with_grouping(mimo_los_test)
# test_mimo_nlos, test_mimo_nlos_labels = extract_and_concatenate_with_grouping(mimo_nlos_test)
# test_siso_los, test_siso_los_labels = extract_and_concatenate_with_grouping(siso_los_test)
# test_siso_nlos, test_siso_nlos_labels = extract_and_concatenate_with_grouping(siso_nlos_test)
train_tdls, train_cdls, train_tdls_labels, train_cdls_labels = extract_and_concatenate_with_grouping_tdl_cdl(tdl_cdl_train)
test_tdls, test_cdls, test_tdls_labels, test_cdls_labels = extract_and_concatenate_with_grouping_tdl_cdl(tdl_cdl_test)

# train_mimo_los_channel, train_mimo_los_labels_channel = extract_and_concatenate_with_grouping_channel(mimo_los_train)
# train_mimo_nlos_channel, train_mimo_nlos_labels_channel = extract_and_concatenate_with_grouping_channel(mimo_nlos_train)
# train_siso_los_channel, train_siso_los_labels_channel = extract_and_concatenate_with_grouping_channel(siso_los_train)
# train_siso_nlos_channel, train_siso_nlos_labels_channel = extract_and_concatenate_with_grouping_channel(siso_nlos_train)
# test_mimo_los_channel, test_mimo_los_labels_channel = extract_and_concatenate_with_grouping_channel(mimo_los_test)
# test_mimo_nlos_channel, test_mimo_nlos_labels_channel = extract_and_concatenate_with_grouping_channel(mimo_nlos_test)
# test_siso_los_channel, test_siso_los_labels_channel = extract_and_concatenate_with_grouping_channel(siso_los_test)
# test_siso_nlos_channel, test_siso_nlos_labels_channel = extract_and_concatenate_with_grouping_channel(siso_nlos_test)

# Extract and concatenate for MIMO datasets with channel dimension
train_tdls_channel, train_cdls_channel, label_tdl, label_cdl = extract_and_concatenate_with_grouping_channel_tdl_cdl(tdl_cdl_train)
test_tdls_channel, test_cdls_channel, test_label_tdl, test_label_cdl = extract_and_concatenate_with_grouping_channel_tdl_cdl(tdl_cdl_test)


# %%
print("train_tdls shape:", train_tdls.shape)
print("train_cdls shape:", train_cdls.shape)
print("train_tdls_channel shape:", train_tdls_channel.shape)
print("train_cdls_channel shape:", train_cdls_channel.shape)
print("train_tdls_labels shape:", train_tdls_labels.shape)
print("train_cdls_labels shape:", train_cdls_labels.shape)
print("test_tdls shape:", test_tdls.shape)
print("test_cdls shape:", test_cdls.shape)
print("test_tdls_channel shape:", test_tdls_channel.shape)
print("test_cdls_channel shape:", test_cdls_channel.shape)
print("test_tdls_labels shape:", test_tdls_labels.shape)
print("test_cdls_labels shape:", test_cdls_labels.shape)

# %%
# # Combine dimensions for MIMO data (e.g., 1 and 2 -> 1 * 2 = 2)
# train_mimo_los = train_mimo_los.reshape(train_mimo_los.shape[0], train_mimo_los.shape[1], train_mimo_los.shape[2] * train_mimo_los.shape[3], train_mimo_los.shape[4], train_mimo_los.shape[5])
# train_mimo_nlos = train_mimo_nlos.reshape(train_mimo_nlos.shape[0], train_mimo_nlos.shape[1], train_mimo_nlos.shape[2] * train_mimo_nlos.shape[3], train_mimo_nlos.shape[4], train_mimo_nlos.shape[5])
# test_mimo_los = test_mimo_los.reshape(test_mimo_los.shape[0], test_mimo_los.shape[1], test_mimo_los.shape[2] * test_mimo_los.shape[3], test_mimo_los.shape[4], test_mimo_los.shape[5])
# test_mimo_nlos = test_mimo_nlos.reshape(test_mimo_nlos.shape[0], test_mimo_nlos.shape[1], test_mimo_nlos.shape[2] * test_mimo_nlos.shape[3], test_mimo_nlos.shape[4], test_mimo_nlos.shape[5])

# # Combine dimensions for SISO data (e.g., 1 and 1 -> 1 * 1 = 1, effectively removing them)
# train_siso_los = train_siso_los.reshape(train_siso_los.shape[0], train_siso_los.shape[1], train_siso_los.shape[2] * train_siso_los.shape[3], train_siso_los.shape[4], train_siso_los.shape[5])
# train_siso_nlos = train_siso_nlos.reshape(train_siso_nlos.shape[0], train_siso_nlos.shape[1], train_siso_nlos.shape[2] * train_siso_nlos.shape[3], train_siso_nlos.shape[4], train_siso_nlos.shape[5])
# test_siso_los = test_siso_los.reshape(test_siso_los.shape[0], test_siso_los.shape[1], test_siso_los.shape[2] * test_siso_los.shape[3], test_siso_los.shape[4], test_siso_los.shape[5])
# test_siso_nlos = test_siso_nlos.reshape(test_siso_nlos.shape[0], test_siso_nlos.shape[1], test_siso_nlos.shape[2] * test_siso_nlos.shape[3], test_siso_nlos.shape[4], test_siso_nlos.shape[5])

# Combine dimensions for TDL/CDL data (e.g., 1 and 1 -> 1 * 1 = 1, effectively removing them)
train_tdls = train_tdls.reshape(train_tdls.shape[0], train_tdls.shape[1], train_tdls.shape[2] * train_tdls.shape[3], train_tdls.shape[4], train_tdls.shape[5])
train_cdls = train_cdls.reshape(train_cdls.shape[0], train_cdls.shape[1], train_cdls.shape[2] * train_cdls.shape[3], train_cdls.shape[4], train_cdls.shape[5])
print("train_tdls shape after reshape:", train_tdls.shape)
print("train_cdls shape after reshape:", train_cdls.shape)

train_tdls_channel = train_tdls_channel.reshape(train_tdls_channel.shape[0], train_tdls_channel.shape[1], train_tdls_channel.shape[2] * train_tdls_channel.shape[3], train_tdls_channel.shape[4], train_tdls_channel.shape[5])
train_cdls_channel = train_cdls_channel.reshape(train_cdls_channel.shape[0], train_cdls_channel.shape[1], train_cdls_channel.shape[2] * train_cdls_channel.shape[3], train_cdls_channel.shape[4], train_cdls_channel.shape[5])
print("train_tdls_channel shape after reshape:", train_tdls_channel.shape)
print("train_cdls_channel shape after reshape:", train_cdls_channel.shape)
    
test_cdls = test_cdls.reshape(test_cdls.shape[0], test_cdls.shape[1], test_cdls.shape[2] * test_cdls.shape[3], test_cdls.shape[4], test_cdls.shape[5])
test_tdls = test_tdls.reshape(test_tdls.shape[0], test_tdls.shape[1], test_tdls.shape[2] * test_tdls.shape[3], test_tdls.shape[4], test_tdls.shape[5])
print("test_tdls shape after reshape:", test_tdls.shape)
print("test_cdls shape after reshape:", test_cdls.shape)
test_tdls_channel = test_tdls_channel.reshape(test_tdls_channel.shape[0], test_tdls_channel.shape[1], test_tdls_channel.shape[2] * test_tdls_channel.shape[3], test_tdls_channel.shape[4], test_tdls_channel.shape[5])
test_cdls_channel = test_cdls_channel.reshape(test_cdls_channel.shape[0], test_cdls_channel.shape[1], test_cdls_channel.shape[2] * test_cdls_channel.shape[3], test_cdls_channel.shape[4], test_cdls_channel.shape[5])
print("test_tdls_channel shape after reshape:", test_tdls_channel.shape)
print("test_cdls_channel shape after reshape:", test_cdls_channel.shape)


# %%
train_data = np.concatenate((train_tdls, train_cdls), axis=0)
train_data_labels = np.concatenate((train_tdls_labels, train_cdls_labels), axis=0)
test_data = np.concatenate((test_tdls, test_cdls), axis=0)
test_data_labels = np.concatenate((test_tdls_labels, test_cdls_labels), axis=0)
print("train_data shape:", train_data.shape)
print("train_labels shape:", train_data_labels.shape)
print("test_data shape:", test_data.shape)
print("test_labels shape:", test_data_labels.shape)

train_data_channel = np.concatenate((train_tdls_channel, train_cdls_channel), axis=0)
train_data_channel_labels = np.concatenate((label_tdl, label_cdl), axis=0)
test_data_channel = np.concatenate((test_tdls_channel, test_cdls_channel), axis=0)
test_data_channel_labels = np.concatenate((test_label_tdl, test_label_cdl), axis=0)
print("train_data_channel shape:", train_data_channel.shape)
print("test_data_channel shape:", test_data_channel.shape)
print("train_labels shape:", train_data_labels.shape)
print("test_labels shape:", test_data_labels.shape)

# %%
train_data = train_data.reshape(-1, 4, 14, 192)  # Shape: (2000 * 8, 14, 192)
# train_data_labels = train_data_labels.repeat(8)  # Repeat labels for each sample in the batch

test_data = test_data.reshape(-1, 4, 14, 192)  # Shape: (8000 * 8, 14, 192)
# test_data_labels = test_data_labels.repeat(8)  # Repeat labels for each sample in the batch

print(train_data.shape)  # Expected: (2000, 8, 14, 192)
print(train_data_labels.shape)  # Expected: (2000,)
print(test_data.shape)  # Expected: (8000, 8, 14, 192)
print(test_data_labels.shape)  # Expected: (8000,)

# check the variabes histogram (count the number of samples in each class)
unique, counts = np.unique(train_data_labels, return_counts=True)
print("Unique classes in train_data_labels:", unique)
print("Counts of each class in train_data_labels:", counts)
unique, counts = np.unique(test_data_labels, return_counts=True)
print("Unique classes in test_data_labels:", unique)
print("Counts of each class in test_data_labels:", counts)

train_data_channel = train_data_channel.reshape(-1, 4, 14, 192)  # Shape: (2000 * 8, 14, 192)
# train_data_labels_channel = train_data_labels_channel.repeat(8)  # Repeat labels for each sample in the batch
test_data_channel = test_data_channel.reshape(-1, 4, 14, 192)  # Shape: (8000 * 8, 14, 192)
# test_data_labels_channel = test_data_labels_channel.repeat(8)  # Repeat labels for each sample in the batch
print(train_data_channel.shape)  # Expected: (2000, 8, 14, 192)
# print(train_data_labels_channel.shape)  # Expected: (2000,)
print(test_data_channel.shape)  # Expected: (8000, 8, 14, 192)
# check the variabes histogram (count the number of samples in each class)


# check the variabes histogram (count the number of samples in each class)
unique, counts = np.unique(train_data_channel_labels, return_counts=True)
print("Unique classes in train_data_labels_channel:", unique)
print("Counts of each class in train_data_labels_channel:", counts)
unique, counts = np.unique(test_data_channel_labels, return_counts=True)
print("Unique classes in test_data_labels_channel:", unique)
print("Counts of each class in test_data_labels_channel:", counts)

# %%

# Define batch size (treating 8 as the batch size)
batch_size = 1
with tf.device('/CPU:0'):
    # Create TensorFlow dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_data_labels))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_data_labels))
    test_dataset = test_dataset.cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)

    train_channel_dataset = tf.data.Dataset.from_tensor_slices((train_data_channel, train_data_channel_labels))
    train_channel_dataset = train_channel_dataset.shuffle(buffer_size=1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_channel_dataset = tf.data.Dataset.from_tensor_slices((test_data_channel, test_data_channel_labels))
    test_channel_dataset = test_channel_dataset.cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)
# Print the shapes of the datasets
for data, labels in train_dataset.take(1):
    print("train_dataset shape:", data.shape, labels.shape)
for data, labels in test_dataset.take(1):
    print("test_dataset shape:", data.shape, labels.shape)
for data, labels in train_channel_dataset.take(1):
    print("train_channel_dataset shape:", data.shape, labels.shape)
for data, labels in test_channel_dataset.take(1):
    print("test_channel_dataset shape:", data.shape, labels.shape)
    


# %%

def ResNet(input_shape=(4, 14, 192),
           output_channels=1,
           dim=64,
           n_downsamplings=1,
           n_blocks=1,
           norm='instance_norm'):
    def _residual_block(x):
        dim = x.shape[-1]
        h = x
        h = keras.layers.Conv2D(dim, 1, strides=1, padding='same', dilation_rate=1)(h)
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
        h = keras.layers.Conv2D(dim, (1, 1), strides=(1, 1), padding='same', dilation_rate=1)(h)
        h = tf.keras.layers.MaxPooling2D(pool_size=(2, 1), strides=(2, 1), padding='same')(h)
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


# %% [markdown]
# Model

# %%
# n_downsamplings = 3
# n_blocks = 4

n_downsamplings = 3
n_blocks = 4

# Explicitly set the batch size to 8
input_shape = (4, 14, 192)
batch_size = 8

num_classes = 2
dim = 192
model = ResNetClassifier(
    input_shape=input_shape,
    num_classes=num_classes,
    dim=dim,
    n_downsamplings=n_downsamplings,
    n_blocks=n_blocks,
    norm='instance_norm'
)

model.summary()

# %%
# Print the model summary
train_dataset = train_dataset
valid_dataset = test_dataset
test_dataset = test_dataset


# %%

# Define directories for saving logs and models
nlos_los_log_dir = os.path.join("logs_nlos_los", datetime.now().strftime("%Y%m%d-%H%M%S"))
nlos_los_save_dir = os.path.join("saved_models_nlos_los", datetime.now().strftime("%Y%m%d-%H%M%S"))
channel_log_dir = os.path.join("logs_channel", datetime.now().strftime("%Y%m%d-%H%M%S"))
channel_save_dir = os.path.join("saved_models_channel", datetime.now().strftime("%Y%m%d-%H%M%S"))
# Create directories if they don't exist
os.makedirs(nlos_los_log_dir, exist_ok=True)
os.makedirs(channel_log_dir, exist_ok=True)
os.makedirs(channel_save_dir, exist_ok=True)
os.makedirs(nlos_los_save_dir, exist_ok=True)


# %%

# # TensorBoard writer
# writer = tf.summary.create_file_writer(nlos_los_log_dir)

# # Define the loss function
# loss_fn = tf.keras.losses.BinaryCrossentropy() if num_classes == 2 else tf.keras.losses.SparseCategoricalCrossentropy()

# # Define the optimizer
# optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)

# # Define metrics
# train_accuracy = BinaryAccuracy() if num_classes == 2 else SparseCategoricalAccuracy()
# val_accuracy = BinaryAccuracy() if num_classes == 2 else SparseCategoricalAccuracy()

# # Optimize dataset loading
# # Define training and validation steps with @tf.function for graph execution
# @tf.function
# def train_step(x_batch, y_batch):
#     with tf.GradientTape() as tape:
#         logits = model(x_batch, training=True)
#         loss = loss_fn(y_batch, logits)
#     gradients = tape.gradient(loss, model.trainable_variables)
#     optimizer.apply_gradients(zip(gradients, model.trainable_variables))
#     train_accuracy.update_state(y_batch, logits)
#     return loss

# @tf.function
# def val_step(x_batch, y_batch):
#     logits = model(x_batch, training=False)
#     loss = loss_fn(y_batch, logits)
#     val_accuracy.update_state(y_batch, logits)
#     return loss

# %%
# # Training loop
# epochs = 100  # Set the number of epochs
# best_loss = float('inf')  # Track the best validation loss


# for epoch in range(epochs):
#     print(f"Epoch {epoch + 1}/{epochs}")
#     epoch_loss = 0.0
#     train_accuracy.reset_state()

#     # Training loop
#     for x_batch, y_batch in train_dataset:
#         loss = train_step(x_batch, y_batch)
#         epoch_loss += loss.numpy()

#     epoch_loss /= len(train_dataset)
#     train_acc = train_accuracy.result().numpy()

#     # Log training metrics to TensorBoard
#     with writer.as_default():
#         tf.summary.scalar("training loss", epoch_loss, step=epoch)
#         tf.summary.scalar("training accuracy", train_acc, step=epoch)

#     print(f"Epoch {epoch + 1} Loss: {epoch_loss:.4f}, Accuracy: {train_acc:.4f}")

#     # Validation loop
#     val_loss = 0.0
#     val_accuracy.reset_state()
#     for x_val, y_val in valid_dataset:
#         loss = val_step(x_val, y_val)
#         val_loss += loss.numpy()

#     val_loss /= len(valid_dataset)
#     val_acc = val_accuracy.result().numpy()

#     # Log validation metrics to TensorBoard
#     with writer.as_default():
#         tf.summary.scalar("validation loss", val_loss, step=epoch)
#         tf.summary.scalar("validation accuracy", val_acc, step=epoch)

#     print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

#     # Save the ONNX model at each epoch
#     onnx_model_path = os.path.join(nlos_los_save_dir, f"model_epoch_{epoch + 1:02d}.onnx")
#     spec = (tf.TensorSpec((None,) + input_shape, tf.float32),)
#     onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=11)
#     with open(onnx_model_path, "wb") as f:
#         f.write(onnx_model.SerializeToString())
#     print(f"Model for epoch {epoch + 1} converted to ONNX and saved at {onnx_model_path}")

#     # Save the best model based on validation loss
#     if val_loss < best_loss or epoch == 0:
#         best_loss = val_loss
#         best_onnx_model_path = os.path.join(nlos_los_save_dir, "best_model.onnx")
#         with open(best_onnx_model_path, "wb") as f:
#             f.write(onnx_model.SerializeToString())
#         print("Best model saved in ONNX format based on validation loss")


# %%
# import onnxruntime as ort  # ONNX Runtime for inference

# # Load the best model
# best_model_path = os.path.join(nlos_los_save_dir, "best_model.onnx")
# if os.path.exists(best_model_path):
#     print(f"Loading best model from {best_model_path}")
#     # Use ONNX Runtime to load the ONNX model
#     ort_session = ort.InferenceSession(best_model_path)
# else:
#     raise FileNotFoundError(f"Best model not found at {best_model_path}")

# # Evaluate the model
# test_loss = 0.0
# test_accuracy = BinaryAccuracy() if num_classes == 2 else SparseCategoricalAccuracy()
# test_accuracy.reset_state()

# y_true = []
# y_pred = []
# y_prob = []

# for x_test, y_test in test_dataset:
#     # Convert input to numpy for ONNX Runtime
#     x_test_np = x_test.numpy()
#     # Run inference using ONNX Runtime
#     test_logits = ort_session.run(None, {ort_session.get_inputs()[0].name: x_test_np})[0]
#     test_loss += loss_fn(y_test, test_logits).numpy()
#     test_accuracy.update_state(y_test, test_logits)

#     # Collect true labels, predicted labels, and probabilities
#     y_true.extend(y_test.numpy())
#     y_prob.extend(tf.nn.softmax(test_logits).numpy() if num_classes > 2 else tf.nn.sigmoid(test_logits).numpy())
#     y_pred.extend(tf.argmax(test_logits, axis=1).numpy() if num_classes > 2 else (test_logits > 0.5).astype(int))

# test_loss /= len(test_dataset)
# test_acc = test_accuracy.result().numpy()

# print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

# # Convert probabilities to a flat list for binary classification
# if num_classes == 2:
#     y_prob = [p[0] for p in y_prob]

# # Visualize metrics
# if num_classes == 2:
#     # Binary classification
#     plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix (Binary Classification)")
#     plot_roc_curve(y_true, y_prob, title="ROC Curve (Binary Classification)")
#     print_metrics(y_true, y_pred, y_prob)
# else:
#     # Multiclass classification
#     classes = list(range(num_classes))  # Replace with actual class names if available
#     plot_confusion_matrix_multiclass(y_true, y_pred, classes, title="Confusion Matrix (Multiclass Classification)")
#     plot_roc_curve_multiclass(y_true, np.array(y_prob), classes, title="ROC Curve (Multiclass Classification)")
#     print_metrics_multiclass(y_true, y_pred, y_prob, classes)

# %%
# n_downsamplings = 3
# n_blocks = 4

n_downsamplings = 3
n_blocks = 4

# Explicitly set the batch size to 8
input_shape = (4, 14, 192)
batch_size = 8

num_classes = np.unique(train_data_channel_labels).shape[0]
dim = 192
model = ResNetClassifier(
    input_shape=input_shape,
    num_classes=num_classes,
    dim=dim,
    n_downsamplings=n_downsamplings,
    n_blocks=n_blocks,
    norm='instance_norm'
    
)

model.summary()

# %%
# TensorBoard writer
writer = tf.summary.create_file_writer(channel_log_dir)

# Define the loss function
loss_fn = tf.keras.losses.BinaryCrossentropy() if num_classes == 2 else tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
# print the loss function
# Define the optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)

# Define metrics
train_accuracy = BinaryAccuracy() if num_classes == 2 else SparseCategoricalAccuracy()
val_accuracy = BinaryAccuracy() if num_classes == 2 else SparseCategoricalAccuracy()
# Training loop
epochs = 100  # Set the number of epochs
best_loss = float('inf')  # Track the best validation loss

train_dataset = train_channel_dataset
valid_dataset = test_channel_dataset
test_dataset = test_channel_dataset
@tf.function
def train_step(x_batch, y_batch):
    with tf.GradientTape() as tape:
        logits = model(x_batch, training=True)
        loss = loss_fn(y_batch, logits)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_accuracy.update_state(y_batch, logits)
    return loss

@tf.function
def val_step(x_batch, y_batch):
    logits = model(x_batch, training=False)
    loss = loss_fn(y_batch, logits)
    val_accuracy.update_state(y_batch, logits)
    return loss


# %%
# Training loop
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    epoch_loss = 0.0
    train_accuracy.reset_state()

    for x_batch, y_batch in train_dataset:
        loss = train_step(x_batch, y_batch)
        
        epoch_loss += loss.numpy()
    
    epoch_loss /= len(train_dataset)
    train_acc = train_accuracy.result().numpy()

    # Log training metrics to TensorBoard
    with writer.as_default():
        tf.summary.scalar("training loss", epoch_loss, step=epoch)
        tf.summary.scalar("training accuracy", train_acc, step=epoch)

    print(f"Epoch {epoch + 1} Loss: {epoch_loss:.4f}, Accuracy: {train_acc:.4f}")

    # Validation loop
    val_loss = 0.0
    val_accuracy.reset_state()
    for x_val, y_val in valid_dataset:
        loss = val_step(x_val, y_val)
        val_loss += loss.numpy()

    val_loss /= len(valid_dataset)
    val_acc = val_accuracy.result().numpy()

    # Log validation metrics to TensorBoard
    with writer.as_default():
        tf.summary.scalar("validation loss", val_loss, step=epoch)
        tf.summary.scalar("validation accuracy", val_acc, step=epoch)

    print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

    # Save the ONNX model at each epoch
    onnx_model_path = os.path.join(channel_save_dir, f"model_epoch_{epoch + 1:02d}.onnx")
    spec = (tf.TensorSpec((None,) + input_shape, tf.float32),)
    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=11)
    with open(onnx_model_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"Model for epoch {epoch + 1} converted to ONNX and saved at {onnx_model_path}")

    # Save the best model based on validation loss
    
    if val_loss < best_loss or epoch == 0:
        best_loss = val_loss
        best_onnx_model_path = os.path.join(channel_save_dir, "best_model.onnx")
        with open(best_onnx_model_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        print("Best model saved in ONNX format based on validation loss")


# %%
# Load the best model
best_model_path = os.path.join(channel_save_dir, "best_model.onnx")
if os.path.exists(best_model_path):
    print(f"Loading best model from {best_model_path}")
    # Use ONNX Runtime to load the ONNX model
    ort_session = ort.InferenceSession(best_model_path)
else:
    raise FileNotFoundError(f"Best model not found at {best_model_path}")

# Evaluate the model
test_loss = 0.0
test_accuracy = BinaryAccuracy() if num_classes == 2 else SparseCategoricalAccuracy()
test_accuracy.reset_state()

y_true = []
y_pred = []
y_prob = []

for x_test, y_test in test_dataset:
    # Convert input to numpy for ONNX Runtime
    x_test_np = x_test.numpy()
    # Run inference using ONNX Runtime
    test_logits = ort_session.run(None, {ort_session.get_inputs()[0].name: x_test_np})[0]
    test_loss += loss_fn(y_test, test_logits).numpy()
    test_accuracy.update_state(y_test, test_logits)

    # Collect true labels, predicted labels, and probabilities
    y_true.extend(y_test.numpy())
    y_prob.extend(tf.nn.softmax(test_logits).numpy() if num_classes > 2 else tf.nn.sigmoid(test_logits).numpy())
    y_pred.extend(tf.argmax(test_logits, axis=1).numpy() if num_classes > 2 else (test_logits > 0.5).astype(int))

test_loss /= len(test_dataset)
test_acc = test_accuracy.result().numpy()

print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

# Convert probabilities to a flat list for binary classification
if num_classes == 2:
    y_prob = [p[0] for p in y_prob]

# Visualize metrics
if num_classes == 2:
    # Binary classification
    plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix (Binary Classification)")
    plot_roc_curve(y_true, y_prob, title="ROC Curve (Binary Classification)")
    print_metrics(y_true, y_pred, y_prob)
else:
    # Multiclass classification
    classes = list(range(num_classes))  # Replace with actual class names if available
    plot_confusion_matrix_multiclass(y_true, y_pred, classes, title="Confusion Matrix (Multiclass Classification)")
    plot_roc_curve_multiclass(y_true, np.array(y_prob), classes, title="ROC Curve (Multiclass Classification)")
    print_metrics_multiclass(y_true, y_pred, y_prob)
    




