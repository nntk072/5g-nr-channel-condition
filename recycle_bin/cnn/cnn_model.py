# %%
import os
if os.getenv("CUDA_VISIBLE_DEVICES") is None:
    gpu_num = 1 # Use "" to use the CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
    print(f"Using GPU {gpu_num}")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)
# Avoid warnings from TensorFlow
tf.get_logger().setLevel('ERROR')


from cnn.model import ResNetClassifier
from utils import load_features_from_hdf5, extract_and_concatenate_with_grouping

# %%
import numpy as np

# Load features from HDF5 files

mimo_los_test = load_features_from_hdf5('dataset_sionna_mimo_los_test.h5')
mimo_los_train = load_features_from_hdf5('dataset_sionna_mimo_los.h5')
mimo_nlos_test = load_features_from_hdf5('dataset_sionna_mimo_nlos_test.h5')
mimo_nlos_train = load_features_from_hdf5('dataset_sionna_mimo_nlos.h5')
siso_los_test = load_features_from_hdf5('dataset_sionna_siso_los_test.h5')
siso_los_train = load_features_from_hdf5('dataset_sionna_siso_los.h5')
siso_nlos_test = load_features_from_hdf5('dataset_sionna_siso_nlos_test.h5')
siso_nlos_train = load_features_from_hdf5('dataset_sionna_siso_nlos.h5')

# Extract and concatenate for MIMO datasets
train_mimo_los, train_mimo_los_labels = extract_and_concatenate_with_grouping(mimo_los_train)
train_mimo_nlos, train_mimo_nlos_labels = extract_and_concatenate_with_grouping(mimo_nlos_train)
train_siso_los, train_siso_los_labels = extract_and_concatenate_with_grouping(siso_los_train)
train_siso_nlos, train_siso_nlos_labels = extract_and_concatenate_with_grouping(siso_nlos_train)

test_mimo_los, test_mimo_los_labels = extract_and_concatenate_with_grouping(mimo_los_test)
test_mimo_nlos, test_mimo_nlos_labels = extract_and_concatenate_with_grouping(mimo_nlos_test)

test_siso_los, test_siso_los_labels = extract_and_concatenate_with_grouping(siso_los_test)
test_siso_nlos, test_siso_nlos_labels = extract_and_concatenate_with_grouping(siso_nlos_test)



print(train_mimo_los.shape)  # Expected: (2000, 8, 1, 2, 14, 192)
print(train_mimo_los_labels.shape)  # Expected: (2000,)
print(train_mimo_nlos.shape)  # Expected: (2000, 8, 1, 2, 14, 192)
print(train_mimo_nlos_labels.shape)  # Expected: (2000,)
print(train_siso_los.shape)  # Expected: (20, 8, 1, 1, 14, 192)
print(train_siso_los_labels.shape)  # Expected: (20,)
print(train_siso_nlos.shape)  # Expected: (2000, 8, 1, 1, 14, 192)
print(train_siso_nlos_labels.shape)  # Expected: (2000,)

print(test_mimo_los.shape)  # Expected: (8000, 8, 1, 2, 14, 192)
print(test_mimo_los_labels.shape)  # Expected: (8000,)
print(test_mimo_nlos.shape)  # Expected: (1000, 8, 1, 2, 14, 192)
print(test_mimo_nlos_labels.shape)  # Expected: (1000,)
print(test_siso_los.shape)  # Expected: (10, 8, 1, 1, 14, 192)
print(test_siso_los_labels.shape)  # Expected: (10,)
print(test_siso_nlos.shape)  # Expected: (1000, 8, 1, 1, 14, 192)
print(test_siso_nlos_labels.shape)  # Expected: (1000,)


# %%
def extract_and_concatenate_with_grouping(dataset):
    y_tdl = []
    y_cdl = []
    labels_tdl = []
    labels_cdl = []

    # Define the mapping for labels
    label_mapping = {'los': 1, 'nlos': 0}

    for data in dataset:
        y = data['y']  # Shape: (8, 1, 2, 14, 192)
        
        # Split into real and imaginary parts
        y_real = np.real(y)  # Extract real part
        y_imag = np.imag(y)  # Extract imaginary part
        
        # Concatenate real and imaginary parts along the channel dimension
        y_combined = np.concatenate([y_real, y_imag], axis=2)  # Shape: (8, 1, 4, 14, 192)
        tdl = False
        # Check the third dimension is 1 or 2 (after concatenation, it will be 2 or 4)
        if y.shape[2] == 1:
            # duplicate the channel dimension
            y_combined = np.repeat(y_combined, 2, axis=2)
            y_tdl.append(y_combined)
            tdl = True
        elif y.shape[2] == 2:
            y_cdl.append(y_combined)
        
        # Repeat the label for each group (not individual samples)
        if 'label' in data:
            label = data['label']
            label = label_mapping[label.lower()]  # Map 'los' to 1 and 'nlos' to 0
            if tdl:
                labels_tdl.append(label)
            else:
                labels_cdl.append(label)
    
    # Stack y_values along the first axis to preserve grouping
    y_tdl = np.stack(y_tdl, axis=0)  # Shape: (num_groups, 8, 1, 4, 14, 192)
    y_cdl = np.stack(y_cdl, axis=0)  # Shape: (num_groups, 8, 1, 4, 14, 192)
    labels_tdl = np.array(labels_tdl)  # Shape: (num_groups,)
    labels_cdl = np.array(labels_cdl)  # Shape: (num_groups,)
    
    return y_tdl, y_cdl, labels_tdl, labels_cdl


# Extract and concatenate for TDL/CDL dataset
tdl_cdl_test = load_features_from_hdf5('dataset_tdl_cdl_test.h5')
tdl_cdl_train = load_features_from_hdf5('dataset_tdl_cdl.h5')
train_tdls, train_cdls, train_tdls_labels, train_cdls_labels = extract_and_concatenate_with_grouping(tdl_cdl_train)
test_tdls, test_cdls, test_tdls_labels, test_cdls_labels = extract_and_concatenate_with_grouping(tdl_cdl_test)
# Print shape
print(train_tdls.shape)  # Expected: (2000, 8, 1, 1, 14, 192)
print(train_cdls.shape)  # Expected: (2000, 8, 1, 2, 14, 192)
print(train_tdls_labels.shape)  # Expected: (2000,)
print(train_cdls_labels.shape)  # Expected: (2000,)

# %%
# Combine dimensions for MIMO data (e.g., 1 and 2 -> 1 * 2 = 2)
train_mimo_los = train_mimo_los.reshape(train_mimo_los.shape[0], train_mimo_los.shape[1], train_mimo_los.shape[2] * train_mimo_los.shape[3], train_mimo_los.shape[4], train_mimo_los.shape[5])
train_mimo_nlos = train_mimo_nlos.reshape(train_mimo_nlos.shape[0], train_mimo_nlos.shape[1], train_mimo_nlos.shape[2] * train_mimo_nlos.shape[3], train_mimo_nlos.shape[4], train_mimo_nlos.shape[5])
test_mimo_los = test_mimo_los.reshape(test_mimo_los.shape[0], test_mimo_los.shape[1], test_mimo_los.shape[2] * test_mimo_los.shape[3], test_mimo_los.shape[4], test_mimo_los.shape[5])
test_mimo_nlos = test_mimo_nlos.reshape(test_mimo_nlos.shape[0], test_mimo_nlos.shape[1], test_mimo_nlos.shape[2] * test_mimo_nlos.shape[3], test_mimo_nlos.shape[4], test_mimo_nlos.shape[5])

# Combine dimensions for SISO data (e.g., 1 and 1 -> 1 * 1 = 1, effectively removing them)
train_siso_los = train_siso_los.reshape(train_siso_los.shape[0], train_siso_los.shape[1], train_siso_los.shape[2] * train_siso_los.shape[3], train_siso_los.shape[4], train_siso_los.shape[5])
train_siso_nlos = train_siso_nlos.reshape(train_siso_nlos.shape[0], train_siso_nlos.shape[1], train_siso_nlos.shape[2] * train_siso_nlos.shape[3], train_siso_nlos.shape[4], train_siso_nlos.shape[5])
test_siso_los = test_siso_los.reshape(test_siso_los.shape[0], test_siso_los.shape[1], test_siso_los.shape[2] * test_siso_los.shape[3], test_siso_los.shape[4], test_siso_los.shape[5])
test_siso_nlos = test_siso_nlos.reshape(test_siso_nlos.shape[0], test_siso_nlos.shape[1], test_siso_nlos.shape[2] * test_siso_nlos.shape[3], test_siso_nlos.shape[4], test_siso_nlos.shape[5])

# Combine dimensions for TDL/CDL data (e.g., 1 and 1 -> 1 * 1 = 1, effectively removing them)
train_tdls = train_tdls.reshape(train_tdls.shape[0], train_tdls.shape[1], train_tdls.shape[2] * train_tdls.shape[3], train_tdls.shape[4], train_tdls.shape[5])
train_cdls = train_cdls.reshape(train_cdls.shape[0], train_cdls.shape[1], train_cdls.shape[2] * train_cdls.shape[3], train_cdls.shape[4], train_cdls.shape[5])

# Print shapes to verify
print(train_mimo_los.shape)  # Expected: (2000, 8, 2, 14, 192)
print(train_mimo_nlos.shape)  # Expected: (2000, 8, 2, 14, 192)
print(test_mimo_los.shape)  # Expected: (8000, 8, 2, 14, 192)
print(test_mimo_nlos.shape)  # Expected: (1000, 8, 2, 14, 192)

print(train_siso_los.shape)  # Expected: (20, 8, 14, 192)
print(train_siso_nlos.shape)  # Expected: (2000, 8, 14, 192)
print(test_siso_los.shape)  # Expected: (10, 8, 14, 192)
print(test_siso_nlos.shape)  # Expected: (1000, 8, 14, 192)

# %%
import tensorflow as tf
# combine the tdl/cdl datasets for using tf.data later using numpy
train_data = np.stack((train_tdls, train_cdls), axis=0)
test_data = np.stack((test_tdls, test_cdls), axis=0)
train_data_labels = np.stack((train_tdls_labels, train_cdls_labels), axis=0)
test_data_labels = np.stack((test_tdls_labels, test_cdls_labels), axis=0)
# Reshape the data to remove the batch size dimension (8)
train_mimo_los = train_mimo_los.reshape(-1, 2, 14, 192)  # Shape: (2000 * 8, 2, 14, 192)
train_mimo_los_labels = train_mimo_los_labels.repeat(8)  # Repeat labels for each sample in the batch

train_mimo_nlos = train_mimo_nlos.reshape(-1, 2, 14, 192)  # Shape: (2000 * 8, 2, 14, 192)
train_mimo_nlos_labels = train_mimo_nlos_labels.repeat(8)  # Repeat labels for each sample in the batch

test_mimo_los = test_mimo_los.reshape(-1, 2, 14, 192)  # Shape: (8000 * 8, 2, 14, 192)
test_mimo_los_labels = test_mimo_los_labels.repeat(8)  # Repeat labels for each sample in the batch

test_mimo_nlos = test_mimo_nlos.reshape(-1, 2, 14, 192)  # Shape: (1000 * 8, 2, 14, 192)
test_mimo_nlos_labels = test_mimo_nlos_labels.repeat(8)  # Repeat labels for each sample in the batch

train_tdls = train_tdls.reshape(-1, 4, 14, 192)  # Shape: (2000 * 8, 14, 192)
train_tdls_labels = train_tdls_labels.repeat(8)  # Repeat labels for each sample in the batch

train_cdls = train_cdls.reshape(-1, 4, 14, 192)  # Shape: (2000 * 8, 2, 14, 192)
train_cdls_labels = train_cdls_labels.repeat(8)  # Repeat labels for each sample

test_tdls = test_tdls.reshape(-1, 4, 14, 192)  # Shape: (8000 * 8, 14, 192)
test_tdls_labels = test_tdls_labels.repeat(8)  # Repeat labels for each sample in the batch

test_cdls = test_cdls.reshape(-1, 4, 14, 192)  # Shape: (1000 * 8, 14, 192)
test_cdls_labels = test_cdls_labels.repeat(8)  # Repeat labels for each sample in the

train_data = train_data.reshape(-1, 4, 14, 192)  # Shape: (2000 * 8, 14, 192)
train_data_labels = train_data_labels.repeat(8)  # Repeat labels for each sample in the batch

test_data = test_data.reshape(-1, 4, 14, 192)  # Shape: (8000 * 8, 14, 192)
test_data_labels = test_data_labels.repeat(8)  # Repeat labels for each sample in the batch

# Define batch size (treating 8 as the batch size)
batch_size = 8
# with tf.device('/CPU:0'):
    # Create TensorFlow dataset
# tdl_dataset = tf.data.Dataset.from_tensor_slices((train_tdls, train_tdls_labels))
# tdl_dataset = tdl_dataset.shuffle(buffer_size=1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)
# tdl_dataset_test = tf.data.Dataset.from_tensor_slices((test_tdls, test_tdls_labels))
# tdl_dataset_test = tdl_dataset_test.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# cdl_dataset = tf.data.Dataset.from_tensor_slices((train_cdls, train_cdls_labels))
# cdl_dataset = cdl_dataset.shuffle(buffer_size=1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)
# cdl_dataset_test = tf.data.Dataset.from_tensor_slices((test_cdls, test_cdls_labels))
# cdl_dataset_test = cdl_dataset_test.batch(batch_size).prefetch(tf.data.AUTOTUNE)

#tdl/cdl dataset
train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_data_labels))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)
# train_dataset = train_dataset.cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)
test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_data_labels))
test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
# test_dataset = test_dataset.cache().batch(batch_size).prefetch(tf.data.AUTOTUNE)
    # Example: Iterate through the DataLoader
# for batch_data, batch_labels in tdl_dataset.take(1):
#     print("Batch data shape (TDL):", batch_data.shape)  # Expected: (batch_size, 14, 192)
#     print("Batch labels shape (TDL):", batch_labels.shape)  # Expected: (batch_size,)
# for batch_data, batch_labels in cdl_dataset.take(1):
#     print("Batch data shape (CDL):", batch_data.shape)  # Expected: (batch_size, 14, 192)
#     print("Batch labels shape (CDL):", batch_labels.shape)  # Expected: (batch_size,)
# # Example: Iterate through the DataLoader
# for batch_data, batch_labels in tdl_dataset_test.take(1):
#     print("Batch data shape (TDL):", batch_data.shape)  # Expected: (batch_size, 14, 192)
#     print("Batch labels shape (TDL):", batch_labels.shape)  # Expected: (batch_size,)
# for batch_data, batch_labels in cdl_dataset_test.take(1):
#     print("Batch data shape (CDL):", batch_data.shape)  # Expected: (batch_size, 14, 192)
#     print("Batch labels shape (CDL):", batch_labels.shape)  # Expected: (batch_size,)
# # Example: Iterate through the DataLoader
for batch_data, batch_labels in train_dataset.take(1):
    print("Batch data shape (TDL/CDL):", batch_data.shape)  # Expected: (batch_size, 2, 14, 192)
    print("Batch labels shape (TDL/CDL):", batch_labels.shape)  # Expected: (batch_size,)
for batch_data, batch_labels in test_dataset.take(1):
    print("Batch data shape (TDL/CDL):", batch_data.shape)  # Expected: (batch_size, 2, 14, 192)
    print("Batch labels shape (TDL/CDL):", batch_labels.shape)  # Expected: (batch_size,)



# %%
import tensorflow as tf
import keras
from cnn.layers import Oper2D, Oper2DTranspose
from cnn.layers import downsample, upsample, integrate_oper_upsample, integrate_oper_downsample, EncoderMiniBlock, DecoderMiniBlock


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
        h = keras.layers.Conv2D(dim, 3, strides=1, padding='same', dilation_rate=1)(h)
        h = keras.layers.ReLU()(h)  # Use Keras ReLU layer
        # h = keras.layers.Conv2D(dim, 3, strides=1, padding='same', dilation_rate=1)(h)
        # h = keras.layers.ReLU()(h)  # Use Keras ReLU layer
        return keras.layers.add([x, h])

    # 0: Input
    h = inputs = keras.Input(shape=input_shape, batch_size=batch_size)

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

# %%
# n_downsamplings = 3
# n_blocks = 4

n_downsamplings = 2
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
import os
from datetime import datetime
import tensorflow as tf
import tf2onnx
from keras.metrics import BinaryAccuracy, SparseCategoricalAccuracy
from tqdm import tqdm  # For dynamic progress updates


# Define directories for saving logs and models
log_dir = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
model_dir = "saved_models"
os.makedirs(model_dir, exist_ok=True)

# TensorBoard writer
train_writer = tf.summary.create_file_writer(os.path.join(log_dir, "train"))
val_writer = tf.summary.create_file_writer(os.path.join(log_dir, "val"))

# Define the loss function
loss_fn = tf.keras.losses.BinaryCrossentropy() if num_classes == 2 else tf.keras.losses.SparseCategoricalCrossentropy()

# Define the optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)

# Define metrics
train_accuracy = BinaryAccuracy() if num_classes == 2 else SparseCategoricalAccuracy()
val_accuracy = BinaryAccuracy() if num_classes == 2 else SparseCategoricalAccuracy()

# Optimize dataset loading
# Define training and validation steps with @tf.function for graph execution
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

# Training loop
epochs = 100  # Set the number of epochs
best_loss = float('inf')  # Track the best validation loss

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    epoch_loss = 0.0
    train_accuracy.reset_state()

    # Training loop
    for x_batch, y_batch in train_dataset:
        loss = train_step(x_batch, y_batch)
        epoch_loss += loss.numpy()

    epoch_loss /= len(train_dataset)
    train_acc = train_accuracy.result().numpy()

    # Log training metrics to TensorBoard
    with train_writer.as_default():
        tf.summary.scalar("loss", epoch_loss, step=epoch)
        tf.summary.scalar("accuracy", train_acc, step=epoch)

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
    with val_writer.as_default():
        tf.summary.scalar("loss", val_loss, step=epoch)
        tf.summary.scalar("accuracy", val_acc, step=epoch)

    print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

    # Save the model at each epoch
    model.save(os.path.join(model_dir, f"model_epoch_{epoch + 1:02d}.keras"))

    # Save the best model based on validation loss
    if val_loss < best_loss:
        best_loss = val_loss
        model.save(os.path.join(model_dir, "best_model.keras"))
        print("Best model saved based on validation loss")

# Evaluate the model
test_loss = 0.0
test_accuracy = BinaryAccuracy() if num_classes == 2 else SparseCategoricalAccuracy()
test_accuracy.reset_state()

for x_test, y_test in test_dataset:
    test_logits = model(x_test, training=False)
    test_loss += loss_fn(y_test, test_logits).numpy()
    test_accuracy.update_state(y_test, test_logits)
test_loss /= len(test_dataset)
test_acc = test_accuracy.result().numpy()

print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

# Convert the best model to ONNX format
best_model_path = os.path.join(model_dir, "best_model.keras")
onnx_model_path = os.path.join(model_dir, "model.onnx")
model = tf.keras.models.load_model(best_model_path)
spec = (tf.TensorSpec((None,) + input_shape, tf.float32),)
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=11)
with open(onnx_model_path, "wb") as f:
    f.write(onnx_model.SerializeToString())
print(f"Model converted to ONNX and saved at {onnx_model_path}")

# %%
import os
from keras.models import load_model

# Directory where models are saved
model_dir = "saved_models"

# List all saved models
saved_models = sorted([f for f in os.listdir(model_dir) if f.startswith("model_epoch") and f.endswith(".h5")])

# Iterate through each saved model and evaluate on the test dataset
for model_file in saved_models:
    model_path = os.path.join(model_dir, model_file)
    print(f"Loading model: {model_file}")
    
    # Load the model
    model = load_model(model_path)
    
    # Evaluate the model on the test dataset
    loss, accuracy = model.evaluate(test_dataset, verbose=0)
    print(f"Model: {model_file} | Test Loss: {loss:.4f} | Test Accuracy: {accuracy:.4f}")


