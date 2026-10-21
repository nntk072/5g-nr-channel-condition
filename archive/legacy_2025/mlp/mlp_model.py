import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import joblib

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import ModelCheckpoint
import os

# Configure TensorFlow to use the GPU
gpu_num = 1  # Change this to the appropriate GPU number
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)

# Avoid warnings from TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Using GPU: {gpus}")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU found. Using CPU.")

def train_mlp(features, labels, features_test, labels_test):
    # X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    X_train = features
    y_train = labels
    X_test = features_test
    y_test = labels_test
    # mlp = MLPClassifier(hidden_layer_sizes=(128,64,32), max_iter=1000, random_state=42)
    # using batch size of 32
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),  # Three hidden layers with 128, 64, and 32 neurons
        activation='tanh',  # Activation function
        solver='adam',  # Solver for weight optimization
        alpha=0.0001,  # L2 penalty (regularization term) parameter
        batch_size=32,  # Size of minibatches for stochastic optimizers
        learning_rate='adaptive',  # Learning rate schedule for weight updates
        max_iter=1000,  # Maximum number of iterations
        random_state=42,  # Seed for random number generation
        early_stopping=True)
    mlp.fit(X_train, y_train)
    
    y_train_pred = mlp.predict(X_train)
    y_train_prob = mlp.predict_proba(X_train)[:, 1]
    
    y_test_pred = mlp.predict(X_test)
    y_test_prob = mlp.predict_proba(X_test)[:, 1]
    
    return y_train, y_train_pred, y_train_prob, y_test, y_test_pred, y_test_prob, mlp

# def train_mlp_channel(features, labels, features_test, labels_test, model_name = 'mlp_model.pkl'):
#     X_train = features
#     y_train = labels
#     X_test = features_test
#     y_test = labels_test
#     mlp = MLPClassifier(
#         hidden_layer_sizes=(256, 128, 64),  # Three hidden layers with 256, 128, and 64 neurons
#         activation='tanh',  # Activation function
#         solver='adam',  # Solver for weight optimization
#         alpha=0.0001,  # L2 penalty (regularization term) parameter
#         batch_size='auto',  # Size of minibatches for stochastic optimizers
#         learning_rate='adaptive',  # Learning rate schedule for weight updates
#         max_iter=1000,  # Maximum number of iterations
#         random_state=42,  # Seed for random number generation
#         early_stopping=False,  # Whether to use early stopping to terminate training when validation score is not improving
#     )
#     mlp.fit(X_train, y_train)
    
#     y_train_pred = mlp.predict(X_train)
#     y_train_prob = mlp.predict_proba(X_train)
    
#     y_test_pred = mlp.predict(X_test)
#     y_test_prob = mlp.predict_proba(X_test)

#     # save model
#     joblib.dump(mlp, model_name)
#     return y_train, y_train_pred, y_train_prob, y_test, y_test_pred, y_test_prob

def lr_schedule(epoch, lr):
    if epoch < 100:
        return lr  # Keep the initial learning rate for the first 100 epochs
    else:
        decay_rate = 0.0001 / 0.001  # Final LR / Initial LR
        total_epochs = 1000 - 100  # Decay over the remaining epochs
        decay_step = (epoch - 100) / total_epochs
        return 0.001 * (decay_rate ** decay_step)  # Exponential decay

def train_mlp_channel(features, labels, features_test, labels_test, model_name='mlp_model.h5'):
    X_train = features
    y_train = labels
    X_test = features_test
    y_test = labels_test
    
    # Define the MLP model using TensorFlow and Keras
    model = Sequential([
        Dense(128, activation='tanh', input_shape=(X_train.shape[1],)),
        Dense(64, activation='tanh'),
        Dense(32, activation='tanh'),
        Dense(16, activation='tanh'),
        Dense(len(set(y_train)), activation='softmax')  # Output layer with softmax activation for multi-class classification
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',  # Use sparse_categorical_crossentropy for integer labels
        metrics=['accuracy'],
    )
    
    # Define the learning rate scheduler
    lr_scheduler = LearningRateScheduler(lr_schedule)
    
    # Define the ModelCheckpoint callback to save the model at each epoch
    checkpoint_dir = 'mlp_checkpoints'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, 'mlp_model_epoch_{epoch:02d}.h5'),
        save_best_only=False,  # Save the model at every epoch, not just the best one
        save_weights_only=False,  # Save the entire model, not just the weights
        verbose=1
    )
    
    # Train the model
    model.fit(
        X_train, y_train,
        epochs=1000,
        batch_size=32,
        verbose=1,
        callbacks=[lr_scheduler, checkpoint_callback],
        validation_data=(X_test, y_test)
    )
    
    # Get predictions
    y_train_pred = model.predict(X_train).argmax(axis=1)
    y_train_prob = model.predict(X_train)
    
    y_test_pred = model.predict(X_test).argmax(axis=1)
    y_test_prob = model.predict(X_test)
    
    return y_train, y_train_pred, y_train_prob, y_test, y_test_pred, y_test_prob, model