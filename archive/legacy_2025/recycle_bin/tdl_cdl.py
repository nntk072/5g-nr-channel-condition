# %%
import os # Configure which GPU

if os.getenv("CUDA_VISIBLE_DEVICES") is None:
    # gpu_num = 0 # Use "" to use the CPU
    # os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import Sionna
try:
    import sionna
except ImportError as e:
    # Install Sionna if package is not already installed
    import os
    os.system("pip install sionna")
    import sionna

# Configure the notebook to use only a single GPU and allocate only as much memory as needed
# For more details, see https://www.tensorflow.org/guide/gpu
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)
# Avoid warnings from TensorFlow
tf.get_logger().setLevel('ERROR')

# Colab does currently not support the latest version of ipython.
# Thus, the preview does not work in Colab. However, whenever possible we
# strongly recommend to use the scene preview mode.
try: # detect if the notebook runs in Colab
    import google.colab
    no_preview = True # deactivate preview
except:
    if os.getenv("SIONNA_NO_PREVIEW"):
        no_preview = True
    else:
        no_preview = False

resolution = [480,320] # increase for higher quality of renderings

import matplotlib.pyplot as plt
import numpy as np
import time
import pandas as pd

# Set random seed for reproducibility

# Import Sionna RT components
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, Camera, Paths

# For link-level simulations
from sionna.channel import cir_to_ofdm_channel, subcarrier_frequencies, OFDMChannel, ApplyOFDMChannel, CIRDataset
from sionna.nr import PUSCHConfig, PUSCHTransmitter, PUSCHReceiver
from sionna.utils import compute_ber, ebnodb2no, PlotBER

from nr_pusch.model import Model, CIRGenerator, PUSCHModel
from nr_pusch.configs import SISOConfig, MIMOConfig

sionna.config.seed = 42
tf.random.set_seed(42)
np.random.seed(42)

# %%
# Define the number of samples per scenario and model type
num_samples_per_scenario = 2

# Initialize the dataset
dataset = []


# Define the scenarios and Eb/No values
scenarios = ["A", "B", "C", "D", "E"]
ebno_min = -3
ebno_max = 19

model_types = ["tdl","cdl"]

mimo_configuration = MIMOConfig(
    num_prb=16,
    mcs_index=14,
    num_layers=1,
    mcs_table=1,
    domain="freq",
    num_tx_ant=4,
    num_tx=1,
    subcarrier_spacing= 15e3,
    num_rx_ant=2,
    num_rx=1,
    num_symbols=14,
    batch_size=1,
    min_speed = 0,
    max_speed = 10
)
# # System parameters
# subcarrier_spacing = 30e3
# num_time_steps = 14 # Total number of ofdm symbols per slot

siso_configuration = SISOConfig(
    batch_size=1,
    num_rx=1,
    num_rx_ant=1,
    num_tx=1,
    num_tx_ant=1,
    subcarrier_spacing=15e3,
    num_time_steps=14,
    num_layers=1,
    min_speed = 0,
    max_speed = 10
)
# Loop through each model type
for model_type in model_types:
    # Loop through each scenario
    for scenario in scenarios:
        if model_type == "tdl":
            model = Model(
                perfect_csi=False,
                detector=None,
                config=siso_configuration,
                model_type=model_type,
                scenario=scenario,
            )
            # continue
        else:
            model = Model(
                perfect_csi=False,
                detector=None,
                # config=mimo_configuration,
                config=siso_configuration,
                model_type=model_type,
                scenario=scenario,
            )
        # Loop through the number of samples per scenario
        for sample_idx in range(num_samples_per_scenario):
            ebno_db = np.random.uniform(ebno_min, ebno_max)
            print(f"Generating sample {sample_idx + 1}/{num_samples_per_scenario} for scenario {scenario} and model type {model_type}")

            # Randomly select an Eb/No value (Monte Carlo sampling)

            # Extract features (Simulation)
            features = model.extract_features(ebno_db=ebno_db)
            # print(features["y"].shape)
            
            # Add additional metadata
            features["scenario"] = scenario
            features["ebno_db"] = ebno_db
            features["model_type"] = model_type
            # print shape of features y here
            if scenario == "A" or scenario == "B" or scenario == "C":
                features["label"] = "NLOS"
            else:
                features["label"] = "LOS"
            # Append the features to the dataset (Aggregation)
            dataset.append(features)


# %%



# %%
num_samples_per_scenario = 1
dataset_test = []
# Generate Eb/No values with a step size of 1000
ebno_values = np.linspace(ebno_min, ebno_max, num_samples_per_scenario)

# Loop through each model type
for model_type in model_types:
    # Loop through each scenario
    for scenario in scenarios:
        if model_type == "tdl":
            model = Model(
                perfect_csi=False,
                detector=None,
                config=siso_configuration,
                model_type=model_type,
                scenario=scenario,
            )
        else:
            model = Model(
                perfect_csi=False,
                detector=None,
                # config=mimo_configuration,
                config=siso_configuration,
                model_type=model_type,
                scenario=scenario,
            )
        # Loop through the number of samples per scenario
        for sample_idx, ebno_db in enumerate(ebno_values):
            print(f"Generating sample {sample_idx + 1}/{num_samples_per_scenario} for scenario {scenario} and model type {model_type}")

            # Extract features (Simulation)
            features = model.extract_features(ebno_db=ebno_db)
            print(features["y"].shape)
            # Add additional metadata
            features["scenario"] = scenario
            features["ebno_db"] = ebno_db
            features["model_type"] = model_type
            if scenario in ["A", "B", "C"]:
                features["label"] = "NLOS"
            else:
                features["label"] = "LOS"
            # Append the features to the dataset (Aggregation)
            dataset_test.append(features)


# %%
import h5py
import numpy as np
import pandas as pd
import json

# Function to convert non-serializable types to serializable types
def convert_to_serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    elif isinstance(value, (np.float32, np.float64)):
        return float(value)
    elif isinstance(value, (np.int32, np.int64)):
        return int(value)
    elif isinstance(value, (np.complex64, np.complex128)):
        return [value.real, value.imag]
    elif isinstance(value, (np.bool_)):
        return bool(value)
    else:
        return str(value)

# Save the 'y' feature and additional features of the dataset to an HDF5 file
def save_features_to_hdf5(filename, dataset):
    with h5py.File(filename, 'w') as f:
        for i, features in enumerate(dataset):
            grp = f.create_group(str(i))
            if 'y' in features:
                grp.create_dataset('y', data=features['y'])
            if 'ebno_db' in features:
                grp.attrs['ebno_db'] = json.dumps(convert_to_serializable(features['ebno_db']))
            if 'model_type' in features:
                grp.attrs['model_type'] = features['model_type']
            if 'label' in features:
                grp.attrs['label'] = features['label']
            if 'scenario' in features:
                grp.attrs['scenario'] = features['scenario']

# Load the 'y' feature and additional features of the dataset from an HDF5 file
def load_features_from_hdf5(filename):
    dataset = []
    with h5py.File(filename, 'r') as f:
        for key in f.keys():
            grp = f[key]
            features = {}
            features['y'] = grp['y'][:]
            if 'ebno_db' in grp.attrs:
                features['ebno_db'] = json.loads(grp.attrs['ebno_db'])
            if 'model_type' in grp.attrs:
                features['model_type'] = grp.attrs['model_type']
            if 'label' in grp.attrs:
                features['label'] = grp.attrs['label']
            if 'scenario' in grp.attrs:
                features['scenario'] = grp.attrs['scenario']
            dataset.append(features)
    return dataset
# Convert the dataset to a pandas DataFrame
df = pd.DataFrame(dataset)
df.head()

# Save the dataset to a CSV file
df.to_csv("dataset_tdl_cdl_training_siso.csv", index=False)
# # Save the dataset to a CSV file
df = pd.DataFrame(dataset_test)
df.to_csv("dataset_tdl_cdl_testing.csv", index=False)
# Example usage
# Save the features of the dataset
save_features_to_hdf5('dataset_tdl_cdl_siso.h5', dataset)
save_features_to_hdf5('dataset_tdl_cdl_test_siso.h5', dataset_test)



