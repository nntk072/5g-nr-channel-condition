import pandas as pd
import cmath
import h5py
import json
import numpy as np
import os

def load_dataset(file_path):
    """Load a dataset from a CSV file."""
    return pd.read_csv(file_path)

def combine_datasets(datasets):
    """Combine multiple datasets into a single DataFrame."""
    return pd.concat(datasets, ignore_index=True)

def preprocess_data(df):
    """Preprocess the DataFrame by resetting the index and selecting features."""
    df = df.reset_index()
    # features = df[["snr", "highest_received_power", "kurtosis", "skewness", "mean", "std", "delay_spread"]]
    features = df[["highest_received_power", "kurtosis", "skewness", "mean", "std"]]

    labels = df["label"]
    
    # Convert complex numbers to their absolute values
    features["mean_abs"] = features["mean"].apply(lambda x: abs(complex(x)))
    features["mean_phase"] = features["mean"].apply(lambda x: cmath.phase(complex(x)))    # remove the original mean column
    features = features.drop(columns=["mean"])
    # features["normalized_delay_power_correlation"] = features["normalized_delay_power_correlation"].apply(lambda x: abs(complex(x)))
    
    # Convert labels to binary values
    labels = labels.map({'LOS': 1, 'NLOS': 0})
    
    return features, labels

def load_and_preprocess_datasets(file_paths):
    """Load and preprocess multiple datasets."""
    datasets = [load_dataset(file_path) for file_path in file_paths]
    combined_df = combine_datasets(datasets)
    return preprocess_data(combined_df)


def preprocess_data_channel(df):
    """Preprocess the DataFrame by resetting the index and selecting features."""
    df = df.reset_index()
    features = df[["highest_received_power", "kurtosis", "skewness", "mean", "std"]]
    labels = df["scenario"]
    model_types = df["model_type"]
    # Convert complex numbers to their absolute values
    features["mean_abs"] = features["mean"].apply(lambda x: abs(complex(x)))
    features["mean_phase"] = features["mean"].apply(lambda x: cmath.phase(complex(x)))    # remove the original mean column
    features = features.drop(columns=["mean"])
    
    # Map labels to multiple classes (ABCDE)
    # label_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
    # finding the number of unique labels in scenario column
    # unique_labels = labels.unique()
    # label_mapping = {}
    # for i, label in enumerate(unique_labels):
    #     label_mapping[label] = i
    # labels = labels.map(label_mapping)        
    # The label here is model_type_scenarios, you must combine the model_type and scenario to get the label (now it is in string format, such as scenario: "mimo" and model_type: "mimo")
    labels = model_types + "_" + labels 
    unique_labels = labels.unique()
    label_mapping = {}
    for i, label in enumerate(unique_labels):
        label_mapping[label] = int(i)
    labels = labels.map(label_mapping)
        
    return features, labels

def preprocess_data_ccnn(df):
    # Take only y values as feature
    # dataset now is h5py file, take y and label as features
    df = df.reset_index()
    features = df[["y"]]
    labels = df["label"]
    return features, labels

def preprocess_data_ccnn_channel(df):
    # Take only y values as feature
    df = df.reset_index()
    features = df[["y"]]
    labels = df["scenario"]
    unique_labels = labels.unique()
    label_mapping = {}
    for i, label in enumerate(unique_labels):
        label_mapping[label] = int(i)
    labels = labels.map(label_mapping)
    
    return features, labels

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
            dataset.append(features)
    return dataset

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

def extract_and_concatenate_with_grouping(dataset, without_duplicate=False):
    y_combined_all = []
    labels_all = []
    
    # Define the mapping for labels
    label_mapping = {'los': 1, 'nlos': 0}

    for data in dataset:
        y = data['y']  # Shape: (8, 1, 2, 14, 192)
        
        # Split into real and imaginary parts
        y_real = np.real(y)  # Extract real part
        y_imag = np.imag(y)  # Extract imaginary part
        
        # Concatenate real and imaginary parts along the channel dimension
        y_combined = np.concatenate([y_real, y_imag], axis=2)  # Shape: (8, 1, 4, 14, 192)
        
        # Check if the third dimension is 1 or 2 (after concatenation, it will be 2 or 4)
        if y.shape[2] == 1:
            if not without_duplicate:
                # Duplicate the channel dimension
                y_combined = np.repeat(y_combined, 2, axis=2)
            # Duplicate the channel dimension
            # y_combined = np.repeat(y_combined, 2, axis=2)
        
        # Append the combined data
        y_combined_all.append(y_combined)
        
        # Map the label ('los' to 1, 'nlos' to 0)
        if 'label' in data:
            label = data['label']
            label = label_mapping[label.lower()]  # Map 'los' to 1 and 'nlos' to 0
            labels_all.append(label)

    # Stack y_values and labels along the first axis
    y_combined_all = np.stack(y_combined_all, axis=0)  # Shape: (num_groups, 8, 1, 4, 14, 192)
    labels_all = np.array(labels_all)  # Shape: (num_groups,)

    return y_combined_all, labels_all

def extract_and_concatenate_with_grouping_without_duplicate(dataset, without_duplicate=False):
    y_combined_all = []
    labels_all = []
    
    # Define the mapping for labels
    label_mapping = {'los': 1, 'nlos': 0}

    for data in dataset:
        y = data['y']  # Shape: (8, 1, 2, 14, 192)
        
        # Split into real and imaginary parts
        y_real = np.real(y)  # Extract real part
        y_imag = np.imag(y)  # Extract imaginary part
        
        # Concatenate real and imaginary parts along the channel dimension
        y_combined = np.concatenate([y_real, y_imag], axis=2)  # Shape: (8, 1, 4, 14, 192)
        
        # Check if the third dimension is 1 or 2 (after concatenation, it will be 2 or 4)
        if y.shape[2] == 1:
            # Duplicate the channel dimension
            if not without_duplicate:
                y_combined = np.repeat(y_combined, 2, axis=2)
        
        # Append the combined data
        y_combined_all.append(y_combined)
        
        # Map the label ('los' to 1, 'nlos' to 0)
        if 'label' in data:
            label = data['label']
            label = label_mapping[label.lower()]  # Map 'los' to 1 and 'nlos' to 0
            labels_all.append(label)

    # Stack y_values and labels along the first axis
    y_combined_all = np.stack(y_combined_all, axis=0)  # Shape: (num_groups, 8, 1, 4, 14, 192)
    labels_all = np.array(labels_all)  # Shape: (num_groups,)

    return y_combined_all, labels_all

def extract_and_concatenate_with_grouping_tdl_cdl(dataset, without_duplicate=False):
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
            if not without_duplicate:
                # Duplicate the channel dimension
                y_combined = np.repeat(y_combined, 2, axis=2)
            # duplicate the channel dimension
            # y_combined = np.repeat(y_combined, 2, axis=2)
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
def extract_and_concatenate_with_grouping_tdl_cdl_without_duplicate(dataset):
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
        # use the data['model_type'] to define tdl or cdl, and then append the data
        if data['model_type'] == 'tdl':
            y_tdl.append(y_combined)
            tdl = True
        elif data['model_type'] == 'cdl':
            y_cdl.append(y_combined)
            tdl = False
        
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
def extract_and_concatenate_with_grouping_channel(dataset, without_duplicate=False):
    y_combined_all = []
    labels_all = []

    # Create a mapping for combined labels
    unique_labels = set()
    for data in dataset:
        model_type = data['model_type'].lower()  # e.g., 'tdl' or 'cdl'
        scenario = data['scenario'].lower()  # e.g., 'a', 'b', etc.
        combined_label = f"{model_type}_{scenario}"
        unique_labels.add(combined_label)

    # Assign unique integer IDs to each combined label
    label_mapping = {label: idx for idx, label in enumerate(sorted(unique_labels))}

    for data in dataset:
        y = data['y']  # Shape: (8, 1, 2, 14, 192)

        # Split into real and imaginary parts
        y_real = np.real(y)  # Extract real part
        y_imag = np.imag(y)  # Extract imaginary part

        # Concatenate real and imaginary parts along the channel dimension
        y_combined = np.concatenate([y_real, y_imag], axis=2)  # Shape: (8, 1, 4, 14, 192)

        # Check the third dimension is 1 or 2 (after concatenation, it will be 2 or 4)
        if y.shape[2] == 1:
            # Duplicate the channel dimension
            if not without_duplicate:
                y_combined = np.repeat(y_combined, 2, axis=2)
            # y_combined = np.repeat(y_combined, 2, axis=2)

        # Append the combined data
        y_combined_all.append(y_combined)

        # Generate the combined label
        model_type = data['model_type'].lower()
        scenario = data['scenario'].lower()
        combined_label = f"{model_type}_{scenario}"
        label = label_mapping[combined_label]  # Map to unique integer ID

        # Append the label
        
        labels_all.append(label)

    # Stack y_values and labels along the first axis
    y_combined_all = np.stack(y_combined_all, axis=0)  # Shape: (num_groups, 8, 1, 4, 14, 192)
    labels_all = np.array(labels_all)  # Shape: (num_groups,)

    return y_combined_all, labels_all

def extract_and_concatenate_with_grouping_channel_tdl_cdl(dataset):
    y_tdl = []
    y_cdl = []
    label_tdl = []
    label_cdl = []

    # Create a mapping for combined labels
    unique_labels = set()
    for data in dataset:
        model_type = data['model_type'].lower()  # e.g., 'tdl' or 'cdl'
        scenario = data['scenario'].lower()  # e.g., 'a', 'b', etc.
        combined_label = f"{model_type}_{scenario}"
        unique_labels.add(combined_label)

    # Assign unique integer IDs to each combined label
    label_mapping = {label: idx for idx, label in enumerate(sorted(unique_labels))}

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
            # Duplicate the channel dimension
            y_combined = np.repeat(y_combined, 2, axis=2)
            y_tdl.append(y_combined)
            tdl = True
        elif y.shape[2] == 2:
            y_cdl.append(y_combined)

        # Generate the combined label
        model_type = data['model_type'].lower()
        scenario = data['scenario'].lower()
        combined_label = f"{model_type}_{scenario}"
        label = label_mapping[combined_label]  # Map to unique integer ID
        # label_list.append(label)
        if tdl:
            label_tdl.append(label)
        else:
            label_cdl.append(label)

    # Stack y_values and labels along the first axis
    y_tdl = np.stack(y_tdl, axis=0)  # Shape: (num_groups, 8, 1, 4, 14, 192)
    y_cdl = np.stack(y_cdl, axis=0)  # Shape: (num_groups, 8, 1, 4, 14, 192)
    # label_list = np.array(label_list)  # Shape: (num_groups,)
    label_tdl = np.array(label_tdl)  # Shape: (num_groups,)
    label_cdl = np.array(label_cdl)  # Shape: (num_groups,)
    return y_tdl, y_cdl, label_tdl, label_cdl


def extract_and_concatenate_with_grouping_channel_tdl_cdl_without_duplicate(dataset):
    y_tdl = []
    y_cdl = []
    label_tdl = []
    label_cdl = []

    # Create a mapping for combined labels
    unique_labels = set()
    for data in dataset:
        model_type = data['model_type'].lower()  # e.g., 'tdl' or 'cdl'
        scenario = data['scenario'].lower()  # e.g., 'a', 'b', etc.
        combined_label = f"{model_type}_{scenario}"
        unique_labels.add(combined_label)

    # Assign unique integer IDs to each combined label
    label_mapping = {label: idx for idx, label in enumerate(sorted(unique_labels))}

    for data in dataset:
        y = data['y']  # Shape: (8, 1, 2, 14, 192)

        # Split into real and imaginary parts
        y_real = np.real(y)  # Extract real part
        y_imag = np.imag(y)  # Extract imaginary part

        # Concatenate real and imaginary parts along the channel dimension
        y_combined = np.concatenate([y_real, y_imag], axis=2)  # Shape: (8, 1, 4, 14, 192)
        
        # Check the third dimension is 1 or 2 (after concatenation, it will be 2 or 4)

        # Generate the combined label
        model_type = data['model_type'].lower()
        scenario = data['scenario'].lower()
        combined_label = f"{model_type}_{scenario}"
        
        # using the model type to define tdl or cdl, and then append the data
        if model_type == 'tdl':
            y_tdl.append(y_combined)
            tdl = True
        elif model_type == 'cdl':
            y_cdl.append(y_combined)
            tdl = False
        label = label_mapping[combined_label]  # Map to unique integer ID
        # label_list.append(label)
        if tdl:
            label_tdl.append(label)
        else:
            label_cdl.append(label)

    # Stack y_values and labels along the first axis
    y_tdl = np.stack(y_tdl, axis=0)  # Shape: (num_groups, 8, 1, 4, 14, 192)
    y_cdl = np.stack(y_cdl, axis=0)  # Shape: (num_groups, 8, 1, 4, 14, 192)
    # label_list = np.array(label_list)  # Shape: (num_groups,)
    label_tdl = np.array(label_tdl)  # Shape: (num_groups,)
    label_cdl = np.array(label_cdl)  # Shape: (num_groups,)
    return y_tdl, y_cdl, label_tdl, label_cdl

def process_for_complex_cnn(dataset):
    y_combined_all = []
    labels_all = []
    
    # Define the mapping for labels
    label_mapping = {'los': 1, 'nlos': 0}

    for data in dataset:
        y = data['y']  # Shape: (8, 1, 2, 14, 192)
        
        # Check if the third dimension is 1 or 2
        # if y.shape[2] == 1:
        #     # Duplicate the channel dimension
        #     y = np.repeat(y, 2, axis=2)
        
        # Append the combined data
        y_combined_all.append(y)
        
        # Map the label ('los' to 1, 'nlos' to 0)
        if 'label' in data:
            label = data['label']
            label = label_mapping[label.lower()]  # Map 'los' to 1 and 'nlos' to 0
            labels_all.append(label)

    # Stack y_values and labels along the first axis
    y_combined_all = np.stack(y_combined_all, axis=0)  # Shape: (num_groups, 8, 1, 2, 14, 192)
    labels_all = np.array(labels_all)  # Shape: (num_groups,)

    return y_combined_all, labels_all

def process_for_complex_cnn_tdl_cdl(dataset):
    y_tdl = []
    y_cdl = []
    labels_tdl = []
    labels_cdl = []

    # Define the mapping for labels
    label_mapping = {'los': 1, 'nlos': 0}

    for data in dataset:
        y = data['y']  # Shape: (8, 1, 2, 14, 192), complex-valued

        tdl = False
        # Use the data['model_type'] to define tdl or cdl, and then append the data
        if data['model_type'].lower() == 'tdl':
            y_tdl.append(y)
            tdl = True
        elif data['model_type'].lower() == 'cdl':
            y_cdl.append(y)
            tdl = False

        # Repeat the label for each group (not individual samples)
        if 'label' in data:
            label = data['label']
            label = label_mapping[label.lower()]  # Map 'los' to 1 and 'nlos' to 0
            if tdl:
                labels_tdl.append(label)
            else:
                labels_cdl.append(label)

    # Stack y_values along the first axis to preserve grouping
    y_tdl = np.stack(y_tdl, axis=0) if y_tdl else np.empty((0,))
    y_cdl = np.stack(y_cdl, axis=0) if y_cdl else np.empty((0,))
    labels_tdl = np.array(labels_tdl)
    labels_cdl = np.array(labels_cdl)

    return y_tdl, y_cdl, labels_tdl, labels_cdl

def process_for_complex_cnn_channel(dataset):
    y_combined_all = []
    labels_all = []

    # Create a mapping for combined labels
    unique_labels = set()
    for data in dataset:
        model_type = data['model_type'].lower()  # e.g., 'tdl' or 'cdl'
        scenario = data['scenario'].lower()  # e.g., 'a', 'b', etc.
        combined_label = f"{model_type}_{scenario}"
        unique_labels.add(combined_label)

    # Assign unique integer IDs to each combined label
    label_mapping = {label: idx for idx, label in enumerate(sorted(unique_labels))}

    for data in dataset:
        y = data['y']  # Shape: (8, 1, 2, 14, 192)

        # # Check if the third dimension is 1 or 2
        # if y.shape[2] == 1:
        #     # Duplicate the channel dimension
        #     y = np.repeat(y, 2, axis=2)

        # Append the combined data
        y_combined_all.append(y)

        # Generate the combined label
        model_type = data['model_type'].lower()
        scenario = data['scenario'].lower()
        combined_label = f"{model_type}_{scenario}"
        label = label_mapping[combined_label]  # Map to unique integer ID

        # Append the label
        labels_all.append(label)

    # Stack y_values and labels along the first axis
    y_combined_all = np.stack(y_combined_all, axis=0)  # Shape: (num_groups, 8, 1, 2, 14, 192)
    labels_all = np.array(labels_all)  # Shape: (num_groups,)

    return y_combined_all, labels_all

def process_for_complex_cnn_channel_tdl_cdl(dataset):
    y_tdl = []
    y_cdl = []
    label_tdl = []
    label_cdl = []

    # Create a mapping for combined labels
    unique_labels = set()
    for data in dataset:
        model_type = data['model_type'].lower()  # e.g., 'tdl' or 'cdl'
        scenario = data['scenario'].lower()      # e.g., 'a', 'b', etc.
        combined_label = f"{model_type}_{scenario}"
        unique_labels.add(combined_label)

    # Assign unique integer IDs to each combined label
    label_mapping = {label: idx for idx, label in enumerate(sorted(unique_labels))}

    for data in dataset:
        y = data['y']  # Shape: (8, 1, 2, 14, 192), complex-valued

        model_type = data['model_type'].lower()
        scenario = data['scenario'].lower()
        combined_label = f"{model_type}_{scenario}"
        label = label_mapping[combined_label]

        if model_type == 'tdl':
            y_tdl.append(y)
            label_tdl.append(label)
        elif model_type == 'cdl':
            y_cdl.append(y)
            label_cdl.append(label)

    y_tdl = np.stack(y_tdl, axis=0) if y_tdl else np.empty((0,))
    y_cdl = np.stack(y_cdl, axis=0) if y_cdl else np.empty((0,))
    label_tdl = np.array(label_tdl)
    label_cdl = np.array(label_cdl)
    return y_tdl, y_cdl, label_tdl, label_cdl