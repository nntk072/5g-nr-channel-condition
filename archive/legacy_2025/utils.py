
import numpy as np
import json
import h5py
    

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

def extract_and_concatenate_with_grouping(dataset):
    y_values = []
    labels = []
    
    for data in dataset:
        y = data['y']  # Shape: (8, 1, 2, 14, 192)
        y_values.append(y)
        
        # Repeat the label for each group (not individual samples)
        if 'label' in data:
            label = data['label']
            labels.append(label)  # One label per group
    
    # Stack y_values along the first axis to preserve grouping
    y_values = np.stack(y_values, axis=0)  # Shape: (num_groups, 8, 1, 2, 14, 192)
    labels = np.array(labels)  # Shape: (num_groups,)
    
    return y_values, labels
