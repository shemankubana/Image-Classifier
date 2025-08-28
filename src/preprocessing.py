import os
import numpy as np
import tensorflow as tf
from keras.datasets import cifar10
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split # New import for splitting data
import pickle

# Define paths relative to the project root
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'validation') # New directory for validation data
TEST_DIR = os.path.join(DATA_DIR, 'test')

def load_and_preprocess_data(val_split_ratio=0.15): # Added val_split_ratio parameter
    """
    Loads the CIFAR-10 dataset, preprocesses it (normalization, one-hot encoding),
    splits the training data into training and validation sets,
    and saves all sets into their respective directories as numpy arrays.
    """
    print("Loading CIFAR-10 dataset...")
    # Load the CIFAR-10 dataset
    (x_train_full, y_train_full), (x_test, y_test) = cifar10.load_data()
    print("CIFAR-10 dataset loaded.")

    # Normalize pixel values to be between 0 and 1
    x_train_full = x_train_full.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    print("Data normalized.")

    # One-hot encode the labels
    num_classes = 10
    y_train_full_one_hot = to_categorical(y_train_full, num_classes)
    y_test_one_hot = to_categorical(y_test, num_classes)
    print("Labels one-hot encoded.")

    # Split the full training data into training and validation sets
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full, y_train_full_one_hot,
        test_size=val_split_ratio,
        random_state=42, # for reproducibility
        stratify=y_train_full_one_hot # ensure class distribution is maintained
    )
    print(f"Training data split into train ({1 - val_split_ratio:.0%}) and validation ({val_split_ratio:.0%}).")

    # Create directories if they don't exist
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VAL_DIR, exist_ok=True) # Create validation directory
    os.makedirs(TEST_DIR, exist_ok=True)

    # Save processed data as numpy arrays
    np.save(os.path.join(TRAIN_DIR, 'x_train.npy'), x_train)
    np.save(os.path.join(TRAIN_DIR, 'y_train.npy'), y_train)
    np.save(os.path.join(VAL_DIR, 'x_val.npy'), x_val) # Save validation data
    np.save(os.path.join(VAL_DIR, 'y_val.npy'), y_val) # Save validation data
    np.save(os.path.join(TEST_DIR, 'x_test.npy'), x_test)
    np.save(os.path.join(TEST_DIR, 'y_test.npy'), y_test_one_hot)
    print(f"Processed data saved to {TRAIN_DIR}, {VAL_DIR} and {TEST_DIR}")

    return x_train, y_train, x_val, y_val, x_test, y_test_one_hot

def load_processed_data():
    """
    Loads preprocessed data from the data/train, data/validation, and data/test directories.
    If data is not found, it calls load_and_preprocess_data() to create it.
    """
    x_train_path = os.path.join(TRAIN_DIR, 'x_train.npy')
    y_train_path = os.path.join(TRAIN_DIR, 'y_train.npy')
    x_val_path = os.path.join(VAL_DIR, 'x_val.npy') # Path for validation data
    y_val_path = os.path.join(VAL_DIR, 'y_val.npy') # Path for validation data
    x_test_path = os.path.join(TEST_DIR, 'x_test.npy')
    y_test_path = os.path.join(TEST_DIR, 'y_test.npy')

    if (os.path.exists(x_train_path) and os.path.exists(y_train_path) and
        os.path.exists(x_val_path) and os.path.exists(y_val_path) and # Check validation paths
        os.path.exists(x_test_path) and os.path.exists(y_test_path)):
        print("Loading preprocessed data from disk...")
        x_train = np.load(x_train_path)
        y_train = np.load(y_train_path)
        x_val = np.load(x_val_path) # Load validation data
        y_val = np.load(y_val_path) # Load validation data
        x_test = np.load(x_test_path)
        y_test = np.load(y_test_path)
        print("Preprocessed data loaded.")
    else:
        print("Preprocessed data not found. Running initial data acquisition and preprocessing...")
        x_train, y_train, x_val, y_val, x_test, y_test = load_and_preprocess_data()

    return x_train, y_train, x_val, y_val, x_test, y_test

if __name__ == "__main__":
    print("Running preprocessing.py directly for initial data setup.")
    x_train, y_train, x_val, y_val, x_test, y_test = load_processed_data()
    print(f"x_train shape: {x_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"x_val shape: {x_val.shape}")
    print(f"y_val shape: {y_val.shape}")
    print(f"x_test shape: {x_test.shape}")
    print(f"y_test shape: {y_test.shape}")