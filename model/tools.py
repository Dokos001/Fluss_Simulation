import os
import re
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from model.DataGeneration import DataGenerator
from sklearn.model_selection import train_test_split
import h5py

SAVE_ADDON = "_static_Receiver"
UNIQUE_ADDON = "_Unique"
output_dir = "Datasets"


def generate_Timeline(t_start, t_stop, t_step):
    t = np.arange(t_start, t_stop, t_step)
    return t

def display_train_val_loss():
    df = pd.read_csv('log.csv', delimiter = ';')
    trainloss = np.array(df['loss'])
    valloss = np.array(df['val_loss'])
    plt.plot(trainloss, 'b')
    plt.plot(valloss, color = 'orange')
    plt.title("Verlauf der Loss Funktion")
    plt.legend(['Training','Validierung'])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig('train_and_val_loss.png')
    plt.show()
    
def create_MLDataset(dataset_path, dist_sequenzes_noisy, sequenzes, test_size, random_state, time_variable = True):
    
    X_test_pure, y_test_pure = shuffle(dist_sequenzes_noisy, sequenzes, random_state=random_state)
    
    X_train, X_test, y_train, y_test = train_test_split(
        dist_sequenzes_noisy, sequenzes, test_size=test_size, random_state=random_state
    )
    
    X_train, X_val, y_train, y_val                  = train_test_split(X_train, y_train, test_size=0.20, random_state=random_state)
    X_train, X_test, y_train, y_test, X_val, y_val  = map(np.array, [X_train, X_test, y_train, y_test, X_val, y_val])
    X_train = (X_train - np.mean(X_train)) / np.std(X_train)
    X_test  = (X_test - np.mean(X_test)) / np.std(X_test)
    X_val   = (X_val - np.mean(X_val)) / np.std(X_val)
    X_test_pure  = (X_test_pure - np.mean(X_test_pure)) / np.std(X_test_pure)
    

    splits = {
    "X_train": X_train,
    "y_train": y_train,
    "X_test": X_test,
    "y_test": y_test,
    "X_val": X_val,
    "y_val": y_val,
    "X_test_pure": X_test_pure,
    "y_test_pure": y_test_pure,
}

    for name, data in splits.items():
        pd.DataFrame(data).to_csv(
            os.path.join(dataset_path, f"{name}.csv"),
            header=False,
            index=False
        )

    return X_train, X_test, y_train, y_test, X_val, y_val, X_test_pure, y_test_pure

def load_MLDataset(dataset_path):
    

    X_train = pd.read_csv(os.path.join(dataset_path,"X_train"), header=None).to_numpy()
    y_train = pd.read_csv(os.path.join(dataset_path,"y_train"), header=None).to_numpy()
    X_test  = pd.read_csv(os.path.join(dataset_path,"X_test"), header=None).to_numpy()
    y_test  = pd.read_csv(os.path.join(dataset_path,"y_test"), header=None).to_numpy()
    X_val   = pd.read_csv(os.path.join(dataset_path,"X_val"), header=None).to_numpy()
    y_val   = pd.read_csv(os.path.join(dataset_path,"y_val"), header=None).to_numpy()

    return X_train, X_test, y_train, y_test, X_val, y_val

def load_pureTest_MLDataset(dataset_path):
    X_test  = pd.read_csv(os.path.join(dataset_path,"X_test_pure"), header=None).to_numpy()
    y_test  = pd.read_csv(os.path.join(dataset_path,"y_test_pure"), header=None).to_numpy()

    return X_test, y_test


def load_RawData_from_hdf5(filename):
    dist_sequenzes = []
    dist_sequenzes_noisy = []
    ideal_sequenzes = []
    ideal_sequenzes_noisy = []
    sequenzes = []

    with h5py.File(filename, 'r') as hf:
        # Disturbed Signals
        grp = hf['disturbed_signals']
        dist_sequenzes = [sig for sig in grp['data']]
        dist_sequenzes_noisy = [sig for sig in grp['data_noisy']]

        # Ideal Signals
        grp = hf['ideal_signals']
        ideal_sequenzes = [sig for sig in grp['data']]
        ideal_sequenzes_noisy = [sig for sig in grp['data_noisy']]

        # Original Sequences
        grp = hf['original_sequences']
        sequenzes = [sig for sig in grp['data']]

    return dist_sequenzes, dist_sequenzes_noisy, ideal_sequenzes, ideal_sequenzes_noisy, sequenzes


def save_complete_dataset(dist_sequenzes, dist_sequenzes_noisy, ideal_sequenzes, ideal_sequenzes_noisy, sequenzes, dataset_name, cfg):

        with h5py.File(dataset_name, 'w') as hf:

             # --- include MetaData ---
            hf.attrs["dataset_name"] = dataset_name
            for k, v in cfg.items():
                hf.attrs[k] = v
            grp = hf.create_group('disturbed_signals')
            grp.create_dataset('data', data=dist_sequenzes)
            grp.create_dataset('data_noisy', data=dist_sequenzes_noisy)

            grp = hf.create_group('ideal_signals')
            grp.create_dataset('data', data=ideal_sequenzes)
            grp.create_dataset('data_noisy', data=ideal_sequenzes_noisy)

            grp = hf.create_group('original_sequences')
            grp.create_dataset('data', data=sequenzes)

        print("Complete dataset saved to RawDatasets")

def sanitize(x):
    return re.sub(r'\D', '', f"{x:.4g}")

def generate_RawDataset_name(cfg):
    save_dir = "./RawDatasets"
    os.makedirs(save_dir, exist_ok=True)

    dataset_name = (
        f"b{cfg['NUMBER_OF_BITS']}"
        f"_f{sanitize(cfg['F_RX'])}"
        f"_v{sanitize(cfg['V_0'])}"
        f"_dz{sanitize(cfg['DZ'])}"
        f"_U{sanitize(cfg['U'])}"
        f"_r{sanitize(cfg['CHANNEL_RADIUS'])}"
        f"_zo{sanitize(cfg['Z_OFFSET'])}"
        f"_z{sanitize(cfg['Z_DEPTH'])}"
        f"_rate{sanitize(cfg['BIT_RATE'])}"
        f"_duration{sanitize(cfg['T_STOP'] - cfg['T_START'])}"
        f"_{'3d' if cfg['RECEIVER_DIMENSION_3D'] else '2d'}"
    )

    dataset_name += "_complete_dataset"
    dataset_name = os.path.join(save_dir, dataset_name)
    dataset_name += ".h5"
    return dataset_name

def generate_MLDataset_name(cfg):
    save_dir = "./MLDatasets"
    os.makedirs(save_dir, exist_ok=True)

    filter_str = "-".join(str(f) for f in cfg["FILTERS"])

    model_name = (
        f"bs{sanitize(cfg['BATCH_SIZE'])}"
        f"_ep{sanitize(cfg['EPOCHS'])}"
        f"_arrays{sanitize(cfg['NUMBER_OF_ARRAYS'])}"
        f"_bits{sanitize(cfg['NUMBER_OF_BITS'])}"
        f"_lr{sanitize(cfg['LEARNING_RATE'])}"
        f"_filters{filter_str}"
        f"_conv{sanitize(cfg['NUM_OF_CONV_LAYERS'])}"
        f"_lstmU{sanitize(cfg['LSTM_UNITS'])}"
        f"_lstmL{sanitize(cfg['LSTM_LAYERS'])}"
        f"_do{sanitize(cfg['DROPOUT'])}"
    )

    model_name = os.path.join(save_dir, model_name)
    return model_name

def generate_Model_name(cfg):
    save_dir = "./NNModels"
    os.makedirs(save_dir, exist_ok=True)

    dataset_name = (
        f"b{cfg['NUMBER_OF_BITS']}"
        f"_f{sanitize(cfg['F_RX'])}"
        f"_v{sanitize(cfg['V_0'])}"
        f"_dz{sanitize(cfg['DZ'])}"
        f"_U{sanitize(cfg['U'])}"
        f"_r{sanitize(cfg['CHANNEL_RADIUS'])}"
        f"_zo{sanitize(cfg['Z_OFFSET'])}"
        f"_z{sanitize(cfg['Z_DEPTH'])}"
        f"_rate{sanitize(cfg['BIT_RATE'])}"
        f"_duration{sanitize(cfg['T_STOP'] - cfg['T_START'])}"
        f"_{'3d' if cfg['RECEIVER_DIMENSION_3D'] else '2d'}"
    )

    dataset_name = os.path.join(save_dir, dataset_name)
    return dataset_name

def save_results(csv_path, model_name, results: dict):

    row = {"model_name": model_name}
    row.update(results)

    if not os.path.exists(csv_path):
        df = pd.DataFrame([row]).to_csv(csv_path, index=False)
        print(f"Created results.csv: {csv_path}")
    else:
        df = pd.read_csv(csv_path)
        for key in row.keys():
            if key not in df.columns:
                df[key] = None
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df.to_csv(csv_path, index=False)
        print(f"Updated results.csv: {csv_path}")
        