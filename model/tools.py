import datetime
import os
import re
import uuid
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from model.DataGeneration import DataGenerator
from sklearn.model_selection import train_test_split
import h5py
import json

SAVE_ADDON = "_static_Receiver"
UNIQUE_ADDON = "_Unique"
output_dir = "Datasets"


def generate_Timeline(t_start, t_stop, t_step):
    t = np.arange(t_start, t_stop, t_step)
    return t

def display_train_val_loss(history_df, model_path, testbed_name):
    trainloss = np.array(history_df['loss'])
    valloss = np.array(history_df['val_loss'])
    plt.plot(trainloss, 'b', label='Training Loss')
    plt.plot(valloss, color = 'orange', label='Validation Loss')
    plt.title("Verlauf der Loss Funktion")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    save_path = os.path.join(model_path,testbed_name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(os.path.join(save_path, 'train_and_val_loss.png'))
    plt.show()
    
def create_MLDataset(dataset_path, dist_sequenzes_noisy, sequenzes, test_size, random_state):


    X_test_pure, y_test_pure = shuffle(dist_sequenzes_noisy, sequenzes, random_state=random_state)
    print(f"Total dataset size: {len(dist_sequenzes_noisy)} samples")
    print(f"Dataset shape: {np.array(dist_sequenzes_noisy).shape}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        dist_sequenzes_noisy, sequenzes, test_size=test_size, random_state=random_state
    )
    
    X_train, X_val, y_train, y_val                  = train_test_split(X_train, y_train, test_size=test_size, random_state=random_state)
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
    os.makedirs(dataset_path, exist_ok=True)
    for name, data in splits.items():
        pd.DataFrame(data).to_csv(
            os.path.join(dataset_path, f"{name}.csv"),
            header=False,
            index=False
        )

    return X_train, X_test, y_train, y_test, X_val, y_val, X_test_pure, y_test_pure

def load_MLDataset(dataset_path):
    
    X_train = pd.read_csv(os.path.join(dataset_path,"X_train.csv"), header=None).to_numpy()
    y_train = pd.read_csv(os.path.join(dataset_path,"y_train.csv"), header=None).to_numpy()
    X_test  = pd.read_csv(os.path.join(dataset_path,"X_test.csv"), header=None).to_numpy()
    y_test  = pd.read_csv(os.path.join(dataset_path,"y_test.csv"), header=None).to_numpy()
    X_val   = pd.read_csv(os.path.join(dataset_path,"X_val.csv"), header=None).to_numpy()
    y_val   = pd.read_csv(os.path.join(dataset_path,"y_val.csv"), header=None).to_numpy()

    return X_train, X_test, y_train, y_test, X_val, y_val

def load_pureTest_MLDataset(dataset_path):
    X_test  = pd.read_csv(os.path.join(dataset_path,"X_test_pure.csv"), header=None).to_numpy()
    y_test  = pd.read_csv(os.path.join(dataset_path,"y_test_pure.csv"), header=None).to_numpy()

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
        os.makedirs(os.path.dirname(dataset_name), exist_ok=True)
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

def generate_RawDataset_name():
    save_dir = "./Datasets"
    os.makedirs(save_dir, exist_ok=True)

    dataset_name = "complete_dataset"
    u = uuid.uuid4().hex[:6]
    dataset_name = dataset_name + "_"+u
    dataset_path = os.path.join(save_dir, dataset_name)
    return dataset_path, dataset_name

def generate_MLDataset_name(RawDataset_name, cfg):
    save_dir = "./MLDatasets"
    os.makedirs(save_dir, exist_ok=True)

    model_name = os.path.join(save_dir, os.path.basename(RawDataset_name).replace(".h5",""))
    return model_name

def generate_Model_name(cfg):
    save_dir = "./NNModels"
    os.makedirs(save_dir, exist_ok=True)
    u = uuid.uuid4().hex[:6]
    model_name = cfg["MODEL_NAME"] + f"model_{u}"
    model_path = os.path.join(save_dir, model_name)
    return model_path, model_name

def save_results(cfg, csv_path, results: dict):
    file_path = os.path.join(csv_path, "results.csv")

    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
    else:
        df = pd.DataFrame()

    model_name = cfg["MODEL_NAME"]
    testbed_name = cfg["TESTBED_NAME"]

    for col in ["model_name", "testbed_name"]:
        if col not in df.columns:
            df[col] = None

    mask = (df["model_name"] == model_name) & (df["testbed_name"] == testbed_name)

    if mask.any():
        row_index = df.index[mask][0]
    else:
        row_index = len(df)
        df.loc[row_index, ["model_name", "testbed_name"]] = [model_name, testbed_name]

    for key, value in results.items():
        if key in ["model_name", "testbed_name"]:
            continue
        df.loc[row_index, key] = value

    df.to_csv(file_path, index=False)




def logModelParameters(csv_path, model_name, modelcfg):

    file_path = os.path.join(csv_path, "model_logs.csv")
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
    else:
        df = pd.DataFrame()

    new_row = {"model_name": model_name}

    for key, value in modelcfg.items():
        if isinstance(value, (list, dict)):
            value = json.dumps(value, sort_keys=True)
        new_row[key] = value

    if df.empty:
        df = pd.DataFrame([new_row])
        df.to_csv(file_path, index=False)
    else:

        for col in new_row.keys():
            if col not in df.columns:
                df[col] = None

        comparison = (df[list(new_row.keys())] == pd.Series(new_row)).all(axis=1)

        if not comparison.any():
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            df.to_csv(file_path, index=False)

def logTestbedParameters(csv_path, testbed_name, testbedcfg):

    file_path = os.path.join(csv_path, "testbed_logs.csv")
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
    else:
        df = pd.DataFrame()
    
    
    new_row = {"testbed_name": testbed_name}

    for key, value in testbedcfg.items():
        if isinstance(value, (list, dict)):
            value = json.dumps(value, sort_keys=True)
        new_row[key] = value

    if df.empty:
        df = pd.DataFrame([new_row])
        df.to_csv(file_path, index=False)
    else:

        for col in new_row.keys():
            if col not in df.columns:
                df[col] = None

        comparison = (df[list(new_row.keys())] == pd.Series(new_row)).all(axis=1)

        if not comparison.any():
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            df.to_csv(file_path, index=False)


def make_run_id():
    t = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    u = uuid.uuid4().hex[:6]
    return f"run_{t}_{u}"

def change_config(key_to_change, new_value, config_path):
    
    with open(config_path, 'r') as f:
        config = json.load(f)

    config[key_to_change] = new_value

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)