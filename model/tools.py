import os
import re
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
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
    
def create_Dataset(number_of_Arrays, number_of_bits, test_size, random_state, time_variable = True, unique = True, f_rx = 0.5, dimReceiver = False):
    Gen = DataGenerator(f_rx= f_rx)
    [t, dist_sequenzes, ideal_sequenzes, sequenzes] = Gen.createDataSet(number_of_Arrays, number_of_bits, unique=unique, DimReceiver=dimReceiver)
    
    if not time_variable:
        dist_sequenzes = ideal_sequenzes
    
    X_train, X_test, y_train, y_test = train_test_split(
        dist_sequenzes, sequenzes, test_size=test_size, random_state=random_state
    )
    X_train, X_val, y_train, y_val                  = train_test_split(X_train, y_train, test_size=0.20, random_state=random_state)
    X_train, X_test, y_train, y_test, X_val, y_val  = map(np.array, [X_train, X_test, y_train, y_test, X_val, y_val])
    X_train = (X_train - np.mean(X_train)) / np.std(X_train)
    X_test  = (X_test - np.mean(X_test)) / np.std(X_test)
    X_val   = (X_val - np.mean(X_val)) / np.std(X_val)

    string_static = ""
    if not time_variable:
        string_static = SAVE_ADDON
    string_unique = ""
    if unique:
        string_unique = UNIQUE_ADDON
    
    f_rx = str(f_rx).replace(".", "")

    


    df = pd.DataFrame(X_train)
    df.to_csv(os.path.join(output_dir,"X_train"+string_static+string_unique+"f_rx"+f_rx+".csv"),header=False, index=False)
    df = pd.DataFrame(y_train)
    df.to_csv(os.path.join(output_dir,"y_train"+string_static+string_unique+"f_rx"+f_rx+".csv"),header=False, index=False) 
    df = pd.DataFrame(X_test)
    df.to_csv(os.path.join(output_dir,"X_test"+string_static+string_unique+"f_rx"+f_rx+".csv"),header=False, index=False) 
    df = pd.DataFrame(y_test)
    df.to_csv(os.path.join(output_dir,"y_test"+string_static+string_unique+"f_rx"+f_rx+".csv"),header=False, index=False) 
    df = pd.DataFrame(X_val)
    df.to_csv(os.path.join(output_dir, "X_val"+string_static+string_unique+"f_rx"+f_rx+".csv"),header=False, index=False) 
    df = pd.DataFrame(y_val)
    df.to_csv(os.path.join(output_dir, "y_val"+string_static+string_unique+"f_rx"+f_rx+".csv"),header=False, index=False)

    return X_train, X_test, y_train, y_test, X_val, y_val

def load_Dataset(time_variable = True, unique = True, f_rx = 0.5):
    string_static = ""
    if not time_variable:
        string_static = SAVE_ADDON
    string_unique = ""
    if unique:
        string_unique = UNIQUE_ADDON
    f_rx = str(f_rx).replace(".", "")

    X_train = pd.read_csv(os.path.join(output_dir,"X_train"+string_static+string_unique+"f_rx"+f_rx+".csv"), header=None).to_numpy()
    y_train = pd.read_csv(os.path.join(output_dir,"y_train"+string_static+string_unique+"f_rx"+f_rx+".csv"), header=None).to_numpy()
    X_test  = pd.read_csv(os.path.join(output_dir,"X_test"+string_static+string_unique+"f_rx"+f_rx+".csv"), header=None).to_numpy()
    y_test  = pd.read_csv(os.path.join(output_dir,"y_test"+string_static+string_unique+"f_rx"+f_rx+".csv"), header=None).to_numpy()
    X_val   = pd.read_csv(os.path.join(output_dir,"X_val"+string_static+string_unique+"f_rx"+f_rx+".csv"), header=None).to_numpy()
    y_val   = pd.read_csv(os.path.join(output_dir,"y_val"+string_static+string_unique+"f_rx"+f_rx+".csv"), header=None).to_numpy()

    return X_train, X_test, y_train, y_test, X_val, y_val

def create_pureTest_Dataset(number_of_Arrays, number_of_bits, random_state, time_variable = True, unique = True, load = False):
    string = "_pureTestSet"
    string_static = ""
    if not time_variable:
        string_static = SAVE_ADDON
    string_unique = ""
    if unique:
        string_unique = UNIQUE_ADDON
    
    if load: 
        X_test  = pd.read_csv(os.path.join(output_dir,"X_test"+string+string_static+string_unique+".csv"), header=None).to_numpy()
        y_test  = pd.read_csv(os.path.join(output_dir,"y_test"+string+string_static+string_unique+".csv"), header=None).to_numpy()
    else:
    
        Gen = DataGenerator()
        [t, dist_sequenzes, ideal_sequenzes, sequenzes] = Gen.createDataSet(number_of_Arrays, number_of_bits, unique=unique)
        if time_variable:
            X_test = dist_sequenzes
        else:
            X_test = ideal_sequenzes
            
        X_test = (X_test - np.mean(X_test)) / np.std(X_test)
        y_test = sequenzes
        
        string = "_pureTestSet"
        string_static = ""
        if not time_variable:
            string_static = SAVE_ADDON
        string_unique = ""
        if unique:
            string_unique = UNIQUE_ADDON
        
        df = pd.DataFrame(X_test)
        df.to_csv(os.path.join(output_dir,"X_test"+string_static+string_unique+".csv"),header=False, index=False) 
        df = pd.DataFrame(y_test)
        df.to_csv(os.path.join(output_dir,"y_test"+string_static+string_unique+".csv"),header=False, index=False)


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

        print("Complete dataset saved to 'complete_dataset.h5'")

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
        f"_z{sanitize(cfg['Z_DEPTH'])}"
        f"_{'3d' if cfg['RECEIVER_DIMENSION_3D'] else '2d'}"
    )

    dataset_name += "_complete_dataset"
    dataset_name = os.path.join(save_dir, dataset_name)
    dataset_name += ".h5"
    return dataset_name
    