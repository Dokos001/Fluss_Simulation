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

def split_dataset(X, y, test_size, random_state):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_train, X_val, y_train, y_val   = train_test_split(X_train, y_train, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test, X_val, y_val


def extract_dataset_from_Bartunik_Data(real_signal, transform_to_volume_signal = False):
    t = real_signal['time']
    t_corrected = t[150:]
    resonanceShift = real_signal['value']
    resonanceShift_corrected = resonanceShift[150:]
    L0 = 250e-6 #H
    C = 68e-12 #F
    f0 = 1/(2*np.pi*np.sqrt(L0*C))# Hz

    # Custom Sequence Lengths based on actual data
    amount_of_data_points = [
                            163,163,164,
                            163,163,164,
                            163,163,164,
                            160,160,160,
                            166,166,167,
                            160,160,160,
                            163,163,164,
                            163,163,164,
                            163,163,164,
                            163,163,164,
                            163,163,164,
                            163,163,164,
                            163,163,164,
                            160,160,160,
                            163,163,164,
                            163,163,164,
                            163,163,164,
                            163,163,164,
                            160,160,160,
                            163,163,164,
                            160,160,160,
                            163,163,164,
                            163,163,164,
                            163,163,164,
                            160,160,160,
                            163,163,164,
                            163,163,164,
                            163,163,164,
                            163,163,164,
                            160,160,160,
                            163,163,164,
                            163,163,164,
                            163,163,164,
                            163,163,164,
                            163,163,164
                            ]#[490, 490, 490, 480, 500, 480, 490, 490, 490, 490, 490, 490, 490, 480, 490, 490, 490, 490, 480, 490, 480, 490, 490, 490, 480, 490,# 490, 490, 490, 480, 490, 490, 490, 490, 490]
    
    # Calculated through 5 khz shift per 30 ul, spions
    #k_hz_per_ul = 166.7 # hz/ul
    
    #baseline_shift_window = 120
    #resonanceShift_baseline_corrected =  np.convolve(resonanceShift_corrected, np.ones(baseline_shift_window)/baseline_shift_window, mode='valid')
    #resonanceShift_baseline_corrected_padded = np.pad(resonanceShift_baseline_corrected, (baseline_shift_window-1, 0), mode='edge')
    #resonanceShift_baseline_corrected = resonanceShift_corrected - resonanceShift_baseline_corrected_padded
    #resonanceShift_baseline_corrected_volume_signal = (resonanceShift_baseline_corrected *1000) / -k_hz_per_ul
    #signal = resonanceShift_baseline_corrected_volume_signal
    signal = resonanceShift_corrected # convert to hz
    signal_save = signal

    if transform_to_volume_signal:
        signal = signal
        Leff = pow(1/(2*np.pi*  signal),2)/C
        signal = Leff
        print("f min:", np.min(signal_save))
        print("f max:", np.max(signal_save))
        print("f unique count:", len(np.unique(signal_save)))
        print(f"Leff range: {np.min(Leff):.2e} H to {np.max(Leff):.2e} H")
        print(f"Shape of Leff signal: {Leff.shape}")

    
    sequences = []
    start_index = 0
    for i in range(len(amount_of_data_points)):
        end_index = start_index + amount_of_data_points[i]
        sequences.append(signal[start_index:end_index])
        start_index = end_index
    
    # SNR estimation

    Noise_segment = sequences[1][:100]  # Assuming the first 100 points are noise
    Signal_segment = sequences[1][110:200]  # Assuming the next 300 points contain the signal
    Noise_power = np.mean(Noise_segment**2)
    Signal_power = np.mean(Signal_segment**2)
    SNR = 10 * np.log10(Signal_power / Noise_power)
    print(f"Estimated SNR: {SNR:.2f} dB")

    target_length = 170 #500

    for i, seq in enumerate(sequences):
        seq = np.array(seq, dtype=float) 
        if len(seq) < target_length:  # Only pad sequences that are shorter than the target length
            sequences[i] = np.pad(seq, (0, target_length - len(seq)), mode='edge')
    
    sequences = sequences[:-1]
    print(f"Length of sequences: {len(sequences)}, Length of each sequence: {len(sequences[0])}")
    labels = []
    with open('RealeMessungen/Complete Transmission Sequence.txt', 'r') as f:
        labels = [int(x) for x in list(f.read())]
    sequence_labels = []

    AmountOfLabelsPerSequence = 10

    if (len(labels) % AmountOfLabelsPerSequence) != 0:
        padding_amount = AmountOfLabelsPerSequence - (len(labels) % AmountOfLabelsPerSequence)
        labels = np.pad(labels, (0, padding_amount), mode='constant')

    for i in range(0, len(labels), AmountOfLabelsPerSequence):
        label_seq = labels[i:i+AmountOfLabelsPerSequence]
        sequence_labels.append(label_seq)
        print(f"{int(i/10)}: {label_seq}")
    print(len(sequences), len(sequence_labels))

    for i in range(len(sequences)):
        plt.figure(figsize=(12, 6))
        plt.plot(sequences[i], color='blue')
        plt.title('Real World Signal Over Time')
        plt.xlabel('index')
        plt.ylabel('Received Signal s')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'figures/real_signal_sequence_{i}.png', dpi=300)
        #print(f"Labels for sequence {i}: {sequence_labels[i]}")
        plt.close()

    splits = {
    "X_test_pure": sequences,
    "y_test_pure": sequence_labels
    }
    savepath = f"Datasets/RealDataBartunik2023_{AmountOfLabelsPerSequence}Bit"
    if transform_to_volume_signal:
        savepath = savepath + "_Leff_Signal"
    os.makedirs(savepath, exist_ok=True)
    for name, data in splits.items():
        pd.DataFrame(data).to_csv(
            os.path.join(savepath, f"{name}.csv"),
            header=False,
            index=False
        )


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

def load_RawData_from_Bartunik_hdf5(filename):
    dist_sequenzes = []
    dist_sequenzes_noisy = []
    sequenzes = []

    with h5py.File(filename, 'r') as hf:
        # Disturbed Signals
        grp = hf['disturbed_signals']
        dist_sequenzes = [sig for sig in grp['data']]
        dist_sequenzes_noisy = [sig for sig in grp['data_noisy']]


        # Original Sequences
        grp = hf['original_sequences']
        sequenzes = [sig for sig in grp['data']]

    return dist_sequenzes, dist_sequenzes_noisy, sequenzes


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

def save_modified_dataset(dist_sequenzes, dist_sequenzes_noisy, sequenzes, dataset_name, cfg):
        os.makedirs(os.path.dirname(dataset_name), exist_ok=True)
        with h5py.File(dataset_name, 'w') as hf:

             # --- include MetaData ---
            hf.attrs["dataset_name"] = dataset_name
            for k, v in cfg.items():
                hf.attrs[k] = v
            grp = hf.create_group('disturbed_signals')
            grp.create_dataset('data', data=dist_sequenzes)
            grp.create_dataset('data_noisy', data=dist_sequenzes_noisy)

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

def extractMean_and_plot_impulse_response(signal, t, save_path):
    impulse_response = signal - np.mean(signal[0:300])
    t = t[532:572]
    t = t-t[0]
    impulse_response = impulse_response[532:572]
    h_norm = impulse_response / np.sum(np.abs(impulse_response))
    h_norm = abs(h_norm)

    test_sequence = [0,1,0,1,0,1,0,1,1,1]
    
    ones = np.zeros(5).tolist() + np.ones(6).tolist()+ np.zeros(5).tolist()
    ones = np.array(ones)
    
    print(f"Ones pattern: {ones}")
    zeros = np.zeros(16)
    bit_sequence = []
    for bit in test_sequence:
        if bit == 1:
            bit_sequence.extend(ones)
        else:
            bit_sequence.extend(zeros)
    bit_sequence = np.array(bit_sequence)
    test_sequence_convolved = np.convolve(bit_sequence, h_norm, mode='same')

    plt.figure(figsize=(12, 6))
    plt.plot(t, h_norm, color='red')
    plt.title('Estimated Impulse Response')
    plt.xlabel('Time (s)')
    plt.ylabel('Impulse Response (Hz/s)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(test_sequence_convolved, color='blue')
    plt.title('Sequence Convolved with Impulse Response')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path[:-3]+"_convolved.png", dpi=300)
    plt.close()
    print(f"Impulse response shape: {h_norm.shape}, Time shape: {t.shape}")
    print(f"Convolved sequence shape: {test_sequence_convolved.shape}")
    return h_norm, t

