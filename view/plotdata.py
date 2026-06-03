import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
from scipy.signal import argrelmin, find_peaks


def plot_fringing_effects(x, E, E_norm ):

    E_norm_turned = [E_norm[i] for i in range(len(E)-1, -1, -1)]  # Reverse the order of E_norm

    weighting_Funktion = E_norm_turned+ np.ones(30).tolist()+ E_norm  
    x_long = np.linspace(0, 1, len(x)*2+30)  # Extended x-axis for the weighting function

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(x, E, 'r')
    plt.xlabel('Distance from the edge of the plates (m)')
    plt.ylabel('Electric field strength (V/m)')
    plt.title('Fringing Effects on Electric Field Strength')
    plt.grid(True)
    """
    plt.subplot(4, 1, 2)
    plt.plot(x, E_norm, 'r')
    plt.xlabel('Distance from the edge of the plates (m)')
    plt.ylabel('Electric field strength (V/m)')
    plt.title('Fringing Effects on Electric Field Strength')
    plt.grid(True)
    plt.subplot(4, 1, 3)
    plt.plot(x, E_norm_turned, 'r')
    plt.xlabel('Distance from the edge of the plates (m)')
    plt.ylabel('Electric field strength (V/m)')
    plt.title('Fringing Effects on Electric Field Strength')
    plt.grid(True)
    """
    plt.subplot(2, 1, 2)
    plt.plot(x_long, weighting_Funktion, 'r')
    plt.xlabel('DIstance over the capacitor')
    plt.ylabel('Electric field strength in %')
    plt.title('Weighting Function for Bit Contribution')
    plt.grid(True)
    plt.show()

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

def display_learning_rate_history(history_df, model_path, testbed_name):
    learningRate = np.array(history_df['lr'])
    plt.plot(learningRate, 'b', label='Learning Rate')
    plt.title("Verlauf der Learning Rate")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.grid(True)
    save_path = os.path.join(model_path,testbed_name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(os.path.join(save_path, 'learning_rate_history.png'))
    plt.show()

def display_learning_rate_history_and_loss_in_one_plot(history_df, model_path, testbed_name):
    learningRate = np.array(history_df['lr'])
    trainloss = np.array(history_df['loss'])
    valloss = np.array(history_df['val_loss'])
    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(trainloss, 'b', label='Training Loss')
    ax1.plot(valloss, color = 'orange', label='Validation Loss')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()

    color = 'tab:red'
    ax2.set_ylabel('Learning Rate', color=color)
    ax2.plot(learningRate, color=color, label='Learning Rate')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title("Zusammenhang Learning Rate und Loss")
    plt.legend()
    plt.grid(True)
    save_path = os.path.join(model_path,testbed_name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(os.path.join(save_path, 'learning_rate_and_loss_history.png'))
    plt.show()

def plot_a_sequence(t,dist_sequenzes, ideal_sequenzes, dist_sequenzes_noisy, ideal_sequenzes_noisy, z_varyRx, sequence_index, testbed_path):

    s_disturbed = dist_sequenzes[sequence_index]
    s_ideal     = ideal_sequenzes[sequence_index]
    print(f"Length of s_disturbed: {len(s_disturbed)}, Length of s_ideal: {len(s_ideal)}")
    s_disturbed_noisy = dist_sequenzes_noisy[sequence_index]
    s_ideal_noisy     = ideal_sequenzes_noisy[sequence_index]

    if not isinstance(dist_sequenzes_noisy, np.ndarray):
        dist_sequenzes_noisy = np.array(dist_sequenzes_noisy)
    if not isinstance(ideal_sequenzes_noisy, np.ndarray):
        ideal_sequenzes_noisy = np.array(ideal_sequenzes_noisy)

    noise_est_dist = s_disturbed_noisy - s_disturbed
    clean_rms_dist = np.sqrt(np.mean(s_disturbed**2))
    noise_rms_dist = np.sqrt(np.mean(noise_est_dist**2))


    noise_est_ideal = s_ideal_noisy - s_ideal
    clean_rms_ideal = np.sqrt(np.mean(s_ideal**2))
    noise_rms_ideal = np.sqrt(np.mean(noise_est_ideal**2))

    snr_linear_dist = clean_rms_dist / noise_rms_dist
    snr_db_dist = 20 * np.log10(snr_linear_dist)

    snr_linear_ideal = clean_rms_ideal / noise_rms_ideal
    snr_db_ideal = 20 * np.log10(snr_linear_ideal)

    print(f"Estimated SNR for disturbed signal: {snr_db_dist:.2f} dB")
    print(f"Estimated SNR for ideal signal: {snr_db_ideal:.2f} dB")

    rms_signal_dataset_dist = np.sqrt(np.mean(np.array(dist_sequenzes)**2))
    rms_noise_dataset_dist  = np.sqrt(np.mean((dist_sequenzes_noisy - dist_sequenzes)**2))
    snr_db_dataset_dist     = 20 * np.log10(rms_signal_dataset_dist / rms_noise_dataset_dist)
    print(f"Estimated SNR for dataset (disturbed signals): {snr_db_dataset_dist:.2f} dB")

    rms_signal_dataset_ideal = np.sqrt(np.mean(np.array(ideal_sequenzes)**2))
    rms_noise_dataset_ideal  = np.sqrt(np.mean((ideal_sequenzes_noisy - ideal_sequenzes)**2))
    snr_db_dataset_ideal     = 20 * np.log10(rms_signal_dataset_ideal / rms_noise_dataset_ideal)
    print(f"Estimated SNR for dataset (ideal signals): {snr_db_dataset_ideal:.2f} dB")



    # Plot both received signals (disturbed and ideal)
    f, (a0, a1, a2) = plt.subplots(3, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [1,2,2]})
    
    plt.rcParams.update({'axes.titlesize': 20})
    a0.plot(t, z_varyRx, 'k')
    a0.set_xlabel('Time in s')
    a0.set_ylabel('Receiver position z [m]')
    a0.set_title('Varying Receiver Position over one Sequence')
    a0.grid(True)
    a1.plot(t, s_disturbed, 'k')
    a1.plot(t, s_ideal, 'r')
    a1.set_xlabel('Time in s')
    a1.set_ylabel('Received signal s')
    a1.set_title('Received signal with static and oscillating receiver position')
    a1.legend(['Disturbed signal', 'Ideal signal'])
    a1.grid(True)
    a2.plot(t, s_disturbed_noisy, 'k')
    a2.plot(t, s_ideal_noisy, 'r')
    a2.set_xlabel('Time in s')
    a2.set_ylabel('Received signal s')
    a2.set_title('Received signal with static and oscillating receiver position with noise')
    a2.legend(['Disturbed signal', 'Ideal signal'])
    a2.grid(True)
    f.tight_layout()
    f.savefig(os.path.join(testbed_path, 'example_sequence.png'), dpi=300)


    
    dt = t[1] - t[0]  # Assuming uniform sampling
    fs = 1.0 / dt # Sampling frequency
    N = len(s_disturbed)
    yf_dist = np.fft.fft(s_disturbed)
    yf_ideal = np.fft.fft(s_ideal)
    xf = np.fft.fftfreq(N, 1/fs)

    # Plot
    plt.figure(figsize=(10, 5))
    plt.subplot(2,1,1)
    plt.plot(xf, yf_dist, color='k', alpha=0.8)
    plt.title('FFT of Disturbed Signal')
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    plt.xlim(0, fs/20)
    plt.ylim(0, None)
    plt.tight_layout()

    plt.subplot(2,1,2)
    plt.plot(xf, yf_ideal, color='r', alpha=0.8)
    plt.title('FFT of Ideal Signal')
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    plt.xlim(0, fs/20)
    plt.ylim(0, None)
    plt.tight_layout()
    plt.savefig(os.path.join(testbed_path, 'example_sequence_fft.png'), dpi=300)
    plt.show()
    print(f"Plots saved to {testbed_path}.")

def plot_weightingFunction(weighting_function, testbed_path):

    # Plot both received signals (disturbed and ideal)
    plt.figure()
    plt.plot(weighting_function, 'k')
    plt.ylabel('Bit Contribution Weighting Factor')
    plt.title('Weighting Function for 3D Receiver')
    plt.grid(True)
    plt.show()
    plt.savefig(os.path.join(testbed_path, 'weighting_function.png'), dpi=300)

def plot_real_signal(real_signal):
    t = real_signal['timestamp']
    t = t / 1e3  # Convert milliseconds to seconds
    t = t - t[0]  # Normalize time to start from zero
    capacitance_values = real_signal['value']


    peaks = argrelmin(capacitance_values.to_numpy(), order=250)


    plt.figure(figsize=(12, 6))
    plt.plot(t, capacitance_values, color='blue')
    for peak in peaks[0]:
        if not t[peak]  < 50 and not t[peak] > 100: 
            print(t[peak])
            plt.axvline(x=t[peak], color='red', linestyle='--', alpha=0.7)  
    plt.title('Real Signal Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Capacitance')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.savefig('figures/real_signal_plot.png', dpi=300)

def plot_Bartunik_Data(real_signal):
    t_original = real_signal['time']
    resonanceShift = real_signal['value']
    print(f"Original length of data: {len(t_original)}")
    t = t_original[:640]
    resonanceShift = resonanceShift[:640]

    baseline_shift_window = 120
    resonanceShift_baseline_corrected =  np.convolve(resonanceShift, np.ones(baseline_shift_window)/baseline_shift_window, mode='valid')
    resonanceShift_baseline_corrected_padded = np.pad(resonanceShift_baseline_corrected, (baseline_shift_window-1, 0), mode='edge')
    resonanceShift_baseline_corrected = resonanceShift - resonanceShift_baseline_corrected_padded

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(t, resonanceShift, color='blue')
    plt.title('Real Signal Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Resonance Frequency Shift (kHz)')
    plt.grid(True)
    plt.tight_layout()
    plt.subplot(2, 1, 2)
    plt.plot(t, resonanceShift_baseline_corrected, color='orange')
    plt.title('Baseline Corrected Resonance Frequency Shift')
    plt.xlabel('Time (s)')
    plt.ylabel('Resonance Frequency Shift (kHz)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.savefig('figures/BartunikData_RFS.png', dpi=300)



def plot_new_signal_intervals(real_signal):
    resonanceShift = real_signal['value']

    amount_of_data_points = [640, 490, 490, 480, 500, 480, 490, 490, 490, 490, 490, 490, 490, 480, 490, 490, 490, 490, 480, 490, 480, 490, 490, 490, 480, 490, 490, 490, 490, 480, 490, 490, 490, 490, 490]
    
    sequences = []
    start_index = 0
    for i in range(len(amount_of_data_points)):
        end_index = start_index + amount_of_data_points[i]
        sequences.append(resonanceShift[start_index:end_index])
        start_index = end_index

    for i, seq in enumerate(sequences):
        plt.figure(figsize=(12, 6))
        plt.plot(seq)
        plt.title(f'Real Signal Interval {i}')
        plt.xlabel('Time (s)')
        plt.ylabel('Resonance Frequency Shift (kHz)')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        plt.savefig(f'figures/real_signal_interval_{i}.png', dpi=300)

def plot_a_Bartunik_sequence(t, dist_sequenzes, dist_sequenzes_noisy, sequence_index, testbed_path):

    s_disturbed = dist_sequenzes[sequence_index]
    s_disturbed_noisy = dist_sequenzes_noisy[sequence_index]

    if not isinstance(dist_sequenzes_noisy, np.ndarray):
        dist_sequenzes_noisy = np.array(dist_sequenzes_noisy)

    noise_est_dist = s_disturbed_noisy - s_disturbed
    clean_rms_dist = np.sqrt(np.mean(s_disturbed**2))
    noise_rms_dist = np.sqrt(np.mean(noise_est_dist**2))


    snr_linear_dist = clean_rms_dist / noise_rms_dist
    snr_db_dist = 20 * np.log10(snr_linear_dist)

    print(f"Estimated SNR for disturbed signal: {snr_db_dist:.2f} dB")

    rms_signal_dataset_dist = np.sqrt(np.mean(np.array(dist_sequenzes)**2))
    rms_noise_dataset_dist  = np.sqrt(np.mean((dist_sequenzes_noisy - dist_sequenzes)**2))
    snr_db_dataset_dist     = 20 * np.log10(rms_signal_dataset_dist / rms_noise_dataset_dist)
    print(f"Estimated SNR for dataset (disturbed signals): {snr_db_dataset_dist:.2f} dB")

    # Plot both received signals (disturbed and ideal)
    f, (a1, a2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2,2]})
    
    
    a1.plot(t, s_disturbed, 'k')
    a1.set_xlabel('Time in s')
    a1.set_ylabel('Received signal s')
    a1.set_title('Received signal')
    a1.grid(True)
    a2.plot(t, s_disturbed_noisy, 'k')
    a2.set_xlabel('Time in s')
    a2.set_ylabel('Received signal s')
    a2.set_title('Received signal with noise')
    a2.grid(True)
    f.tight_layout()
    f.savefig(os.path.join(testbed_path, 'example_sequence.png'), dpi=300)
    


def plot_noise_comparison(noiseAnalytics):
    t, target_snrs_db, noisy_signal_dict = noiseAnalytics.createNoiseComp()

    plt.figure(figsize=(12, 6))
    for snr, noisy_signal in noisy_signal_dict.items():
        plt.plot(t, noisy_signal, label=f'SNR = {snr} dB')
    plt.title('Noisy Signals at Different SNR Levels')
    plt.xlabel('Time (s)')
    plt.ylabel('Signal Amplitude')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_noisy_signals(t, noisy_signal_dict):
    plt.figure(figsize=(12, 6))
    for snr, signal in noisy_signal_dict.items():
        plt.plot(t, signal, label=f"SNR = {snr} dB")
    plt.title("Noisy Signals at Different SNR Levels")
    plt.xlabel("Time (s)")
    plt.ylabel("Signal Amplitude")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_accuracy(snrs, accuracies):
    plt.figure(figsize=(10, 6))
    plt.plot(snrs, accuracies, marker="o")
    plt.title("Model Accuracy vs. SNR")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.show()

def plot_two_distinct_sequences(t,dist_sequenzes_1, ideal_sequenzes_1, dist_sequenzes_noisy_1, ideal_sequenzes_noisy_1, dist_sequenzes_noisy_2, ideal_sequenzes_noisy_2, z_varyRx, sequence_index):

    s_disturbed_1 = dist_sequenzes_1[sequence_index]
    s_ideal_1     = ideal_sequenzes_1[sequence_index]
    print(f"Length of s_disturbed_1: {len(s_disturbed_1)}, Length of s_ideal_1 : {len(s_ideal_1)}")
    s_disturbed_noisy_1 = dist_sequenzes_noisy_1[sequence_index]
    s_ideal_noisy_1     = ideal_sequenzes_noisy_1[sequence_index]

    if not isinstance(dist_sequenzes_noisy_1, np.ndarray):
        dist_sequenzes_noisy_1 = np.array(dist_sequenzes_noisy_1)
    if not isinstance(ideal_sequenzes_noisy_1, np.ndarray):
        ideal_sequenzes_noisy_1 = np.array(ideal_sequenzes_noisy_1)
    s_disturbed_noisy_2 = dist_sequenzes_noisy_2[sequence_index]
    s_ideal_noisy_2     = ideal_sequenzes_noisy_2[sequence_index]

    if not isinstance(dist_sequenzes_noisy_2, np.ndarray):
        dist_sequenzes_noisy_2 = np.array(dist_sequenzes_noisy_2)
    if not isinstance(ideal_sequenzes_noisy_2, np.ndarray):
        ideal_sequenzes_noisy_2 = np.array(ideal_sequenzes_noisy_2)

    # Plot both received signals (disturbed and ideal)
    f, (a0, a1, a2, a3) = plt.subplots(4, 1, figsize=(15, 13), gridspec_kw={'height_ratios': [1,2,2,2]})
    
    plt.rcParams.update({'axes.titlesize': 20})
    a0.plot(t, z_varyRx, 'k')
    a0.set_xlabel('Time in s')
    a0.set_ylabel('Receiver position z [m]')
    a0.set_title('Varying Receiver Position over one Sequence')
    a0.grid(True)
    a1.plot(t, s_disturbed_1, 'k')
    a1.plot(t, s_ideal_1, 'r')
    a1.set_xlabel('Time in s')
    a1.set_ylabel('Received signal s')
    a1.set_title('Received signal with static and oscillating receiver position')
    a1.legend(['Disturbed signal', 'Ideal signal'])
    a1.grid(True)
    a2.plot(t, s_disturbed_noisy_1, 'k')
    a2.plot(t, s_ideal_noisy_1 , 'r')
    a2.set_xlabel('Time in s')
    a2.set_ylabel('Received signal s')
    a2.set_title('Received signal with static and oscillating receiver position with white gaussian noise')
    a2.legend(['Disturbed signal', 'Ideal signal'])
    a2.grid(True)
    a3.plot(t, s_disturbed_noisy_2, 'k')
    a3.plot(t, s_ideal_noisy_2 , 'r')
    a3.set_xlabel('Time in s')
    a3.set_ylabel('Received signal s')
    a3.set_title('Received signal with static and oscillating receiver position with inverse gaussian noise')
    a3.legend(['Disturbed signal', 'Ideal signal'])
    a3.grid(True)
    f.tight_layout()
    f.savefig('ComparisonSequence.png', dpi=300)