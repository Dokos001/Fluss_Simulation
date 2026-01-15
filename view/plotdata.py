import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq


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

def plot_a_sequence(t,dist_sequenzes, ideal_sequenzes, dist_sequenzes_noisy, ideal_sequenzes_noisy, z_varyRx, sequence_index, testbed_path):

    s_disturbed = dist_sequenzes[sequence_index]
    s_ideal     = ideal_sequenzes[sequence_index]
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