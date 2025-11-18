import itertools
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import invgauss
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
import h5py

class DataGenerator:
    #################################################
    # Parameters
    #################################################

    
    

    

    
    #################################################

    def __init__(self):
                

                print("Generator Ready")

    def sub_ReceiverPosition(self, t, z_ampl, f_rx, z_offset, z_depth, channel_radius=0.00317, channel_wall_thickness=0.00085, U=1.65):
        # Parameters of varying receiver position
        

        x, E, E_norm = self.fringing_effects(channel_radius=channel_radius, channel_wall_thickness=channel_wall_thickness, U=U) #Calculate fringing effects

        E_norm_turned = [E_norm[i] for i in range(len(E)-1, -1, -1)]  # Reverse the order of E_norm

        weighting_Funktion = E_norm_turned+ np.ones(15).tolist()+ E_norm #Create weighting function for 3D receiver
        weighting_Funktion = np.array(weighting_Funktion)
        weighting_Funktion /= np.sum(weighting_Funktion) # Normalize weighting function
        
        # Generation of varying receiver position
        z_varyRx = z_ampl * np.sin(2*np.pi*f_rx * t) + z_offset
        
        # Generation of static receiver position for reference
        z_statRx = z_offset * np.ones(t.shape)

        z_depth_vector = np.arange(0, z_depth, z_depth/len(weighting_Funktion))
        #print("z_varyRx: ", z_varyRx)
        #print("z_depth_vector: ", z_depth_vector)

        self.varying_receiver = z_varyRx

        #print(len(weighting_Funktion), len(z_depth_vector))
        return [z_varyRx, z_statRx, z_depth_vector, weighting_Funktion]

    #2D Volume Receiver for Static and Varying Position
    def sub_ReceivedSignal(self, t, z_Rx, dz, v_0, c_0, bit_sequence):
        s = np.zeros(t.shape)
        for bit in range(len(bit_sequence)):
            if bit_sequence[bit] > 0.5:
                I_Reg2  = (t-bit >= (z_Rx + (dz/2))/v_0)
                I_Reg23 = (t-bit >= (z_Rx - (dz/2))/v_0)
                I_Reg3  = I_Reg23 & ~(I_Reg2)
                bit_contribution = np.zeros(t.shape)
                bit_contribution[I_Reg3] = c_0 * (1 - ( z_Rx[I_Reg3] - (dz/2) ) / ( v_0*(t[I_Reg3]-bit) ))
                bit_contribution[I_Reg2] = c_0 * (dz/2) / ( v_0 * (t[I_Reg2] - bit) )
                s += bit_contribution
        
        return s
    
    #3D Volume Receiver for Static and Varying Position
    def sub_ReceivedSignal_3DReceiver(self, t, z_Rx, z_depth_vector, dz, v_0, c_0, bit_sequence, weight_function):
        s_depth = [] 
        Dim_receiver_correction = z_depth_vector[1] - z_depth_vector[0] # to account for the discrete steps in z-depth

        for i,z in enumerate(z_depth_vector):
            s_z = np.zeros(t.shape)
            for bit in range(len(bit_sequence)):
                if bit_sequence[bit] > 0.5:
                    I_Reg2  = (t-bit >= (z_Rx + z + (dz/2))/v_0)
                    I_Reg23 = (t-bit >= (z_Rx + z - (dz/2))/v_0)
                    I_Reg3  = I_Reg23 & ~(I_Reg2)
                    bit_contribution = np.zeros(t.shape)
                    bit_contribution[I_Reg3] = c_0 * (1 - ( z_Rx[I_Reg3]+z - (dz/2) ) / ( v_0*(t[I_Reg3]-bit) ))
                    bit_contribution[I_Reg2] = c_0 * (z+dz/2) / ( v_0 * (t[I_Reg2] - bit) )
                    s_z += weight_function[i] * bit_contribution
            s_depth.append(s_z)

        s = np.sum(s_depth, axis=0) * Dim_receiver_correction  # Multiply by depth correction to account for discrete summation
        return s

    def createDataSet(self, t, number_arrays, number_bits, unique = False,  DimReceiver = False, f_rx = None, z_ampl = None, z_offset = None, z_depth = None, dz = None, v_0 = None, c_0 = None,  U = None, channel_radius=0.00317, channel_wall_thickness=0.00085):
        # Sample times
        
        if unique:
            sequenzes = self.create_Unique_Dataset(number_bits, number_arrays)
        else:
            sequenzes = [np.random.choice([0, 1], size = (number_bits)) for x in range(number_arrays)]
            
        dist_sequenzes = []
        ideal_sequenzes = []
        dist_sequenzes_noisy = []
        ideal_sequenzes_noisy = []

        z_varyRx, z_statRx, z_depth_vector, weight_function = self.sub_ReceiverPosition(t, z_ampl, f_rx, z_offset, z_depth, channel_radius=channel_radius, channel_wall_thickness=channel_wall_thickness, U=U)

        for seq in tqdm(sequenzes):

            if DimReceiver:
                # Received signal (with/without varying Rx z-position) without noise
                s_varyRx = self.sub_ReceivedSignal_3DReceiver(t, z_varyRx, z_depth_vector, dz, v_0, c_0, seq, weight_function)
                s_statRx = self.sub_ReceivedSignal_3DReceiver(t, z_statRx, z_depth_vector, dz, v_0, c_0, seq, weight_function)
            else:
                # Received signal (with/without varying Rx z-position) without noise
                s_varyRx = self.sub_ReceivedSignal(t, z_varyRx, dz, v_0, c_0, seq)
                s_statRx = self.sub_ReceivedSignal(t, z_statRx, dz, v_0, c_0, seq)

            # Oscillating signal 
            s_disturbed = s_varyRx

            # Ideal signal 
            s_ideal     = s_statRx

            dist_sequenzes.append(s_disturbed)
            ideal_sequenzes.append(s_ideal)
            
        dist_sequenzes = np.array(dist_sequenzes)
        ideal_sequenzes = np.array(ideal_sequenzes)
        dataset_dist_rms = np.sqrt(np.mean(dist_sequenzes**2))
        dataset_ideal_rms = np.sqrt(np.mean(ideal_sequenzes**2))

        raw_noise = invgauss.rvs(mu=1.0, scale=1.0, size=t.shape)   # positive samples
        raw_noise -= np.mean(raw_noise)  # zero-mean

        desired_snr_db = 20  # Desired SNR in dB
        desired_snr_linear = 10 ** (desired_snr_db / 20)
        noise_scaled_dist = raw_noise * (dataset_dist_rms / (desired_snr_linear * np.sqrt(np.mean(raw_noise**2))))
        noise_scaled_ideal = raw_noise * (dataset_ideal_rms / (desired_snr_linear * np.sqrt(np.mean(raw_noise**2))))
        dist_sequenzes_noisy = dist_sequenzes + noise_scaled_dist
        ideal_sequenzes_noisy = ideal_sequenzes + noise_scaled_ideal

        return [dist_sequenzes, ideal_sequenzes, dist_sequenzes_noisy, ideal_sequenzes_noisy,  sequenzes]

    def create_Unique_Dataset(self, n_bits: int, N: int = None):
        """
        Creates unique bit combinations of length n_bits.

        :param n_bits: Number of bits per combination
        :param N: Number of desired unique combinations (ignored if all=True)
        :param all: If True, all possible combinations are generated (warning!)
        :return: List of bitstrings of length n_bits
        """
        
        max_combos = 2 ** n_bits
        if N == max_combos:
            all = True
        else:
            all = False
        sequenzes = []

        if all:
            print(f" Beware: you are about to create {max_combos:,} combinations!")
            if max_combos > 1_000_000:
                confirm = input("This may take a long time and require a lot of memory. "
                                "Do you really want to continue? (y/n): ").strip().lower()
                if confirm != "y":
                    print("Interrupted.")
                else:
                    sequenzes = [list(map(int, bits)) for bits in tqdm(itertools.product('01', repeat=n_bits), total=max_combos)]
        else:
            if N is None:
                raise ValueError("Please specify N or set all=True.")
            if N > max_combos:
                raise ValueError(f"There are only {max_combos} possible combinations, "
                                f"but N={N} was requested.")

            nums = np.random.choice(max_combos, size=N, replace=False)
            bits = ((nums[:, None] & (1 << np.arange(n_bits)[::-1])) > 0).astype(int)
            sequenzes = bits.tolist()

        sequenzes = np.array(sequenzes)
        
        return sequenzes

    def fringing_effects(self, channel_radius=0.00317, channel_wall_thickness=0.00085, U=1.65):

        C = 0.205e-12  # Capacitance in Farads 
        A = 1.56e-5
        d = channel_radius + 2 * channel_wall_thickness  # Distance between the plates (channel radius)
        x = distance = np.linspace(0, 0.017, 40)  # Distance from the edge of the plates
        e0 = 8.854e-12  # Permittivity of free space in F/m
        er1 = 3  # PVC
        er2 = 80  # Water

        d1 = 0.85e-3
        d2 = 3.17e-3  

        Q = C * U

        n = 4  # decreasing signal factor

        E = (Q / (A + e0)) * ((d1 / er1) + (d2 / er2)) * (1 / (1 + np.power((2 * x) / d, n)))  # Electric field with fringing effects

        E_normalize = [float(i) / max(E) for i in E]  # Normalize the electric field

        return x, E, E_normalize