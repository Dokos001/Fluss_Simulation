import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import invgauss
from sklearn.metrics import accuracy_score
import tensorflow as tf
from rich.progress import track

class noiseAnalyser:
    #################################################
    # Parameters
    #################################################
    # Time span under scrutiny
    t_start = 0
    t_stop  = 20
    t_step  = 0.01

    # Flow and concentration profile
    v_0         = 0.00707 # 3ml/min in m/s
    dz          = 0.0005
    c_0         = 1
    #Receiver constants
    z_depth     = 0.0075
    #z_offset    = 0.095
    z_offset    = 0.03 

    U = 1.65 #Volt

    #Model Parameters
    TIME_VARIABLE = True
    UNIQUE = True
    F_RX = 0.05

    SAVE_ADDON = "_static_Receiver"
    UNIQUE_ADDON = "_Unique"

    #Channel Radius
    channel_radius = 0.00317  # in meters (3.17mm)
    channel_wall_thickness = 0.00085  # in meters (0.85mm)

    f_rx = 0.025

    varying_receiver = []

    bit_rate = 1

    bit_sequence = []
    #################################################

    def __init__(self, f_rx = None):

                # Variation of the receiver position
                if f_rx == None:
                    self.f_rx = 0.025
                else: 
                    self.f_rx = f_rx

                self.bit_sequence = np.array([1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1])

                print("Generator Ready")

    def sub_ReceiverPosition(self, t):
        # Parameters of varying receiver position
        z_ampl   =  0.005
        f_Rx     =  self.f_rx

        x, E, E_norm = fringing_effects(self)

        E_norm_turned = [E_norm[i] for i in range(len(E)-1, -1, -1)]  # Reverse the order of E_norm

        weighting_Funktion = E_norm_turned+ np.ones(15).tolist()+ E_norm
        
        # Generation of varying receiver position
        z_varyRx = z_ampl * np.sin(2*np.pi*f_Rx * t) + self.z_offset
        
        # Generation of static receiver position for reference
        z_statRx = self.z_offset * np.ones(t.shape)

        z_depth_vector = np.arange(0, self.z_depth, self.z_depth/len(weighting_Funktion))
        #print("z_varyRx: ", z_varyRx)
        #print("z_depth_vector: ", z_depth_vector)

        self.varying_receiver = z_varyRx

        #print(len(weighting_Funktion), len(z_depth_vector))
        return [z_varyRx, z_statRx, z_depth_vector, weighting_Funktion]

    def sub_ReceivedSignal_3DReceiver(self, t, z_Rx, z_depth_vector, dz, v_0, c_0, bit_sequence, weight_function):
        s_depth = []  # Liste für die Signale jeder Tiefe

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

        # Am Ende alle Arrays kombinieren (z.B. aufsummieren)
        s = np.sum(s_depth, axis=0)
        return s
    
    def createNoiseComp(self):
        # Sample times
        t = np.arange(self.t_start, self.t_stop, self.t_step)
            

        #rng = np.random.default_rng()
        noise = invgauss.rvs(mu = 1,scale = 1, size=t.shape)
        noise = noise - np.mean(noise)  # zentrieren
        

        z_varyRx, z_statRx, z_depth_vector, weight_function = self.sub_ReceiverPosition(t)

        s_statRx = self.sub_ReceivedSignal_3DReceiver(t, z_statRx, z_depth_vector, self.dz, self.v_0, self.c_0, self.bit_sequence, weight_function)

        target_snrs_db = np.arange(0, 61, 5)  # SNR values in dB
        snrs = 10 ** (target_snrs_db / 10)  # Convert dB to linear scale
        signal_power = np.mean(s_statRx**2)  # Signal power
        desired_noise_power = signal_power / snrs  # Desired noise power for each SNR
        current_noise_power = np.var(noise)  # Current noise power
        scale_factor = np.sqrt(desired_noise_power / current_noise_power)  # Scale factor to adjust noise power

        noisy_signal_dict = {}
        for i, factor in enumerate(track(scale_factor, description="Adjusting noise for SNR levels")):
            adjusted_noise = noise * factor
            noisy_signal_dict[target_snrs_db[i]] = adjusted_noise + s_statRx  # Add noise to the signal

        return [t, target_snrs_db, noisy_signal_dict]
    

    def plot_noise_comparison(self):
        t, target_snrs_db, noisy_signal_dict = self.createNoiseComp()

        plt.figure(figsize=(12, 6))
        for snr, noisy_signal in noisy_signal_dict.items():
            plt.plot(t, noisy_signal, label=f'SNR = {snr} dB')
        plt.title('Noisy Signals at Different SNR Levels')
        plt.xlabel('Time (s)')
        plt.ylabel('Signal Amplitude')
        plt.legend()
        plt.grid(True)
        plt.show()

    def test_on_noise(self, snrs, noisy_signal_dict):
        string_static = ""
        if not self.TIME_VARIABLE:
            string_static = self.SAVE_ADDON
        string_unique = ""
        if self.UNIQUE:
            string_unique = self.UNIQUE_ADDON

        # Load the model
        f_rx = str(self.F_RX).replace(".", "")
        model = tf.keras.models.load_model('best_param'+string_static+string_unique+"f_rx"+f_rx+'_model.keras')
        accuracys = []
        for snr in snrs:#track(snrs, description="Testing on noisy signals"):
            if snr not in noisy_signal_dict:
                print(f"SNR {snr} dB not found in the noisy signal dictionary.")
                continue
            
            noisy_signal = noisy_signal_dict[snr]
            # Prepare the input data for the model
            X_test = np.array([noisy_signal])
            y_pred              = model.predict(X_test)
            y_pred = y_pred[0]
            print(f"Predicted values for SNR {snr} dB: {y_pred}")
            binariized_y_pred   = np.where(y_pred > 0.5, 1, 0)
            accuracys.append(accuracy_score(self.bit_sequence, binariized_y_pred))
        
        return accuracys

    def plot_test_results(self):

        t, snrs, noisy_signal_dict = self.createNoiseComp()
        accuracys = self.test_on_noise(snrs, noisy_signal_dict)
        print(accuracys)
        plt.figure(figsize=(10, 6))
        plt.plot(snrs, accuracys, marker='o', linestyle='-', color='b')
        plt.title('Model Accuracy vs SNR')
        plt.xlabel('SNR (dB)')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.xticks(snrs)
        plt.show()

def fringing_effects(self):
    U = self.U
    C = 0.205e-12  # Capacitance in Farads (example value)
    A = 1.56e-5
    d = self.channel_radius+2* self.channel_wall_thickness  # Distance between the plates (channel radius)
    x = distance = np.linspace(0, 0.017, 40)  # Distance from the edge of the plates
    e0 = 8.854e-12  # Permittivity of free space in F/m
    er1 = 3 # PVC 
    er2 = 80 # Water

    d1 = 0.85e-3
    d2 = 3.17e-3  

    Q = C*U

    n = 4 #decreasing signal factor

    E = (Q/(A+e0))* ((d1/er1)+(d2/er2)) * (1 / (1+ np.power((2*x)/d , n)))  # Electric field with fringing effects

    E_normalize = [float(i)/max(E) for i in E]  # Normalize the electric field

    return x, E, E_normalize

def plot_fringing_effects(DataGenerator):
    x, E, E_norm = fringing_effects(DataGenerator)

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

if __name__ == "__main__":
    print("Do you want to test on noise? [y/n]")
    choice = input().strip().lower()
    if choice == 'n':
        print("Plotting Noise Comparison")
        gen = noiseAnalyser()
        gen.plot_noise_comparison()
    elif choice == 'y':
        print("Testing on Noise")
        gen = noiseAnalyser()
        gen.plot_test_results()
        
    
    #plot_fringing_effects(gen)