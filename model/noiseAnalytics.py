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
    bit_rate = 1

    varying_receiver = []


    bit_sequence = []
    #################################################

    def __init__(self, config: dict):
        self.t_start = config.get("T_START", 0)
        self.t_stop = config.get("T_STOP", 20)
        self.t_step = config.get("T_STEP", 0.01)

        self.v_0 = config.get("V_0", 0.00707)
        self.dz = config.get("DZ", 0.0005)
        self.c_0 = config.get("C_0", 1)
        # Receiver constants
        self.z_depth = config.get("Z_DEPTH", 0.0075)
        self.z_offset = config.get("Z_OFFSET", 0.03)

        self.bit_sequence = np.array(config.get("BIT_SEQUENCE", [1,0,1,0,0,1,1,0,0,0,1,1,1]))
        self.bit_rate = config.get("BIT_RATE", 1)
        self.channel_radius = config.get("CHANNEL_RADIUS", 0.00317)
        self.channel_wall_thickness = config.get("CHANNEL_WALL_THICKNESS", 0.00085)
        self.U = config.get("U", 1.65)
        
        self.F_RX = config.get("F_RX", 0.025)
        self.TIME_VARIABLE = config.get("TIME_VARIABLE", True)
        self.UNIQUE = config.get("UNIQUE", True)



    
    def createNoiseComp(self, s_statRx):
        
        #rng = np.random.default_rng()
        noise = invgauss.rvs(mu = 1,scale = 1, size=t.shape)
        noise = noise - np.mean(noise)  # zentrieren
        

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

        return [target_snrs_db, noisy_signal_dict]
    

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