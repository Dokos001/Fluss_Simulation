import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import invgauss
from tqdm import tqdm

class DataGenerator:
    #################################################
    # Parameters
    #################################################
    # Time span under scrutiny
    t_start = 0
    t_stop  = 40
    t_step  = 0.01

    # Flow and concentration profile
    v_0         = 0.00707 # 3ml/min in m/s
    dz          = 0.0005
    c_0         = 1
    #Receiver constants
    z_depth     = 0.0075
    z_offset    = 0.095

    U = 1.65 #Volt


    # Variation of the receiver position
    # see subfunction sub_ReceiverPosition()

    # Bit sequence
    # Parameters
    N = 10  # Number of arrays
    M = 13  # Number of positions per array

    #Channel Radius
    channel_radius = 0.00317  # in meters (3.17mm)
    channel_wall_thickness = 0.00085  # in meters (0.85mm)

    f_rx = 0.025

    varying_receiver = []

    bit_rate = 1
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
        print("z_varyRx: ", z_varyRx)
        print("z_depth_vector: ", z_depth_vector)

        self.varying_receiver = z_varyRx

        print(len(weighting_Funktion), len(z_depth_vector))
        return [z_varyRx, z_statRx, z_depth_vector, weighting_Funktion]

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
    """
    def sub_ReceivedSignal_3DReceiver(self, t, z_Rx, z_depth_vector, dz, v_0, c_0, bit_sequence):
        s = np.zeros(t.shape)
        #print("z_Rx: ", z_Rx, "z_depth_vector: ", z_depth_vector, "dz: ", dz, "v_0: ", v_0, "c_0: ", c_0)
        
        for bit in range(len(bit_sequence)):
            if bit_sequence[bit] > 0.5:
                #3D Erfassung des Signals über VolumenReceiver
                for z in z_depth_vector:
                    #print("t-bit: ",(t-bit), "receiver position: ", (z_Rx+z+(dz/2))/v_0,)
                    I_Reg2  = (t-bit >= (z_Rx + z + (dz/2))/v_0)
                    I_Reg23 = (t-bit >= (z_Rx + z - (dz/2))/v_0)
                    I_Reg3  = I_Reg23 & ~(I_Reg2)
                    bit_contribution = np.zeros(t.shape)
                    bit_contribution[I_Reg3] = c_0 * (1 - ( z_Rx[I_Reg3] - (dz/2) ) / ( v_0*(t[I_Reg3]-bit) ))
                    bit_contribution[I_Reg2] = c_0 * (dz/2) / ( v_0 * (t[I_Reg2] - bit) )
                    s += bit_contribution
        
        return s
    """
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
    
    def velocity_profile(self, r):
        v_max = 2 * self.v_0
        return v_max * (1 - (r / self.channel_radius) ** 2)
    
    """
    def sub_ReceivedSignal_3DReceiver(self, t, z_Rx, z_depth_vector, dz, v_0, c_0, bit_sequence):
        s_depth = []
        r_vector = np.linspace(0, self.channel_radius, num=20)  # 20 radial steps

        for z in z_depth_vector:
            s_z = np.zeros(t.shape)
            for r in r_vector:
                v_local = self.velocity_profile(r)
                for bit in range(len(bit_sequence)):
                    if bit_sequence[bit] > 0.5:
                        I_Reg2  = (t-bit >= (z_Rx + z + (dz/2))/v_local)
                        I_Reg23 = (t-bit >= (z_Rx + z - (dz/2))/v_local)
                        I_Reg3  = I_Reg23 & ~(I_Reg2)
                        bit_contribution = np.zeros(t.shape)
                        bit_contribution[I_Reg3] = c_0 * (1 - ( z_Rx[I_Reg3] + z - (dz/2) ) / ( v_local*(t[I_Reg3]-bit) ))
                        bit_contribution[I_Reg2] = c_0 * (dz/2) / ( v_local * (t[I_Reg2] - bit) )
                        s_z += bit_contribution
            s_depth.append(s_z)
        s = np.sum(s_depth, axis=0)
        return s
    """

    def createDataSet(self, number_arrays, number_bits, unique = False,  DimReceiver = False):
        # Sample times
        t = np.arange(self.t_start, self.t_stop, self.t_step)
        if unique:
            sequenzes = create_Unique_Dataset(number_bits=number_bits)
        else:
            sequenzes = [np.random.choice([0, 1], size = (number_bits)) for x in range(number_arrays)]
            
        dist_sequenzes = []
        ideal_sequenzes = []
        dist_sequenzes_noisy = []
        ideal_sequenzes_noisy = []
        #rng = np.random.default_rng()
        noise = invgauss.rvs(1,scale = 2, size=t.shape)
        #noise = 0.0005 * rng.normal(size=t.shape)
        noise = 0.2* noise


        z_varyRx, z_statRx, z_depth_vector, weight_function = self.sub_ReceiverPosition(t)


        for seq in tqdm(sequenzes):
            

            if DimReceiver:
                # Received signal (with/without varying Rx z-position) without noise
                s_varyRx = self.sub_ReceivedSignal_3DReceiver(t, z_varyRx, z_depth_vector, self.dz, self.v_0, self.c_0, seq, weight_function)
                s_statRx = self.sub_ReceivedSignal_3DReceiver(t, z_statRx, z_depth_vector, self.dz, self.v_0, self.c_0, seq, weight_function)
            else:
                # Received signal (with/without varying Rx z-position) without noise
                s_varyRx = self.sub_ReceivedSignal(t, z_varyRx, self.dz, self.v_0, self.c_0, seq)
                s_statRx = self.sub_ReceivedSignal(t, z_statRx, self.dz, self.v_0, self.c_0, seq)

            # Oscillating signal 
            s_disturbed = s_varyRx

            # Ideal signal 
            s_ideal     = s_statRx

            # Add noise to the signals
            s_disturbed_noisy = s_disturbed + noise
            s_ideal_noisy     = s_ideal + noise


            # Normalization of the signals
            #s_disturbed = [float(i)/max(s_disturbed) for i in s_disturbed]
            #s_ideal = [float(i)/max(s_ideal) for i in s_ideal]
            #s_disturbed_noisy = [float(i)/max(s_disturbed_noisy) for i in s_disturbed_noisy]
            #s_ideal_noisy = [float(i)/max(s_ideal_noisy) for i in s_ideal_noisy]

            dist_sequenzes.append(s_disturbed)
            ideal_sequenzes.append(s_ideal)
            dist_sequenzes_noisy.append(s_disturbed_noisy)
            ideal_sequenzes_noisy.append(s_ideal_noisy)
        


        return [t, dist_sequenzes, ideal_sequenzes, dist_sequenzes_noisy, ideal_sequenzes_noisy,  sequenzes]
    

    def plot_a_sequence(self):
        [t, dist_sequenzes, ideal_sequenzes, dist_sequenzes_noisy, ideal_sequenzes_noisy, sequenzes] = self.createDataSet(self.N, self.M, unique= True, DimReceiver=True)

        s_disturbed = dist_sequenzes[41]
        s_ideal     = ideal_sequenzes[41]
        s_disturbed_noisy = dist_sequenzes_noisy[41]
        s_ideal_noisy     = ideal_sequenzes_noisy[41]


        # Plot both received signals (disturbed and ideal)
        plt.figure()
        plt.subplot(3,1,1)
        plt.plot(t, self.varying_receiver, 'k')
        plt.xlabel('Time in s')
        plt.ylabel('Receiver position z in meters')
        plt.title('Varying Receiver Position over one Sequence')
        plt.grid(True)
        plt.subplot(3,1,2)
        plt.plot(t, s_disturbed, 'k')
        plt.plot(t, s_ideal, 'r')
        plt.xlabel('Time in s')
        plt.ylabel('Received signal s')
        plt.title('Received signal with static and oscillating receiver position')
        plt.legend(['Disturbed signal', 'Ideal signal'])
        plt.grid(True)
        plt.subplot(3,1,3)
        plt.plot(t, s_disturbed_noisy, 'k')
        plt.plot(t, s_ideal_noisy, 'r')
        plt.xlabel('Time in s')
        plt.ylabel('Received signal s')
        plt.title('Received signal with static and oscillating receiver position with noise')
        plt.legend(['Disturbed signal', 'Ideal signal'])
        plt.grid(True)
        plt.show()

'''
    def sub_PointSourceSignal(self, t, x, v_0, D, c_0):
        """ Modelliert eine Punktquelle in der Mitte, advektiert und diffundiert über die Zeit und Position """
        # Berechnung der Konzentration C(x, t) basierend auf der Advektions-Diffusions-Gleichung
        C = np.zeros((len(x), len(t)))
        for i,x_point in enumerate(x):
            C[i] = (c_0 / (np.sqrt(4 * np.pi * D * t + 1e-12))) * np.exp(-((x_point- v_0 * t) ** 2) / (4 * D * t + 1e-12))
        return C

    def plot_point_source(self):
        """ Plotte die Punktquelle, die sich bewegt und diffundiert über Zeit und Position """
        # Zeit- und Positionsvektoren
        t = np.arange(self.t_start, self.t_stop, self.t_step)
        x = np.linspace(0, 0.05, 200)  # Kanal mit Länge 5 cm (0.05 m)

        # Diffusionskoeffizient
        D = 4.0e-12  # m²/s

        # Berechnung des Signals
        signal = self.sub_PointSourceSignal(t, x, self.v_0, D, self.c_0)
        print(signal.shape)
        print(signal[0:5, 0:5])
        # Plotten des Signals als 2D-Diagramm
        plt.figure(figsize=(10, 6))
        plt.contourf(t, x, signal, levels=50, cmap='viridis')
        plt.colorbar(label='Concentration (normalized)')
        plt.xlabel('Position along the channel (m)')
        plt.ylabel('Time (s)')
        plt.title('Advection-Diffusion of a Point Source')
        plt.grid(True)
        plt.show()
'''


def create_Unique_Dataset(number_bits):
    number_bits = 13  
    number_arrays = 2**number_bits 

    # Create all possible combinations of 0 and 1 for the given number of bits
    sequenzes = np.array([list(map(int, format(i, f'0{number_bits}b'))) for i in range(number_arrays)])

    print(sequenzes.shape)
    print(sequenzes[:5])
    assert len(sequenzes) == 2**number_bits, "Something went wrong, but i dont know why!"
    
    return sequenzes

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
    gen = DataGenerator()
    gen.plot_a_sequence()
    #plot_fringing_effects(gen)