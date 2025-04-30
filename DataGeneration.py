import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import invgauss

class DataGenerator:
    #################################################
    # Parameters
    #################################################
    # Time span under scrutiny
    t_start = 0
    t_stop  = 20
    t_step  = 0.01

    # Flow and concentration profile
    v_0 = 0.00707 # 3ml/min in m/s
    dz  = 0.05
    c_0 = 1

    #Receiver constants
    z_depth = 0.0075
    z_offset = 9.5

    # Variation of the receiver position
    # see subfunction sub_ReceiverPosition()

    # Bit sequence
    # Parameters
    N = 10  # Number of arrays
    M = 13  # Number of positions per array

    f_rx = 0.5

    bit_rate = 1
    #################################################

    def __init__(self, f_rx = None):
                
                self.t_start = 0
                self.t_stop  = 20
                self.t_step  = 0.01

                # Flow and concentration profile
                self.v_0 = 0.001
                self.dz  = 0.05
                self.c_0 = 1

                # Receiver constants
                self.z_depth = 0.05
                self.z_offset = 10

                # Variation of the receiver position
                if f_rx == None:
                    self.f_rx = 0.5
                else: 
                    self.f_rx = f_rx

                # Bit sequence
                # Parameters
                self.N = 10  # Number of arrays
                self.M = 13  # Number of positions per array

                self.bit_sequence = np.array([1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1])

                self.bit_rate = 1
                print("Generator Ready")

    def sub_ReceiverPosition(self, t):
        # Parameters of varying receiver position
        z_ampl   =  2
        f_Rx     =  self.f_rx
        
        # Generation of varying receiver position
        z_varyRx = z_ampl * np.sin(2*np.pi*f_Rx * t) + self.z_offset
        
        # Generation of static receiver position for reference
        z_statRx = self.z_offset * np.ones(t.shape)

        z_depth_vector = np.arange(0, self.z_depth, (self.z_depth/self.v_0)/self.t_step)
        
        return [z_varyRx, z_statRx, z_depth_vector]

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
    
    def sub_ReceivedSignal_3DReceiver(self, t, z_Rx, z_depth_vector, dz, v_0, c_0, bit_sequence):
        s = np.zeros(t.shape)
        for bit in range(len(bit_sequence)):
            if bit_sequence[bit] > 0.5:
                #3D Erfassung des Signals über VolumenReceiver
                for z in z_depth_vector:
                    I_Reg2  = (t-bit >= (z_Rx + z + (dz/2))/v_0)
                    I_Reg23 = (t-bit >= (z_Rx + z - (dz/2))/v_0)
                    I_Reg3  = I_Reg23 & ~(I_Reg2)
                    bit_contribution = np.zeros(t.shape)
                    bit_contribution[I_Reg3] = c_0 * (1 - ( z_Rx[I_Reg3] - (dz/2) ) / ( v_0*(t[I_Reg3]-bit) ))
                    bit_contribution[I_Reg2] = c_0 * (dz/2) / ( v_0 * (t[I_Reg2] - bit) )
                    s += bit_contribution
        
        return s

    def createDataSet(self, number_arrays, number_bits, unique = False,  DimReceiver = False):
        # Sample times
        t = np.arange(self.t_start, self.t_stop, self.t_step)
        if unique:
            sequenzes = create_Unique_Dataset(number_bits=number_bits)
        else:
            sequenzes = [np.random.choice([0, 1], size = (number_bits)) for x in range(number_arrays)]
            
        dist_sequenzes = []
        ideal_sequenzes = []
        for seq in sequenzes:
            

            # Receiver position
            z_varyRx, z_statRx, z_depth_vector = self.sub_ReceiverPosition(t)

            if DimReceiver:
                # Received signal (with/without varying Rx z-position) without noise
                s_varyRx = self.sub_ReceivedSignal_3DReceiver(t, z_varyRx, z_depth_vector, self.dz, self.v_0, self.c_0, seq)
                s_statRx = self.sub_ReceivedSignal_3DReceiver(t, z_statRx, z_depth_vector, self.dz, self.v_0, self.c_0, seq)
            else:
                # Received signal (with/without varying Rx z-position) without noise
                s_varyRx = self.sub_ReceivedSignal(t, z_varyRx, self.dz, self.v_0, self.c_0, seq)
                s_statRx = self.sub_ReceivedSignal(t, z_statRx, self.dz, self.v_0, self.c_0, seq)

            # Received signal superimposed with noise
            #rng = np.random.default_rng()
            mu = 0.145
            noise = invgauss.rvs(mu, size=t.shape)
            #noise = 0.0005 * rng.normal(size=t.shape)
            #noise = 0.0001 * noise
            s_disturbed = s_varyRx + noise

            # Ideal signal (static receiver and without noise !=/)
            s_ideal     = s_statRx + noise

            # Normalization of the signals
            s_disturbed = [float(i)/max(s_disturbed) for i in s_disturbed]
            s_ideal = [float(i)/max(s_ideal) for i in s_ideal]

            dist_sequenzes.append(s_disturbed)
            ideal_sequenzes.append(s_ideal)
        


        return [t, dist_sequenzes, ideal_sequenzes, sequenzes]
    

    def plot_a_sequence(self):
        [t, dist_sequenzes, ideal_sequenzes,sequenzes] = self.createDataSet(self.N, self.M, unique= True, DimReceiver=True)

        s_disturbed = dist_sequenzes[5]
        s_ideal     = ideal_sequenzes[5]


        # Plot both received signals (disturbed and ideal)
        plt.figure()
        plt.plot(t, s_disturbed, 'k')
        plt.plot(t, s_ideal, 'r')
        plt.xlabel('Time in s')
        plt.ylabel('Received signal s')
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

if __name__ == "__main__":
    gen = DataGenerator()
    gen.plot_a_sequence()