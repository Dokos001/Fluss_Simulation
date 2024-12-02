import numpy as np
import matplotlib.pyplot as plt


class DataGenerator:
    #################################################
    # Parameters
    #################################################
    # Time span under scrutiny
    t_start = 0
    t_stop  = 20
    t_step  = 0.01

    # Flow and concentration profile
    v_0 = 10
    dz  = 0.05
    c_0 = 1

    # Variation of the receiver position
    # see subfunction sub_ReceiverPosition()

    # Bit sequence
    # Parameters
    N = 10  # Number of arrays
    M = 13  # Number of positions per array

    bit_sequence = np.array([1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1])

    bit_rate = 1
    #################################################

    def __init__(self):
                
                self.t_start = 0
                self.t_stop  = 20
                self.t_step  = 0.01

                # Flow and concentration profile
                self.v_0 = 10
                self.dz  = 0.05
                self.c_0 = 1

                # Variation of the receiver position
                # see subfunction sub_ReceiverPosition()

                # Bit sequence
                # Parameters
                self.N = 10  # Number of arrays
                self.M = 13  # Number of positions per array

                self.bit_sequence = np.array([1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1])

                self.bit_rate = 1
                print("Generator Ready")

    def sub_ReceiverPosition(self, t):
        # Parameters of varying receiver position
        z_offset = 10
        z_ampl   =  2
        f_Rx     =  0.5
        
        # Generation of varying receiver position
        z_varyRx = z_ampl * np.sin(2*np.pi*f_Rx * t) + z_offset
        
        # Generation of static receiver position for reference
        z_statRx = z_offset * np.ones(t.shape)
        
        return [z_varyRx, z_statRx]

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

    def createDataSet(self, number_arrays, number_bits):
        # Sample times
        t = np.arange(self.t_start, self.t_stop, self.t_step)
        sequenzes = [np.random.choice([0, 1], size = (number_bits)) for x in range(number_arrays)]
        dist_sequenzes = []
        ideal_sequenzes = []
        for seq in sequenzes:
        
            # Receiver position
            z_varyRx, z_statRx = self.sub_ReceiverPosition(t)

            # Received signal (with/without varying Rx z-position) without noise
            s_varyRx = self.sub_ReceivedSignal(t, z_varyRx, self.dz, self.v_0, self.c_0, seq)
            s_statRx = self.sub_ReceivedSignal(t, z_statRx, self.dz, self.v_0, self.c_0, seq)

            # Received signal superimposed with noise
            rng = np.random.default_rng()
            noise = 0.0005 * rng.normal(size=t.shape)
            s_disturbed = s_varyRx + noise

            # Ideal signal (static receiver and without noise)
            s_ideal     = s_statRx

            dist_sequenzes.append(s_disturbed)
            ideal_sequenzes.append(s_ideal)
        


        return [t, dist_sequenzes, ideal_sequenzes, sequenzes]
    
    def plot_a_sequence(self):
        [t, dist_sequenzes, ideal_sequenzes,sequenzes] = self.createDataSet()

        s_disturbed = dist_sequenzes[5]
        s_ideal     = ideal_sequenzes[5]

        print(sequenzes[5])

        # Plot both received signals (disturbed and ideal)
        plt.figure()
        plt.plot(t, s_disturbed, 'k')
        plt.plot(t, s_ideal, 'r')
        plt.xlabel('Time in s')
        plt.ylabel('Received signal s')
        plt.show()

