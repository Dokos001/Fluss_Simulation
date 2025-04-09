import pandas as pd
from DataGeneration import DataGenerator
import numpy as np
import matplotlib.pyplot as plt

from tools import load_Dataset

TIME_VARIABLE = True
UNIQUE = True

SAVE_ADDON = "_static_Receiver"
UNIQUE_ADDON = "_Unique"

def main():
    string_static = ""
    if not TIME_VARIABLE:
        string_static = SAVE_ADDON
    string_unique = ""
    if UNIQUE:
        string_unique = UNIQUE_ADDON
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(24, 6))

    [X_train, X_test, y_train, y_test, X_val, y_val] = load_Dataset(time_variable=True, unique= True, f_rx=0.05)


    
    y_pred = pd.read_csv("y_pred"+string_static+string_unique+"f_rx"+"005"+".csv", header=None).to_numpy()
    binariized_y_pred   = [np.where(array > 0.5, 1, 0) for array in y_pred]


    # Beispiel-Daten (ersetze diese durch deine echten Daten)
    X_test = X_test
    y_test = y_test
    y_pred = binariized_y_pred

    index = 0
    first = True
    for j in range(len(y_test)):
        for i in range(len(y_test[0])):
            if y_test[j][i] != y_pred[j][i]:
                if first:
                    print("found")
                    index = j
                    first = False
                

    total_data_points = 1350
    sections = 14

    numbers = [0, index]
    axes = [ax1, ax2]
    # Create 13 intervals starting from 0
    segment_indices = np.linspace(50, total_data_points, sections).astype(int)
    for x in range(2): 
        # Plot der Datenreihe
        axes[x].plot(X_test[numbers[x]])
        # Trennlinien zwischen den Labels
        for i in range(len(segment_indices)):
            axes[x].axvline(x=segment_indices[i], color='black', linestyle='--')

        # Anzeige der Ground Truth Labels (GT) am oberen Rand
        for i in range(len(y_test[numbers[x]])):
            x_center = (segment_indices[i] + segment_indices[i+1]) // 2
            y_value = max(X_test[numbers[x]]+ 0.2)

            # GT-Label in Schwarz

            axes[x].text(x_center, y_value, f"{y_test[numbers[x]][i]}", ha='center', va='bottom', color='black', fontsize=20)

        # Anzeige der Predicted Labels (Pred) direkt unter den GT-Labels
        for i in range(len(y_pred[numbers[x]])):
            x_center = (segment_indices[i] + segment_indices[i+1]) // 2
            y_value = max(X_test[numbers[x]] +0.1)

            # Predicted Labels in Grün oder Rot (je nach Übereinstimmung)
            color = 'green' if y_pred[numbers[x]][i] == y_test[numbers[x]][i] else 'red'
            axes[x].text(x_center, y_value, f"{y_pred[numbers[x]][i]}", ha='center', va='top', color=color, fontsize=20)

        # Dekoration
        axes[x].set_ylim([-2, 5])
        fig.suptitle("Time Series with Ground Truth and Predicted Labels")
        axes[x].set_xlabel("Index")
        axes[x].set_ylabel("Value")
        
        axes[x].grid(True)
    
    fig, (ax3, ax4) = plt.subplots(1,2, figsize=(24, 6))

    [X_train, X_test_static, y_train, y_test_static, X_val, y_val] = load_Dataset(time_variable=False, unique= True)


    index_test_static = 0
    index = 0
    for j in range(len(X_test_static)):
        if np.array_equal(y_test_static[j],[1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0]):
                index_test_static = j
                print("Found")
        if np.array_equal(y_test[j], [1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0]):
                index = j
                print("Found")


    # Plot der Datenreihe
    ax4.plot(X_test[index])
    df = pd.DataFrame(X_test[index])
    df.to_csv("TimeSeries_Data_Variable_Receiver.csv",header=False, index=False)
    ax3.plot(X_test_static[index_test_static])
    df = pd.DataFrame(X_test_static[index_test_static])
    df.to_csv("TimeSeries_Data_Static_Receiver.csv",header=False, index=False)

    t_start = 0
    t_stop  = 20
    t_step  = 0.01
    t = np.arange(t_start, t_stop, t_step)
    gen = DataGenerator(f_rx=0.05)
    z_varyRx, z_statRx = gen.sub_ReceiverPosition(t)
    df = pd.DataFrame(z_varyRx)
    df.to_csv("Z_Sinus_data.csv",header=False, index=False)

    # Dekoration
    ax4.set_title("Variable Receiver Position")
    ax3.set_title("Static Receiver Position")
    ax3.set_ylim([-2, 5])
    ax4.set_ylim([-2, 5])
    fig.suptitle("Time series with static and varaible receiver")
    ax3.set_xlabel("Index")
    ax3.set_ylabel("Value")
    ax3.grid(True)    

    ax4.set_xlabel("Index")
    ax4.set_ylabel("Value")
    ax4.grid(True) 

    plt.show()



if __name__ == "__main__":
    main()