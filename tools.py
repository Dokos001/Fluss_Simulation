import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from DataGeneration import DataGenerator
from sklearn.model_selection import train_test_split

SAVE_ADDON = "_static_Receiver"
UNIQUE_ADDON = "_Unique"
output_dir = "Datasets"

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
    
def create_Dataset(number_of_Arrays, number_of_bits, test_size, random_state, time_variable = True, unique = True, f_rx = 0.5):
    Gen = DataGenerator(f_rx= f_rx)
    [t, dist_sequenzes, ideal_sequenzes, sequenzes] = Gen.createDataSet(number_of_Arrays, number_of_bits, unique=unique)
    
    if not time_variable:
        dist_sequenzes = ideal_sequenzes
    
    X_train, X_test, y_train, y_test = train_test_split(
        dist_sequenzes, sequenzes, test_size=test_size, random_state=random_state
    )
    X_train, X_val, y_train, y_val                  = train_test_split(X_train, y_train, test_size=0.20, random_state=random_state)
    X_train, X_test, y_train, y_test, X_val, y_val  = map(np.array, [X_train, X_test, y_train, y_test, X_val, y_val])
    X_train = (X_train - np.mean(X_train)) / np.std(X_train)
    X_test  = (X_test - np.mean(X_test)) / np.std(X_test)
    X_val   = (X_val - np.mean(X_val)) / np.std(X_val)

    string_static = ""
    if not time_variable:
        string_static = SAVE_ADDON
    string_unique = ""
    if unique:
        string_unique = UNIQUE_ADDON
    
    f_rx = str(f_rx).replace(".", "")

    


    df = pd.DataFrame(X_train)
    df.to_csv(os.path.join(output_dir,"X_train"+string_static+string_unique+"f_rx"+f_rx+".csv"),header=False, index=False)
    df = pd.DataFrame(y_train)
    df.to_csv(os.path.join(output_dir,"y_train"+string_static+string_unique+"f_rx"+f_rx+".csv"),header=False, index=False) 
    df = pd.DataFrame(X_test)
    df.to_csv(os.path.join(output_dir,"X_test"+string_static+string_unique+"f_rx"+f_rx+".csv"),header=False, index=False) 
    df = pd.DataFrame(y_test)
    df.to_csv(os.path.join(output_dir,"y_test"+string_static+string_unique+"f_rx"+f_rx+".csv"),header=False, index=False) 
    df = pd.DataFrame(X_val)
    df.to_csv(os.path.join(output_dir, "X_val"+string_static+string_unique+"f_rx"+f_rx+".csv"),header=False, index=False) 
    df = pd.DataFrame(y_val)
    df.to_csv(os.path.join(output_dir, "y_val"+string_static+string_unique+"f_rx"+f_rx+".csv"),header=False, index=False)

    return X_train, X_test, y_train, y_test, X_val, y_val

def load_Dataset(time_variable = True, unique = True, f_rx = 0.5):
    string_static = ""
    if not time_variable:
        string_static = SAVE_ADDON
    string_unique = ""
    if unique:
        string_unique = UNIQUE_ADDON
    f_rx = str(f_rx).replace(".", "")

    X_train = pd.read_csv(os.path.join(output_dir,"X_train"+string_static+string_unique+"f_rx"+f_rx+".csv"), header=None).to_numpy()
    y_train = pd.read_csv(os.path.join(output_dir,"y_train"+string_static+string_unique+"f_rx"+f_rx+".csv"), header=None).to_numpy()
    X_test  = pd.read_csv(os.path.join(output_dir,"X_test"+string_static+string_unique+"f_rx"+f_rx+".csv"), header=None).to_numpy()
    y_test  = pd.read_csv(os.path.join(output_dir,"y_test"+string_static+string_unique+"f_rx"+f_rx+".csv"), header=None).to_numpy()
    X_val   = pd.read_csv(os.path.join(output_dir,"X_val"+string_static+string_unique+"f_rx"+f_rx+".csv"), header=None).to_numpy()
    y_val   = pd.read_csv(os.path.join(output_dir,"y_val"+string_static+string_unique+"f_rx"+f_rx+".csv"), header=None).to_numpy()

    return X_train, X_test, y_train, y_test, X_val, y_val

def create_pureTest_Dataset(number_of_Arrays, number_of_bits, random_state, time_variable = True, unique = True, load = False):
    string = "_pureTestSet"
    string_static = ""
    if not time_variable:
        string_static = SAVE_ADDON
    string_unique = ""
    if unique:
        string_unique = UNIQUE_ADDON
    
    if load: 
        X_test  = pd.read_csv(os.path.join(output_dir,"X_test"+string+string_static+string_unique+".csv"), header=None).to_numpy()
        y_test  = pd.read_csv(os.path.join(output_dir,"y_test"+string+string_static+string_unique+".csv"), header=None).to_numpy()
    else:
    
        Gen = DataGenerator()
        [t, dist_sequenzes, ideal_sequenzes, sequenzes] = Gen.createDataSet(number_of_Arrays, number_of_bits, unique=unique)
        if time_variable:
            X_test = dist_sequenzes
        else:
            X_test = ideal_sequenzes
            
        X_test = (X_test - np.mean(X_test)) / np.std(X_test)
        y_test = sequenzes
        
        string = "_pureTestSet"
        string_static = ""
        if not time_variable:
            string_static = SAVE_ADDON
        string_unique = ""
        if unique:
            string_unique = UNIQUE_ADDON
        
        df = pd.DataFrame(X_test)
        df.to_csv(os.path.join(output_dir,"X_test"+string_static+string_unique+".csv"),header=False, index=False) 
        df = pd.DataFrame(y_test)
        df.to_csv(os.path.join(output_dir,"y_test"+string_static+string_unique+".csv"),header=False, index=False)


    return X_test, y_test


    