
from model import CBLSTM
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau
from optuna.integration import KerasPruningCallback
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import CSVLogger
import pandas as pd
from sklearn.metrics import accuracy_score
from tools import display_train_val_loss, load_Dataset


BATCH_SIZE          = 32
SHUFFLE_BUFFER_SIZE = 10
EPOCHS              = 20
NUMBER_OF_ARRAYS    = 10000
NUMBER_OF_BITS      = 13
MODEL_SAVE_PATH     = "best_model.h5"
LEARNING_RATE       = 0.002295686807057715
NUM_OF_CONV_LAYERS  = 4
LSTM_UNITS          = 64
LSTM_LAYERS         = 2
DROPOUT             = 0.2

TIME_VARIABLE = True
UNIQUE = True
training = False
F_RX = 0.05

SAVE_ADDON = "_static_Receiver"
UNIQUE_ADDON = "_Unique"

def main():
    model_instance = CBLSTM()
    #[X_train, X_test, y_train, y_test, X_val, y_val] = create_Dataset(number_of_Arrays=NUMBER_OF_ARRAYS, number_of_bits= NUMBER_OF_BITS, test_size=0.20, random_state=42, time_variable=TIME_VARIABLE,unique=UNIQUE, f_rx = F_RX)
    [X_train, X_test, y_train, y_test, X_val, y_val] = load_Dataset(time_variable=TIME_VARIABLE, unique= UNIQUE, f_rx= F_RX)
    #[X_test, y_test] = create_pureTest_Dataset(number_of_Arrays=NUMBER_OF_ARRAYS, number_of_bits= NUMBER_OF_BITS, random_state=42, time_variable= True, unique= False, load= False)
    string_static = ""
    if not TIME_VARIABLE:
        string_static = SAVE_ADDON
    string_unique = ""
    if UNIQUE:
        string_unique = UNIQUE_ADDON
    f_rx = str(F_RX).replace(".", "")
    if training:
    # Modell erstellen
        model = model_instance.create_model(
            learning_rate       = LEARNING_RATE,
            filters             = [32,64,128],
            num_of_conv_Layers  = NUM_OF_CONV_LAYERS,
            lstm_units          = LSTM_UNITS,
            lstm_layers         = LSTM_LAYERS,
            dropout_rate        = DROPOUT

        )
        reduce_lr   = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
        early_stop  = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
        csvlogger   = CSVLogger('log.csv', append = True, separator = ';')
        # Training mit den gewählten Parametern
        model.fit(
            X_train,y_train,
            validation_data = [X_val, y_val],
            batch_size      = BATCH_SIZE,
            epochs          = EPOCHS,
            verbose         = 1,
            callbacks       = [reduce_lr, early_stop, csvlogger]
        )
        
        
        
        model.save('best_param'+string_static+string_unique+"f_rx"+f_rx+'_model.keras')
        print(f"Best model saved")
    else:

        f_rx = str(f_rx).replace(".", "")
        model = tf.keras.models.load_model('best_param'+string_static+string_unique+"f_rx"+f_rx+'_model.keras')
    
    # Evaluierung
    # evaluation = model.evaluate(X_test,y_test, verbose=1)
    # accuracy = evaluation[1]  # Genauigkeit
    y_pred              = model.predict(X_test)
    df = pd.DataFrame(y_pred)
    df.to_csv("y_pred"+string_static+string_unique+"f_rx"+f_rx+".csv",header=False, index=False)
    binariized_y_pred   = [np.where(array > 0.5, 1, 0) for array in y_pred]
    inccorect_bits = 0
    correct_bits = 0
    whole_bits = 0
    average_incorrect_bits = []
    for i in range(len(binariized_y_pred)):
        for bit_ind in range(len(binariized_y_pred[i])):
            temp_inc_bits_sequenze = 0
            if binariized_y_pred[i][bit_ind] != y_test[i][bit_ind]:
                inccorect_bits += 1
                temp_inc_bits_sequenze += 1
            else: 
                correct_bits += 1
            whole_bits += 1
            average_incorrect_bits.append(temp_inc_bits_sequenze)
            
        #print("Pred:", binariized_y_pred[i])
        #print("True:", y_test[i])
        
    accuracy            = accuracy_score(y_test, binariized_y_pred)
    print(f" Accuracy: {accuracy}")
    print(f" BER: {inccorect_bits/whole_bits}")
    print(f" Average_BER: {np.mean(average_incorrect_bits)/13}")
    if(training):
        display_train_val_loss()



if __name__ == "__main__":
    main()