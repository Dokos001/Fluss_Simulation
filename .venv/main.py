import os
import optuna
from model import CBLSTM
from DataGeneration import DataGenerator
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau
from optuna.integration import KerasPruningCallback
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import CSVLogger
import pandas as pd
from sklearn.metrics import accuracy_score
from tools import display_train_val_loss, create_Dataset, load_Dataset

BATCH_SIZE          = 64
SHUFFLE_BUFFER_SIZE = 10
EPOCHS              = 20
NUMBER_OF_ARRAYS    = 10000
NUMBER_OF_BITS      = 13
MODEL_SAVE_PATH     = os.path.dirname(os.path.abspath(__file__))
filter_choices = {
            0: [32,64,128],
            1: [32, 64, 128, 256]#,
            #2: [32, 32, 64, 64, 128, 128, 256, 256]
        }

def main():
    model_instance = CBLSTM()
    [X_train, X_test, y_train, y_test, X_val, y_val] = create_Dataset(number_of_Arrays=NUMBER_OF_ARRAYS, number_of_bits= NUMBER_OF_BITS, test_size=0.20, random_state=42, time_variable= False, unique= True)

    #[X_train, X_test, y_train, y_test, X_val, y_val] = load_Dataset()
    def objective(trial):
        learning_rate       = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        dropout             = trial.suggest_categorical('dropout', [0.2, 0.3])#, 0.5])
        filter_index        = trial.suggest_categorical('filters', [0, 1])#, 2])
        filters             = filter_choices[filter_index]
        num_of_conv_Layers  = trial.suggest_int('num_of_conv_Layers', 2, 5)
        lstm_units          = trial.suggest_categorical('lstm_units', [39, 64, 128])#[13, 26, 39, 64])
        lstm_layers         = trial.suggest_int('lstm_layers', 2, 4)
        batch_size          = trial.suggest_categorical('batch_size', [16, 32, 64])#, 128])


        
        # Modell erstellen
        model = model_instance.create_model(
            learning_rate       = learning_rate,
            filters             = filters,
            num_of_conv_Layers  = num_of_conv_Layers,
            lstm_units          = lstm_units,
            lstm_layers         = lstm_layers,
            dropout_rate        = dropout

        )
        reduce_lr   = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
        early_stop  = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
        pruning     = KerasPruningCallback(trial, 'val_accuracy')
        # Training mit den gewählten Parametern
        model.fit(
            X_train,y_train, 
            validation_data = [X_val, y_val],
            batch_size      = batch_size,
            epochs          = EPOCHS,
            verbose         = 0,
            callbacks       = [reduce_lr,pruning,early_stop]
        )
        
        # Evaluierung
        # evaluation = model.evaluate(X_test,y_test, verbose=1)
        # accuracy = evaluation[1]  # Genauigkeit
        y_pred              = model.predict(X_test)

        binariized_y_pred   = [np.where(array > 0.5, 1, 0) for array in y_pred]

        accuracy            = accuracy_score(y_test, binariized_y_pred)
        return accuracy

    # Optuna-Studie starten
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)  # Anzahl der Trials

    # Beste Ergebnisse anzeigen
    print("Best trial:")
    print(f" Accuracy: {study.best_value}")
    print(f" Params: {study.best_params}")

    reduce_lr   = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
    early_stop  = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
    csvlogger   = CSVLogger('log.csv', append = True, separator = ';')

    # Beste Parameter können verwendet werden, um das Modell zu trainieren
    best_params = study.best_params
    best_model  = model_instance.create_model(
        learning_rate       = best_params['learning_rate'],
        filters             = filter_choices[best_params['filters']],
        num_of_conv_Layers  = best_params['num_of_conv_Layers'],
        lstm_units          = best_params['lstm_units'],
        lstm_layers         = best_params['lstm_layers']
    )
    best_model.fit(
        X_train, y_train,
        validation_data = (X_val, y_val),
        batch_size      = best_params['batch_size'],
        epochs          = EPOCHS,
        callbacks       = [reduce_lr,early_stop, csvlogger]
    )

    # Bestes Modell speichern
    best_model.save('best_param_model_static_unique.keras')
    print(f"Best model saved at: {MODEL_SAVE_PATH}")
    
    # Evaluierung
    # evaluation = model.evaluate(X_test,y_test, verbose=1)
    # accuracy = evaluation[1]  # Genauigkeit
    y_pred              = best_model.predict(X_test)
    binariized_y_pred   = [np.where(array > 0.5, 1, 0) for array in y_pred]
    accuracy            = accuracy_score(y_test, binariized_y_pred)
    print(f" Accuracy: {accuracy}")


if __name__ == "__main__":
    main()