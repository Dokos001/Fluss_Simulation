
import json
import model
from model.model import CBLSTM
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau
from optuna.integration import KerasPruningCallback
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import CSVLogger
import pandas as pd
from sklearn.metrics import accuracy_score
from model.tools import change_config, display_train_val_loss, generate_MLDataset_name, generate_Model_name, load_MLDataset, create_MLDataset, generate_RawDataset_name, load_RawData_from_hdf5, load_pureTest_MLDataset,save_complete_dataset, generate_Timeline, save_results, logModelParameters, logTestbedParameters
import random, os
import typer
from model.noiseAnalytics import noiseAnalyser
from view.plotdata import plot_noisy_signals, plot_accuracy, plot_a_sequence
from model.DataGeneration import DataGenerator
from model.trainer import evaluateModel, trainModel


app = typer.Typer()

modelLogDir = "./indexlogs/modellogs"
testbedLogDir = "./indexlogs/testbedlogs"


# ---------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------
def timeline(config_path: str = "config/config.json"):
    cfg = load_config(config_path)
    t = np.arange(cfg["T_START"], cfg["T_STOP"], cfg["T_STEP"])
    return t

def load_config(path: str = "config/config.json") -> dict:
    """Lädt Hyperparameter & Konfiguration aus einer JSON-Datei"""
    with open(path, "r") as f:
        return json.load(f)

def set_random_seed(seed: int):
    """Setzt Reproduzierbarkeits-Seeds"""
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

def prepare_data(cfg, dataset_path, MLDataset_path, Gen, t, random_state=42):
    """Erzeugt oder lädt die Trainings-/Testdaten entsprechend der Konfiguration."""
    [X_train, X_test, y_train, y_test, X_val, y_val] = None, None, None, None, None, None
    print(dataset_path)
    if os.path.exists(MLDataset_path):
        print("Dataset exists. Loading MLDataset..."+dataset_path)
        if cfg["PURE_TEST_SET"]:
            X_test, y_test = load_pureTest_MLDataset(MLDataset_path)
        else:
            [X_train, X_test, y_train, y_test, X_val, y_val] = load_MLDataset(MLDataset_path)
    else:
        print("MLDataset does not exist. Creating MLDataset...")
        if not os.path.exists(dataset_path):
            print("Raw Dataset does not exist. Creating Raw Dataset...")
            dist_sequenzes, dist_sequenzes_noisy, ideal_sequenzes, ideal_sequenzes_noisy, sequenzes = getOrCreateTimeSeriesData(cfg, dataset_path, Gen, t)
        else:
            print("Loading existing Raw Dataset...")
            dist_sequenzes, dist_sequenzes_noisy, ideal_sequenzes, ideal_sequenzes_noisy, sequenzes = load_RawData_from_hdf5(dataset_path)
        [X_train, X_test, y_train, y_test, X_val, y_val, X_test_pure, y_test_pure] = create_MLDataset(MLDataset_path, dist_sequenzes_noisy, sequenzes, test_size=0.2, random_state=random_state)
    return [X_train, X_test, y_train, y_test, X_val, y_val]

def getOrCreateTimeSeriesData(cfg, dataset_path, Gen, t):
    
    if os.path.exists(dataset_path):
        print(f"Loading existing dataset from {dataset_path}...")
        dist_sequenzes, dist_sequenzes_noisy, ideal_sequenzes, ideal_sequenzes_noisy, sequenzes = load_RawData_from_hdf5(dataset_path)
    else:
        print("Creating new dataset...")
        [dist_sequenzes, ideal_sequenzes, dist_sequenzes_noisy, ideal_sequenzes_noisy, sequenzes] = Gen.createDataSet(t = t, number_arrays = cfg["NUMBER_OF_ARRAYS"], 
                                                                                                                      number_bits = cfg["NUMBER_OF_BITS"], 
                                                                                                                      unique = cfg["UNIQUE"], 
                                                                                                                      DimReceiver = cfg["RECEIVER_DIMENSION_3D"], 
                                                                                                                      f_rx = cfg["F_RX"], z_ampl = cfg["Z_AMPL"], z_offset = cfg["Z_OFFSET"], 
                                                                                                                      z_depth = cfg["Z_DEPTH"], dz = cfg["DZ"], 
                                                                                                                      v_0 = cfg["V_0"], 
                                                                                                                      c_0 = cfg["C_0"],
                                                                                                                      U = cfg["U"],
                                                                                                                      channel_radius=cfg["CHANNEL_RADIUS"],
                                                                                                                      channel_wall_thickness=cfg["CHANNEL_WALL_THICKNESS"])
        save_complete_dataset(dist_sequenzes, dist_sequenzes_noisy, ideal_sequenzes, ideal_sequenzes_noisy, sequenzes, dataset_path, cfg)

    return dist_sequenzes, dist_sequenzes_noisy, ideal_sequenzes, ideal_sequenzes_noisy, sequenzes

def getModelandTestbedName(config_path: str = "config/config.json", modelcfg_path: str = "config/config_model.json", testbedcfg_path: str = "config/config_testbed.json"):
    cfg = load_config(config_path)
    modelcfg = load_config(modelcfg_path)
    testbedcfg = load_config(testbedcfg_path)
    if cfg["TRAIN_NEW_MODEL"]:
        model_path, model_name = generate_Model_name(modelcfg)
        change_config("TRAIN_NEW_MODEL", False, config_path)
        change_config("MODEL_PATH", model_path, config_path)
        change_config("MODEL_NAME", model_name, config_path)
    if cfg["GENERATE_NEW_TESTBED"]:
        testbed_path, testbed_name = generate_RawDataset_name()
        change_config("GENERATE_NEW_TESTBED", False, config_path)
        change_config("TESTBED_PATH", testbed_path, config_path)
        change_config("TESTBED_NAME", testbed_name, config_path)
        name = os.path.splitext(os.path.basename(testbed_name))[0]
        MLDataset_path = os.path.join("./MLDatasets", name+"_MLDataset")
        change_config("MLDATASET_NAME", MLDataset_path, config_path)
    cfg = load_config(config_path)  # Reload config to get updated names

    model_name = cfg["MODEL_NAME"]
    model_path = cfg["MODEL_PATH"]
    testbed_name = cfg["TESTBED_NAME"]
    testbed_path = cfg["TESTBED_PATH"]
    MLDataset_path = cfg["MLDATASET_NAME"]


    logModelParameters(modelLogDir, model_name, modelcfg)
    logTestbedParameters(testbedLogDir, testbed_name, testbedcfg)
    
    return model_path, model_name, testbed_path, testbed_name, MLDataset_path
    

#-------------------------- Hauptprogramm ----------------------------------------

@app.command()
def startTraining(config_path: str = "config/config.json"):
    """
    Trains the model based on the configuration.
    
    :param config_path: Path to the configuration file.
    :type config_path: str
    """

    cfg = load_config(config_path)
    testbedcfg = load_config("config/config_testbed.json")
    modelcfg = load_config("config/config_model.json")
    Gen = DataGenerator()
    t = generate_Timeline(testbedcfg["T_START"], testbedcfg["T_STOP"], testbedcfg["T_STEP"])
    #-------------------------- Random Parameter initialization ---------------------

    set_random_seed(cfg["RANDOM_SEED"])

    #--------------------------------------------------------------------------------
    model_path, model_name, testbed_path, testbed_name, MLDataset_path = getModelandTestbedName(config_path, modelcfg_path="config/config_model.json", testbedcfg_path="config/config_testbed.json")
    X_train, X_test, y_train, y_test, X_val, y_val = prepare_data(testbedcfg, testbed_path, MLDataset_path, Gen, t, random_state=cfg["RANDOM_SEED"])
    model_instance = CBLSTM() # create an instance of the model class
    

    #--------------------------------------------------------------------------------
    
    model = model_instance.create_model(cfg=modelcfg, testbedcfg=testbedcfg,
        learning_rate=modelcfg["LEARNING_RATE"],
        filters=modelcfg["FILTERS"],
        num_of_conv_Layers=modelcfg["NUM_OF_CONV_LAYERS"],
        lstm_units=modelcfg["LSTM_UNITS"],
        lstm_layers=modelcfg["LSTM_LAYERS"],
        dropout_rate=modelcfg["DROPOUT"],
    )
    callbacks = [
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
            EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True),
            #CSVLogger("log.csv", append=False, separator=";"),
            tf.keras.callbacks.TensorBoard(log_dir="./logs", histogram_freq=1),
        ]
    model, history = trainModel(model, model_path, X_train, y_train, X_val, y_val, callbacks, modelcfg["BATCH_SIZE"], modelcfg["EPOCHS"])
    
    history_df = pd.DataFrame(history.history)
    hist_csv_path = os.path.join(os.path.dirname(model_path), model_name, testbed_name,  "training_history.csv")
    os.makedirs(os.path.dirname(hist_csv_path), exist_ok=True)
    with open(hist_csv_path, mode='w+') as f:
        history_df.to_csv(f)

    
# ---------------------------------------------------------------------
# Command: evaluate
# ---------------------------------------------------------------------
@app.command()
def evaluate(
    config_path: str = "config/config.json"
):
    """
    Evaluates the model based on the configuration.
    
    :param config_path: Path to the configuration file.
    :type config_path: str
    """
    # --------------------- Config laden -------------------------
    cfg = load_config(config_path)
    testbedcfg = load_config("config/config_testbed.json")
    modelcfg = load_config("config/config_model.json")

    # --------------------- Model- und Testbednamen holen ---------
    model_path, model_name, testbed_path, testbed_name, MLDataset_path = getModelandTestbedName(config_path, modelcfg_path="config/config_model.json", testbedcfg_path="config/config_testbed.json")

    # --------------------- Seeds setzen -------------------------
    set_random_seed(cfg["RANDOM_SEED"])

    # --------------------- Daten vorbereiten --------------------
    Gen = DataGenerator()
    t = generate_Timeline(testbedcfg["T_START"], testbedcfg["T_STOP"], testbedcfg["T_STEP"])
    X_train, X_test, y_train, y_test, X_val, y_val = prepare_data(testbedcfg, dataset_path= testbed_path, MLDataset_path=MLDataset_path, Gen=Gen, t=t, random_state=cfg["RANDOM_SEED"])

    # --------------------- Modell erstellen ---------------------
    model_path = cfg["MODEL_PATH"]
    # --------------------- Gewichte laden -----------------------
    if os.path.exists(model_path):
        model = tf.keras.saving.load_model(model_path)
    else:
        typer.echo("No existing model found at the specified path.")
        return

    # --------------------- Evaluation im Trainer ----------------
    y_pred, results = evaluateModel(cfg, model, model_path, X_test, y_test)

@app.command()
def generateMLDatasetOutOFRawDataset(config_path: str = "config/config.json"):
    """
    Generates an MLDataset from a RawDataset based on the configuration.
    
    :param config_path: Path to the configuration file.
    :type config_path: str
    """
    cfg = load_config(config_path)
    #-------------------------- Random Parameter initialization ---------------------

    set_random_seed(cfg["RANDOM_SEED"])

    #--------------------------------------------------------------------------------
    model_path, testbed_path, MLDataset_path = getModelandTestbedName(config_path, modelcfg_path="config/config_model.json", testbedcfg_path="config/config_testbed.json")
    dist_sequenzes, dist_sequenzes_noisy, ideal_sequenzes, ideal_sequenzes_noisy, sequenzes = load_RawData_from_hdf5(testbed_path)
    [X_train, X_test, y_train, y_test, X_val, y_val, X_test_pure, y_test_pure] = create_MLDataset(MLDataset_path, dist_sequenzes_noisy, sequenzes, test_size=0.2, random_state=cfg["RANDOM_SEED"])

@app.command()
def displayAndSaveLogGraphs(config_path: str = "config/config.json"):
    """
    Displays and saves training and validation loss graphs based on the configuration.
    
    :param config_path: Path to the configuration file.
    :type config_path: str
    """
    cfg = load_config(config_path)
    model_path = cfg["MODEL_PATH"]
    model_name = cfg["MODEL_NAME"]
    testbed_name = cfg["TESTBED_NAME"]
    hist_csv_path = os.path.join(os.path.dirname(model_path), model_name, testbed_name,  "training_history.csv")
    if os.path.exists(hist_csv_path):
        history_df = pd.read_csv(hist_csv_path)
        display_train_val_loss(history_df= history_df, model_path=model_path, testbed_name=testbed_name)
    else:
        typer.echo("No training history found at the specified path.")

@app.command()
def analyse_noise(config_path: str = "config/config.json"):
    """
    Displays noise analysis based on the configuration.
    
    :param config_path: Path to the configuration file.
    :type config_path: str
    """
    t = timeline(config_path)
    cfg = load_config(config_path)
    analyser = noiseAnalyser(cfg)
    gen = DataGenerator(cfg)
    model_path = generate_Model_name(cfg)

    z_varyRx, z_statRx, z_depth_vector, weight_function = gen.sub_ReceiverPosition(t)
    s_statRx = gen.sub_ReceivedSignal_3DReceiver(t, z_statRx, z_depth_vector, gen.dz, gen.v_0, gen.c_0, gen.bit_sequence, weight_function)


    snrs, noisy_dict = analyser.createNoiseComp(s_statRx)
    snrs, accs = analyser.test_on_noise(model_path or "data/models/best_model.keras")
    plot_noisy_signals(t, noisy_dict)
    plot_accuracy(snrs, accs)

# ---------------------------------------------------------------------
# Command: show-config
# ---------------------------------------------------------------------
@app.command()
def show_config(config_path: str = "config/config.json"):
    """
    Displays the configuration from the specified JSON file.
    
    :param config_path: Path to the configuration file.
    :type config_path: str
    """
    cfg = load_config(config_path)
    typer.echo(json.dumps(cfg, indent=4))



# Currently non functional
@app.command()
def create_config(config_path: str = "config/config.json"):
    """
    Creates a standard configuration file.
    
    :param config_path: Path to the configuration file.
    :type config_path: str
    """
    default_cfg = {
        "BATCH_SIZE": 32,
        "SHUFFLE_BUFFER_SIZE": 10,
        "EPOCHS": 20,
        "NUMBER_OF_ARRAYS": 10000,
        "NUMBER_OF_BITS": 13,
        "MODEL_SAVE_PATH": "best_model.h5",
        "LEARNING_RATE": 0.002295686807057715,
        "NUM_OF_CONV_LAYERS": 4,
        "LSTM_UNITS": 64,
        "LSTM_LAYERS": 2,
        "DROPOUT": 0.2,
        "RANDOM_SEED": 42,
        "TIME_VARIABLE": True,
        "UNIQUE": True,
        "TRAIN_NEW_MODEL": False,
        "F_RX": 0.05,
        "LOAD_DATASET": True,
        "PURE_TEST_SET": False
    }
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(default_cfg, f, indent=4)
    typer.echo(f"Standard configuration created at: {config_path}")

# ---------------------------------------------------------------------
@app.command()
def trychangingConfig():
    """
    Tries changing a configuration parameter.
    """
    config_path = "config/config.json"
    change_config("TESTBED_NAME", "new_testbed_name", config_path)
    cfg = load_config(config_path)
    print(cfg["TESTBED_NAME"])

@app.command()
def plot_example(config_path: str = "config/config.json"):
    """
    Plots an example sequence based on the configuration.
    
    :param config_path: Path to the configuration file.
    :type config_path: str
    """
    cfg = load_config(config_path)
    testbedcfg = load_config("config/config_testbed.json")
    datasetpath = cfg["TESTBED_PATH"]
    Gen = DataGenerator()
    t = generate_Timeline(testbedcfg["T_START"], testbedcfg["T_STOP"], testbedcfg["T_STEP"])
    dist_sequenzes, dist_sequenzes_noisy, ideal_sequenzes, ideal_sequenzes_noisy, sequenzes = getOrCreateTimeSeriesData(testbedcfg,datasetpath, Gen, t)
    z_varyRx, z_statRx, z_depth_vector, weight_function = Gen.sub_ReceiverPosition(t, testbedcfg["Z_AMPL"], testbedcfg["F_RX"],  testbedcfg["Z_OFFSET"], testbedcfg["Z_DEPTH"], channel_radius=testbedcfg["CHANNEL_RADIUS"], channel_wall_thickness=testbedcfg["CHANNEL_WALL_THICKNESS"], U=testbedcfg["U"])
    plot_a_sequence(t,dist_sequenzes, ideal_sequenzes, dist_sequenzes_noisy, ideal_sequenzes_noisy, z_varyRx, sequence_index=cfg["SEQUENCE_INDEX"])

if __name__ == "__main__":
    app()
