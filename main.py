
import json
import model
from model.model import CBLSTM
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau
from model.tuner import run_optuna_study
from model.tuner import run_optuna_study
from optuna.integration import KerasPruningCallback
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import CSVLogger
import pandas as pd
from model.tools import change_config, extract_dataset_from_Bartunik_Data, extractMean_and_plot_impulse_response, generate_Model_name, load_MLDataset, create_MLDataset, generate_RawDataset_name, load_RawData_from_Bartunik_hdf5, load_RawData_from_hdf5, load_pureTest_MLDataset,save_complete_dataset, generate_Timeline, logModelParameters, logTestbedParameters, save_modified_dataset, split_dataset
import random, os
import typer
from model.noiseAnalytics import noiseAnalyser
from view.plotdata import display_learning_rate_history_and_loss_in_one_plot, plot_Bartunik_Data, plot_a_Bartunik_sequence, plot_new_signal_intervals, plot_noisy_signals, plot_accuracy, plot_a_sequence, plot_two_distinct_sequences, plot_weightingFunction, display_learning_rate_history, display_train_val_loss, plot_real_signal
from model.DataGeneration import DataGenerator
from model.trainer import applyTransferLearning, evaluateModel, trainModel
from sklearn.preprocessing import StandardScaler


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

def prepare_data(cfg, dataset_path, MLDataset_path, Gen, t, random_state, use_pure_test_set, test_size):
    """Erzeugt oder lädt die Trainings-/Testdaten entsprechend der Konfiguration."""
    [X_train, X_test, y_train, y_test, X_val, y_val] = None, None, None, None, None, None
    print(dataset_path)
    if os.path.exists(MLDataset_path):
        print("Dataset exists. Loading MLDataset..."+dataset_path)
        if use_pure_test_set:
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
        [X_train, X_test, y_train, y_test, X_val, y_val, X_test_pure, y_test_pure] = create_MLDataset(MLDataset_path, dist_sequenzes_noisy, sequenzes, test_size=test_size, random_state=random_state)
    return [X_train, X_test, y_train, y_test, X_val, y_val]

def getOrCreateTimeSeriesData(cfg, dataset_path, Gen, t, impulse_response = None):
    
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
                                                                                                                      channel_wall_thickness=cfg["CHANNEL_WALL_THICKNESS"], 
                                                                                                                      snr = cfg["SNR_DB"],
                                                                                                                      bitrate = cfg["BIT_RATE"], 
                                                                                                                      cut_forward_tail = cfg["CUT_FORWARD_TAIL"],
                                                                                                                      get_resonance_shift_signal = cfg["GET_RESONANCE_SHIFT_SIGNAL"],
                                                                                                                      impulse_response = impulse_response
                                                                                                                      )
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
        logModelParameters(modelLogDir, model_name, modelcfg)
    if cfg["GENERATE_NEW_TESTBED"]:
        testbed_path, testbed_name = generate_RawDataset_name()
        change_config("GENERATE_NEW_TESTBED", False, config_path)
        change_config("TESTBED_PATH", testbed_path, config_path)
        change_config("TESTBED_NAME", testbed_name, config_path)
        name = os.path.splitext(os.path.basename(testbed_name))[0]
        MLDataset_path = os.path.join(testbed_path,"MLDatasets", name+"_MLDataset_"+str(cfg["TEST_SIZE_RATIO"]))
        change_config("MLDATASET_NAME", MLDataset_path, config_path)
        logTestbedParameters(testbedLogDir, testbed_name, testbedcfg)
    cfg = load_config(config_path)  # Reload config to get updated names

    model_name = cfg["MODEL_NAME"]
    model_path = cfg["MODEL_PATH"]
    testbed_name = cfg["TESTBED_NAME"]
    testbed_path = cfg["TESTBED_PATH"]
    MLDataset_path = cfg["MLDATASET_NAME"]
    
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
    if testbedcfg["CUT_FORWARD_TAIL"]:
        change_config("T_STOP", 5, config_path="config/config_testbed.json")
        testbedcfg = load_config("config/config_testbed.json")

    #-------------------------- Random Parameter initialization ---------------------

    set_random_seed(cfg["RANDOM_SEED"])

    #--------------------------------------------------------------------------------
    model_path, model_name, testbed_path, testbed_name, MLDataset_path = getModelandTestbedName(config_path, modelcfg_path="config/config_model.json", testbedcfg_path="config/config_testbed.json")
    directTestbed_path = os.path.join(testbed_path, testbed_name + ".h5")
    X_train, X_test, y_train, y_test, X_val, y_val = prepare_data(testbedcfg, directTestbed_path, MLDataset_path, Gen, t, random_state=cfg["RANDOM_SEED"], use_pure_test_set=cfg["USE_PURE_TEST_SET"], test_size=cfg["TEST_SIZE_RATIO"])
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
    directTestbed_path = os.path.join(testbed_path, testbed_name + ".h5")
    # --------------------- Seeds setzen -------------------------
    set_random_seed(cfg["RANDOM_SEED"])

    # --------------------- Daten vorbereiten --------------------
    Gen = DataGenerator()
    t = generate_Timeline(testbedcfg["T_START"], testbedcfg["T_STOP"], testbedcfg["T_STEP"])
    X_train, X_test, y_train, y_test, X_val, y_val = prepare_data(testbedcfg, dataset_path= directTestbed_path, MLDataset_path=MLDataset_path, Gen=Gen, t=t, random_state=cfg["RANDOM_SEED"], use_pure_test_set=cfg["USE_PURE_TEST_SET"], test_size=cfg["TEST_SIZE_RATIO"])

    # --------------------- Modell erstellen ---------------------
    model_path = cfg["MODEL_PATH"]
    # --------------------- Gewichte laden -----------------------
    if os.path.exists(model_path):
        model = tf.keras.saving.load_model(model_path)
    else:
        typer.echo("No existing model found at the specified path.")
        return

    # --------------------- Evaluation im Trainer ----------------
    sequence_index = cfg["SEQUENCE_INDEX"]
    y_pred, results = evaluateModel(cfg, model, model_path, X_test, y_test)
    print(f"Prediction {sequence_index}: {np.round(y_pred[sequence_index])}, True Labels: {y_test[sequence_index]}")

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
    model_path, model_name, testbed_path, testbed_name, MLDataset_path = getModelandTestbedName(config_path, modelcfg_path="config/config_model.json", testbedcfg_path="config/config_testbed.json")
    directTestbed_path = os.path.join(testbed_path, testbed_name + ".h5")
    dist_sequenzes, dist_sequenzes_noisy, ideal_sequenzes, ideal_sequenzes_noisy, sequenzes = load_RawData_from_hdf5(directTestbed_path)
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
def displayLearningRateHistory(config_path: str = "config/config.json"):
    """
    Displays and saves the history of the Learning Rate.
    
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
        display_learning_rate_history(history_df= history_df, model_path=model_path, testbed_name=testbed_name)
    else:
        typer.echo("No training history found at the specified path.")

@app.command()
def displayLearningRateAndLossHistoryInOneGraph(config_path: str = "config/config.json"):
    """
    Displays and saves the history of the Learning Rate.
    
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
        display_learning_rate_history_and_loss_in_one_plot(history_df= history_df, model_path=model_path, testbed_name=testbed_name)
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
def plot_real_values():
    """
    Plot a real signal to extract capacitance values.
    """
    config_path = "config/config.json"
    cfg = load_config(config_path)

    real_signal_df = pd.read_csv("RealeMessungen/Messung_20260203_144900.csv")

    plot_real_signal(real_signal_df)

@app.command()
def plot_Bartunik_data():
    """
    Plot a real signal to extract capacitance values.
    """
    config_path = "config/config.json"
    cfg = load_config(config_path)

    real_signal_df = pd.read_csv("RealeMessungen/1.00s.csv")

    plot_new_signal_intervals(real_signal_df)

@app.command()
def test_model_on_Bartunik_Data(config_path: str = "config/config.json"):
    """
    Test the model on real data from Bartunik.
    """

    # --------------------- Config laden -------------------------
    cfg = load_config(config_path)
    testbedcfg = load_config("config/config_testbed.json")
    modelcfg = load_config("config/config_model.json")

    # --------------------- Model- und Testbednamen holen ---------
    model_path, model_name, testbed_path, testbed_name, MLDataset_path = getModelandTestbedName(config_path, modelcfg_path="config/config_model.json", testbedcfg_path="config/config_testbed.json")
    directTestbed_path = os.path.join(testbed_path, testbed_name + ".h5")
    # --------------------- Seeds setzen -------------------------
    set_random_seed(cfg["RANDOM_SEED"])

    # --------------------- Daten vorbereiten --------------------
    testbed_path = "./Datasets/RealDataBartunik2023"
    testbed_name = "RealDataBartunik2023"
    MLDataset_path = "./Datasets/RealDataBartunik2023"
    Gen = DataGenerator()
    t = generate_Timeline(testbedcfg["T_START"], testbedcfg["T_STOP"], testbedcfg["T_STEP"])
    X_train, X_test, y_train, y_test, X_val, y_val = prepare_data(testbedcfg, dataset_path= directTestbed_path, MLDataset_path=MLDataset_path, Gen=Gen, t=t, random_state=cfg["RANDOM_SEED"], use_pure_test_set=cfg["USE_PURE_TEST_SET"], test_size=cfg["TEST_SIZE_RATIO"])

    # --------------------- Modell erstellen ---------------------
    model_path = cfg["MODEL_PATH"]
    # --------------------- Gewichte laden -----------------------
    if os.path.exists(model_path):
        model = tf.keras.saving.load_model(model_path)
    else:
        typer.echo("No existing model found at the specified path.")
        return

    # --------------------- Evaluation im Trainer ----------------
    sequence_index = cfg["SEQUENCE_INDEX"]
    if sequence_index >= len(X_test):
        typer.echo(f"Sequence index {sequence_index} is out of bounds for the test set with {len(X_test)} samples.")
        sequence_index = 0
        typer.echo(f"Defaulting to sequence index 0.")
    y_pred, results = evaluateModel(cfg, model, model_path, X_test, y_test)
    print(f"Prediction {sequence_index}: {(y_pred[sequence_index])}, True Labels: {y_test[sequence_index]}")

@app.command()
def rescale_data_and_train_on_standardized_data(config_path: str = "config/config.json"):
    """
    Takes Dataset and rescales it to a certain range, (mean 0 variance 1) and trains the model on this standardized data.
    After training, the model is evaluated on the Bartunik Data and the results are printed.
    """

    # --------------------- Config laden -------------------------
    cfg = load_config(config_path)
    testbedcfg = load_config("config/config_testbed.json")
    modelcfg = load_config("config/config_model.json")
    Gen = DataGenerator()
    t = generate_Timeline(testbedcfg["T_START"], testbedcfg["T_STOP"], testbedcfg["T_STEP"])
    scaler = StandardScaler()

    # --------------------- Model- und Testbednamen holen ---------
    model_path, model_name, testbed_path, testbed_name, MLDataset_path = getModelandTestbedName(config_path, modelcfg_path="config/config_model.json", testbedcfg_path="config/config_testbed.json")
    directTestbed_path = os.path.join(testbed_path, testbed_name + ".h5")
    # --------------------- Seeds setzen -------------------------
    set_random_seed(cfg["RANDOM_SEED"])

    X_train, X_test, y_train, y_test, X_val, y_val = prepare_data(testbedcfg, directTestbed_path, MLDataset_path, Gen, t, random_state=cfg["RANDOM_SEED"], use_pure_test_set=cfg["USE_PURE_TEST_SET"], test_size=cfg["TEST_SIZE_RATIO"])
    model_instance = CBLSTM() # create an instance of the model class
    print(np.shape(X_train), np.shape(X_test), np.shape(X_val))
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    X_val = scaler.transform(X_val)
    print(f"New Shape: {X_train.shape}, {X_test.shape}, {X_val.shape}")
    #--------------------------------------------------------------------------------
    
    model = model_instance.create_model(cfg=modelcfg, testbedcfg=testbedcfg,
        learning_rate=modelcfg["LEARNING_RATE"],
        filters=modelcfg["FILTERS"],
        num_of_conv_Layers=modelcfg["NUM_OF_CONV_LAYERS"],
        lstm_units=modelcfg["LSTM_UNITS"],
        lstm_layers=modelcfg["LSTM_LAYERS"],
        dropout_rate=modelcfg["DROPOUT"],
        input_shape= 170
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

    # --------------------- Daten vorbereiten --------------------
    testbed_path = "./Datasets/RealDataBartunik2023_10Bit"
    testbed_name = "RealDataBartunik2023_10Bit"
    MLDataset_path = "./Datasets/RealDataBartunik2023_10Bit"
    print("Loading Bartunik Data for Transfer Learning...")
    X_train, X_test, y_train, y_test, X_val, y_val = prepare_data(testbedcfg, dataset_path= testbed_path, MLDataset_path=MLDataset_path, Gen=Gen, t=t, random_state=cfg["RANDOM_SEED"], use_pure_test_set=True, test_size=cfg["TEST_SIZE_RATIO"])
    X_test = scaler.transform(X_test)
    print(f"New Shape: {np.shape(X_test)}")
    X_test = np.expand_dims(X_test, axis=-1)
    print(model.input_shape)
    print(X_test.shape)
    
    X_train, X_test, y_train, y_test, X_val, y_val = split_dataset(X_test, y_test, test_size=0.2, random_state=cfg["RANDOM_SEED"])

    # --------------------- Re-Compile Model --------------------
    
    #for layer in model.layers:
    #    layer.trainable = False

    #trainable = {"dense", "bidirectional_1", "bidirectional"}
    #for layer in trainable:
    #    model.get_layer(layer).trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-04),
        loss='binary_crossentropy', 
        metrics=[
                tf.keras.metrics.BinaryAccuracy(), 
                tf.keras.metrics.Precision(), 
                tf.keras.metrics.Recall()
                ]
    )

    # --------------------- Re-Fit auf Bartunik Data --------------------
    print("Applying Transfer Learning...")
    model = applyTransferLearning(
                                model, 
                                X_train, 
                                y_train, 
                                X_val, 
                                y_val, 
                                callbacks, 
                                batch_size=5, 
                                epochs=50)

    # --------------------- Evaluation im Trainer ----------------
    sequence_index = cfg["SEQUENCE_INDEX"]
    if sequence_index >= len(X_test):
        typer.echo(f"Sequence index {sequence_index} is out of bounds for the test set with {len(X_test)} samples.")
        sequence_index = 0
        typer.echo(f"Defaulting to sequence index 0.")
    y_pred, results = evaluateModel(cfg, model, model_path, X_test, y_test)
    print(f"Prediction {sequence_index}: {np.round(y_pred[sequence_index])}, True Labels: {y_test[sequence_index]}")

    # --------------------- Teste verschieden metric thresholds ----------------
    # Ansatz hat nicht geklappt an der Metric wird es nciht liegen. 
    #y_prob = model.predict(X_val, verbose=0)

    #thresholds = np.arange(0.1, 0.91, 0.05)

    #best_threshold = None
    #best_ber = float("inf")

    #for th in thresholds:
    #    y_pred = (y_prob >= th).astype(int)
    #    ber = np.mean(y_pred != y_val)
    #    print(f"Threshold {th:.2f}: BER = {ber:.6f}")

    #    if ber < best_ber:
    #        best_ber = ber
    #        best_threshold = th

    #print(f"\nBest threshold on validation: {best_threshold:.2f}")
    #print(f"Best validation BER: {best_ber:.6f}")
    #y_test_prob = model.predict(X_test, verbose=0)
    #y_test_pred = (y_test_prob >= best_threshold).astype(int)
    #test_ber = np.mean(y_test_pred != y_test)

    #print(f"Test BER at threshold {best_threshold:.2f}: {test_ber:.6f}")

@app.command()
def train_on_bartunik_only(config_path: str = "config/config.json"):
    """
    Trains a new model on only the bartunik data to derive standard performance ratings
    After training, the model is evaluated and the results are printed.
    """

    # --------------------- Config laden -------------------------
    cfg = load_config(config_path)
    testbedcfg = load_config("config/config_testbed.json")
    modelcfg = load_config("config/config_model.json")
    Gen = DataGenerator()
    t = generate_Timeline(testbedcfg["T_START"], testbedcfg["T_STOP"], testbedcfg["T_STEP"])

    # --------------------- Model- und Testbednamen holen ---------
    model_path, model_name, testbed_path, testbed_name, MLDataset_path = getModelandTestbedName(config_path, modelcfg_path="config/config_model.json", testbedcfg_path="config/config_testbed.json")
    
    # --------------------- Seeds setzen -------------------------
    set_random_seed(cfg["RANDOM_SEED"])

    # --------------------- Daten vorbereiten --------------------
    testbed_path = "./Datasets/RealDataBartunik2023_10Bit_Impulse_Signal"
    testbed_name = "RealDataBartunik2023_10Bit_Impulse_Signal"
    MLDataset_path = "./Datasets/RealDataBartunik2023_10Bit_Impulse_Signal"
    print("Loading Bartunik Data for Transfer Learning...")
    X_train, X_test, y_train, y_test, X_val, y_val = prepare_data(testbedcfg, dataset_path= testbed_path, MLDataset_path=MLDataset_path, Gen=Gen, t=t, random_state=cfg["RANDOM_SEED"], use_pure_test_set=True, test_size=cfg["TEST_SIZE_RATIO"])
    #X_test = np.expand_dims(X_test, axis=-1)
    
    X_train, X_test, y_train, y_test, X_val, y_val = split_dataset(X_test, y_test, test_size=0.2, random_state=cfg["RANDOM_SEED"])
    print(f"length train: {len(X_train)}, length val {len(X_val)}, length test {len(X_test)}")
    
    #--------------------------------------------------------------------------------
    model_instance = CBLSTM() # create an instance of the model class
    model = model_instance.create_model(cfg=modelcfg, testbedcfg=testbedcfg,
        learning_rate=modelcfg["LEARNING_RATE"],
        filters=modelcfg["FILTERS"],
        num_of_conv_Layers=modelcfg["NUM_OF_CONV_LAYERS"],
        lstm_units=modelcfg["LSTM_UNITS"],
        lstm_layers=modelcfg["LSTM_LAYERS"],
        dropout_rate=modelcfg["DROPOUT"],
        input_shape= 170
    )
    callbacks = [
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
            EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True),
            tf.keras.callbacks.TensorBoard(log_dir="./logs", histogram_freq=1),
        ]
    model, history = trainModel(model, model_path, X_train, y_train, X_val, y_val, callbacks, modelcfg["BATCH_SIZE"], modelcfg["EPOCHS"])
    
    history_df = pd.DataFrame(history.history)
    hist_csv_path = os.path.join(os.path.dirname(model_path), model_name, testbed_name,  "training_history.csv")
    os.makedirs(os.path.dirname(hist_csv_path), exist_ok=True)
    with open(hist_csv_path, mode='w+') as f:
        history_df.to_csv(f)

    # --------------------- Evaluation im Trainer ----------------
    sequence_index = cfg["SEQUENCE_INDEX"]
    if sequence_index >= len(X_test):
        typer.echo(f"Sequence index {sequence_index} is out of bounds for the test set with {len(X_test)} samples.")
        sequence_index = 0
        typer.echo(f"Defaulting to sequence index 0.")
    y_pred, results = evaluateModel(cfg, model, model_path, X_test, y_test)
    print(f"Prediction {sequence_index}: {np.round(y_pred[sequence_index])}, True Labels: {y_test[sequence_index]}")



@app.command()
def extract_data_from_Bartunik():
    """
    Create a Dataset of 30 sample long arrays out of Bartunik data to test the model on real data.
    """

    real_signal_df = pd.read_csv("RealeMessungen/1.00s.csv")

    extract_dataset_from_Bartunik_Data(real_signal_df, transform_to_volume_signal=True)

@app.command()
def plot_example(config_path: str = "config/config.json"):
    """
    Plots an example sequence based on the configuration.
    
    :param config_path: Path to the configuration file.
    :type config_path: str
    """
    cfg = load_config(config_path)
    testbedcfg = load_config("config/config_testbed.json")
    model_path, model_name, testbed_path, testbed_name, MLDataset_path = getModelandTestbedName(config_path, modelcfg_path="config/config_model.json", testbedcfg_path="config/config_testbed.json")
    directTestbed_path = os.path.join(testbed_path, testbed_name + ".h5")
    Gen = DataGenerator()
    t = generate_Timeline(testbedcfg["T_START"], testbedcfg["T_STOP"], testbedcfg["T_STEP"])
    dist_sequenzes, dist_sequenzes_noisy, ideal_sequenzes, ideal_sequenzes_noisy, sequenzes = getOrCreateTimeSeriesData(testbedcfg,directTestbed_path, Gen, t)
    index = cfg["SEQUENCE_INDEX"]
    print(f"Plotted Sequence: {sequenzes[index]}")
    z_varyRx, z_statRx, z_depth_vector, weight_function = Gen.sub_ReceiverPosition(t, testbedcfg["Z_AMPL"], testbedcfg["F_RX"],  testbedcfg["Z_OFFSET"], testbedcfg["Z_DEPTH"], channel_radius=testbedcfg["CHANNEL_RADIUS"], channel_wall_thickness=testbedcfg["CHANNEL_WALL_THICKNESS"], U=testbedcfg["U"])
    plot_weightingFunction(weight_function, testbed_path)
    plot_a_sequence(t,dist_sequenzes, ideal_sequenzes, dist_sequenzes_noisy, ideal_sequenzes_noisy, z_varyRx, sequence_index=cfg["SEQUENCE_INDEX"], testbed_path=testbed_path)

@app.command()
def plot_impulse_response(config_path: str = "config/config.json"):
    """
    Plots the impulse response of the system based on the configuration.
    
    :param config_path: Path to the configuration file.
    :type config_path: str
    """

    impulse_response_data = pd.read_csv("RealeMessungen/50mm.csv")
    impulse_response = impulse_response_data["value"].values
    t = impulse_response_data["time"].values
    h_norm, imp_t= extractMean_and_plot_impulse_response(impulse_response, t, os.path.join("./RealeMessungen", "impulse_response.png"))
    

@app.command()
def create_dataset_modified_through_impulse_response(config_path: str = "config/config.json"):
    """
    Creates a dataset by modifying the generated sequences with a real impulse response.
    
    :param config_path: Path to the configuration file.
    :type config_path: str
    """

    testbed_path = "./Datasets/RealDataBartunik2023_10Bit_Impulse_Signal"
    testbed_name = "RealDataBartunik2023_10Bit_Impulse_Signal"
    MLDataset_path = "./Datasets/RealDataBartunik2023_10Bit_Impulse_Signal/MLDatasets"

    change_config("TESTBED_PATH", testbed_path, config_path)
    change_config("TESTBED_NAME", testbed_name, config_path)
    change_config("MLDATASET_NAME", MLDataset_path, config_path)


    cfg = load_config(config_path)
    testbedcfg = load_config("config/config_testbed.json")
    model_path, model_name, testbed_path, testbed_name, MLDataset_path = getModelandTestbedName(config_path, modelcfg_path="config/config_model.json", testbedcfg_path="config/config_testbed.json")
    directTestbed_path = os.path.join(testbed_path, testbed_name + ".h5")

    impulse_response_data = pd.read_csv("RealeMessungen/50mm.csv")
    impulse_response = impulse_response_data["value"].values
    t = impulse_response_data["time"].values
    h_norm, imp_t= extractMean_and_plot_impulse_response(impulse_response, t, os.path.join("./RealeMessungen", "impulse_response.png"))

    Gen = DataGenerator()
    t = generate_Timeline(testbedcfg["T_START"], testbedcfg["T_STOP"], testbedcfg["T_STEP"])
    number_arrays = 10000
    number_bits = 10
    signal, signal_noisy, sequences = Gen.create_synthetic_bartunik_dataset( t, number_arrays, number_bits, unique = True, c_0 = 20, snr = 7, h_norm = h_norm)
    save_modified_dataset(signal, signal_noisy, sequences, directTestbed_path, cfg)

    
    print("MLDataset does not exist. Creating MLDataset...")   
    [X_train, X_test, y_train, y_test, X_val, y_val, X_test_pure, y_test_pure] = create_MLDataset(MLDataset_path, signal_noisy, sequences, test_size=0.2, random_state= cfg["RANDOM_SEED"])

@app.command()
def plot_bartunik_example(config_path: str = "config/config.json"):
    """
    Plots an example sequence from the Bartunik dataset.
    """
    cfg = load_config(config_path)
    testbedcfg = load_config("config/config_testbed.json")
    model_path, model_name, testbed_path, testbed_name, MLDataset_path = getModelandTestbedName(config_path, modelcfg_path="config/config_model.json", testbedcfg_path="config/config_testbed.json")
    directTestbed_path = os.path.join(testbed_path, testbed_name + ".h5")
    Gen = DataGenerator()
    t = generate_Timeline(testbedcfg["T_START"], testbedcfg["T_STOP"], testbedcfg["T_STEP"])
    dist_sequenzes, dist_sequenzes_noisy, sequenzes = load_RawData_from_Bartunik_hdf5(directTestbed_path)
    index = cfg["SEQUENCE_INDEX"]
    print(f"Plotted Sequence: {sequenzes[index]}")
    plot_a_Bartunik_sequence(t,dist_sequenzes, dist_sequenzes_noisy, sequence_index=cfg["SEQUENCE_INDEX"], testbed_path=testbed_path)

@app.command()
def runOptimization(config_path: str = "config/config.json"):
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    print("Loading Config and running optimization with Optuna...")
    cfg = load_config(config_path)
    testbedcfg = load_config("config/config_testbed.json")

    modelinstance = CBLSTM() # create an instance of the model class

    Gen = DataGenerator()
    t = generate_Timeline(testbedcfg["T_START"], testbedcfg["T_STOP"], testbedcfg["T_STEP"])
    set_random_seed(cfg["RANDOM_SEED"])
    model_path, model_name, testbed_path, testbed_name, MLDataset_path = getModelandTestbedName(config_path, modelcfg_path="config/config_model.json", testbedcfg_path="config/config_testbed.json")
    directTestbed_path = os.path.join(testbed_path, testbed_name + ".h5")
    X_train, X_test, y_train, y_test, X_val, y_val = prepare_data(testbedcfg, directTestbed_path, MLDataset_path, Gen, t, random_state=cfg["RANDOM_SEED"], use_pure_test_set=cfg["USE_PURE_TEST_SET"], test_size=cfg["TEST_SIZE_RATIO"])
    print("Dataset loaded. Starting optimization...")

    # -------------------------
    # Run optimization
    # -------------------------
    results = run_optuna_study(
        modelinstance=modelinstance,
        cfg=cfg,
        testbedcfg=testbedcfg,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        n_trials=50,
        study_name="my_study"
    )

    print("\nBest Parameters:")
    print(results["best_params"])
    print(f"Best Score: {results['best_value']:.4f}")

@app.command()
def create_comparison(config_path: str = "config/config.json"):
    """
    Creates a comparison of different datasets based on the user input.
    
    :param config_path: Path to the configuration file.
    :type config_path: str
    """
    cfg = load_config(config_path)
    testbedcfg = load_config("config/config_testbed.json")
    Gen = DataGenerator()
    t = generate_Timeline(testbedcfg["T_START"], testbedcfg["T_STOP"], testbedcfg["T_STEP"])

    directTestbed_path_1 = os.path.join("./Datasets/complete_dataset_fd3d67", "complete_dataset_fd3d67" + ".h5")
    directTestbed_path_2 = os.path.join("./Datasets/complete_dataset_bf1490", "complete_dataset_bf1490" + ".h5")

    dist_sequenzes_1, dist_sequenzes_noisy_1, ideal_sequenzes_1, ideal_sequenzes_noisy_1, sequenzes_1 = getOrCreateTimeSeriesData(testbedcfg,directTestbed_path_1, Gen, t)
    dist_sequenzes_2, dist_sequenzes_noisy_2, ideal_sequenzes_2, ideal_sequenzes_noisy_2, sequenzes_2 = getOrCreateTimeSeriesData(testbedcfg,directTestbed_path_2, Gen, t)

    index = cfg["SEQUENCE_INDEX"]
    z_varyRx, z_statRx, z_depth_vector, weight_function = Gen.sub_ReceiverPosition(t, testbedcfg["Z_AMPL"], testbedcfg["F_RX"],  testbedcfg["Z_OFFSET"], testbedcfg["Z_DEPTH"], channel_radius=testbedcfg["CHANNEL_RADIUS"], channel_wall_thickness=testbedcfg["CHANNEL_WALL_THICKNESS"], U=testbedcfg["U"])

    

    plot_two_distinct_sequences(t,dist_sequenzes_1, ideal_sequenzes_1, dist_sequenzes_noisy_1, ideal_sequenzes_noisy_1, dist_sequenzes_noisy_2, ideal_sequenzes_noisy_2, z_varyRx, index)
    
    

if __name__ == "__main__":
    app()
