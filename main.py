
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
from model.tools import display_train_val_loss, generate_MLDataset_name, load_MLDataset, create_MLDataset, create_pureTest_MLDataset, generate_RawDataset_name, load_RawData_from_hdf5,save_complete_dataset, generate_Timeline
import random, os
import typer
from model.noiseAnalytics import noiseAnalyser
from view.plotdata import plot_noisy_signals, plot_accuracy, plot_a_sequence
from model.DataGeneration import DataGenerator


app = typer.Typer()


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

def prepare_data(cfg, Gen, t):
    """Erzeugt oder lädt die Trainings-/Testdaten entsprechend der Konfiguration."""
    [X_train, X_test, y_train, y_test, X_val, y_val] = None, None, None, None, None, None
    dataset_path = generate_MLDataset_name(cfg)
    if os.path.exists(dataset_path):
        [X_train, X_test, y_train, y_test, X_val, y_val] = load_MLDataset(dataset_path)
    else:
        dist_sequenzes, dist_sequenzes_noisy, ideal_sequenzes, ideal_sequenzes_noisy, sequenzes = getOrCreateTimeSeriesData(cfg, Gen, t)
        if cfg["PURE_TEST_SET"]:
            X_test, y_test = create_pureTest_Dataset(
                number_of_Arrays=cfg["NUMBER_OF_ARRAYS"],
                number_of_bits=cfg["NUMBER_OF_BITS"],
                random_state=cfg["RANDOM_SEED"],
                time_variable=cfg["TIME_VARIABLE"],
                unique=cfg["UNIQUE"]
            )
            [X_train, X_test, y_train, y_test, X_val, y_val] = None, X_test, None, y_test, None, None
        else:
            [X_train, X_test, y_train, y_test, X_val, y_val] = create_MLDataset(dataset_path, dist_sequenzes_noisy, sequenzes, test_size=0.2, random_state=cfg["RANDOM_SEED"], time_variable=cfg["TIME_VARIABLE"])
    return [X_train, X_test, y_train, y_test, X_val, y_val]

def getOrCreateTimeSeriesData(cfg, Gen, t):
    
    dataset_path = generate_RawDataset_name(cfg)
    if os.path.exists(dataset_path):
        print(f"Loading existing dataset from {dataset_path}...")
        dist_sequenzes, dist_sequenzes_noisy, ideal_sequenzes, ideal_sequenzes_noisy, sequenzes = load_RawData_from_hdf5(dataset_path)
    else:
        print("Creating new dataset...")
        [dist_sequenzes, ideal_sequenzes, dist_sequenzes_noisy, ideal_sequenzes_noisy, sequenzes] = Gen.createDataSet(t = t, number_arrays = cfg["NUMBER_OF_ARRAYS"], 
                                                                                                                      number_bits = cfg["NUMBER_OF_BITS"], 
                                                                                                                      unique = cfg["UNIQUE"], 
                                                                                                                      DimReceiver = cfg["RECEIVER_DIMENSION_3D"], 
                                                                                                                      f_rx = cfg["F_RX"], z_offset = cfg["Z_OFFSET"], 
                                                                                                                      z_depth = cfg["Z_DEPTH"], dz = cfg["DZ"], 
                                                                                                                      v_0 = cfg["V_0"], 
                                                                                                                      c_0 = cfg["C_0"],
                                                                                                                      U = cfg["U"],
                                                                                                                      channel_radius=cfg["CHANNEL_RADIUS"],
                                                                                                                      channel_wall_thickness=cfg["CHANNEL_WALL_THICKNESS"])
        save_complete_dataset(dist_sequenzes, dist_sequenzes_noisy, ideal_sequenzes, ideal_sequenzes_noisy, sequenzes, dataset_path, cfg)

    return dist_sequenzes, dist_sequenzes_noisy, ideal_sequenzes, ideal_sequenzes_noisy, sequenzes
    

def get_model_path(cfg: dict) -> str:
    suffix = ""
    if not cfg["TIME_VARIABLE"]:
        suffix += "_static_Receiver"
    if cfg["UNIQUE"]:
        suffix += "_Unique"
    suffix += f"f_rx{str(cfg['F_RX']).replace('.', '')}"
    return f"NNModels/best_param{suffix}_model.keras"
#-------------------------- Hauptprogramm ----------------------------------------

@app.command()
def trainModel(config_path: str = "config/config.json"):

    cfg = load_config(config_path)
    #-------------------------- Random Parameter initialization ---------------------

    set_random_seed(cfg["RANDOM_SEED"])

    #--------------------------------------------------------------------------------
    X_train, X_test, y_train, y_test, X_val, y_val = prepare_data(cfg)
    model_instance = CBLSTM() # create an instance of the model class
    model_path = get_model_path(cfg)

    #--------------------------------------------------------------------------------

    
    if os.path.exists(model_path):
    # Modell erstellen
        model = model_instance.create_model(
            learning_rate=cfg["LEARNING_RATE"],
            filters=cfg["FILTERS"],
            num_of_conv_Layers=cfg["NUM_OF_CONV_LAYERS"],
            lstm_units=cfg["LSTM_UNITS"],
            lstm_layers=cfg["LSTM_LAYERS"],
            dropout_rate=cfg["DROPOUT"],
        )
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model.save(model_path)
        typer.echo(f"Modell gespeichert unter: {model_path}")
        display_train_val_loss()
    else:

        typer.echo("Lade existierendes Modell ...")
        model = tf.keras.models.load_model(model_path)
    
# ---------------------------------------------------------------------
# Command: evaluate
# ---------------------------------------------------------------------
@app.command()
def evaluate(
    config_path: str = "config/config.json",
    model_path: str = typer.Option(None, help="Pfad zum gespeicherten Modell (überschreibt Config)")
):
    
    """Lädt ein gespeichertes Modell und evaluiert es auf Testdaten."""
    cfg = load_config(config_path)
    set_random_seed(cfg["RANDOM_SEED"])

    X_train, X_test, y_train, y_test, X_val, y_val = prepare_data(cfg)
    model_path = model_path or get_model_path(cfg)

    typer.echo(f"Lade Modell: {model_path}")
    model = tf.keras.models.load_model(model_path)

    typer.echo("Starte Evaluation ...")
    y_pred = model.predict(X_test)

    # Save predictions
    os.makedirs("data/processed", exist_ok=True)
    df = pd.DataFrame(y_pred)
    suffix = os.path.splitext(os.path.basename(model_path))[0]
    df.to_csv(f"data/processed/y_pred_{suffix}.csv", header=False, index=False)


    # Metrics
    bin_pred = [np.where(a > 0.5, 1, 0) for a in y_pred]
    acc = accuracy_score(y_test, bin_pred)
    ber = np.mean(np.not_equal(bin_pred, y_test))
    typer.echo(f"Accuracy: {acc:.4f}")
    typer.echo(f"BER: {ber:.6f}")
    typer.echo(f"Evaluation abgeschlossen.")


@app.command()
def analyse_noise(config_path: str = "config/config.json", model_path: str = None):
    t = timeline(config_path)
    cfg = load_config(config_path)
    analyser = noiseAnalyser(cfg)
    gen = DataGenerator(cfg)

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
    """Zeigt aktuelle Konfiguration an"""
    cfg = load_config(config_path)
    typer.echo(json.dumps(cfg, indent=4))

@app.command()
def create_config(config_path: str = "config/config.json"):
    """Erstellt eine Standardkonfigurationsdatei"""
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
    typer.echo(f"Standardkonfiguration erstellt unter: {config_path}")

@app.command()
def plot_example(config_path: str = "config/config.json"):
    cfg = load_config(config_path)
    Gen = DataGenerator()
    t = generate_Timeline(cfg["T_START"], cfg["T_STOP"], cfg["T_STEP"])
    dist_sequenzes, dist_sequenzes_noisy, ideal_sequenzes, ideal_sequenzes_noisy, sequenzes = getOrCreateTimeSeriesData(cfg, Gen, t)
    z_varyRx, z_statRx, z_depth_vector, weight_function = Gen.sub_ReceiverPosition(t, cfg["Z_AMPL"], cfg["F_RX"],  cfg["Z_OFFSET"], cfg["Z_DEPTH"], channel_radius=cfg["CHANNEL_RADIUS"], channel_wall_thickness=cfg["CHANNEL_WALL_THICKNESS"], U=cfg["U"])
    plot_a_sequence(t,dist_sequenzes, ideal_sequenzes, dist_sequenzes_noisy, ideal_sequenzes_noisy, z_varyRx, sequence_index=cfg["SEQUENCE_INDEX"])

if __name__ == "__main__":
    app()
