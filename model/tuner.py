import os

import optuna
import pandas as pd
import tensorflow as tf


def build_objective(modelinstance, cfg, testbedcfg, X_train, y_train, X_val, y_val):

    def objective(trial):

        learning_rate = trial.suggest_float("learning_rate", 3e-4, 3e-3, log=True)

        filters = build_filters(trial)

        num_conv_layers = trial.suggest_int("num_conv_layers", 2, 4)

        lstm_units = trial.suggest_categorical("lstm_units", [32, 64, 128, 192])

        lstm_layers = trial.suggest_int("lstm_layers", 1, 2)

        batch_size = trial.suggest_categorical("batch_size", [16, 24, 32, 48])

        model = modelinstance.create_model(
            cfg=cfg,
            testbedcfg=testbedcfg,
            learning_rate=learning_rate,
            filters=filters,
            num_of_conv_Layers=num_conv_layers,
            lstm_units=lstm_units,
            lstm_layers=lstm_layers,
            dropout_rate=0.2
        )

        monitor_metric = "val_loss"

        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor=monitor_metric,
            patience=3,   
            restore_best_weights=True
        )

        pruning_callback = optuna.integration.TFKerasPruningCallback(
            trial,
            monitor=monitor_metric
        )

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,         
            batch_size=batch_size,
            verbose=0,
            callbacks=[early_stop, pruning_callback]
        )

        return min(history.history[monitor_metric])
    return objective


def run_optuna_study(
    modelinstance,
    cfg,
    testbedcfg,
    X_train,
    y_train,
    X_val,
    y_val,
    n_trials=50,
    study_name="optimization",
    storage=None 
):
    tf.config.optimizer.set_jit(False)
    

    print("Building objective function for Optuna study...")
    objective = build_objective(modelinstance, cfg, testbedcfg, X_train, y_train, X_val, y_val)

    print("Creating Optuna study...")
    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=3,
            n_warmup_steps=2,
            interval_steps=1
        )
    )
    optuna.logging.set_verbosity(optuna.logging.INFO)
    study.optimize(objective, n_trials=n_trials)

    print("Study completed. Saving results...")
    save_study_results(study, output_dir="optuna_results", study_name=study_name)

    return {
        "best_params": study.best_trial.params,
        "best_value": study.best_value,
        "study": study
    }

def save_study_results(study, output_dir="optuna_results", study_name="study"):
    """
    Saves:
    - all trials to CSV
    - best params to CSV
    """

    os.makedirs(output_dir, exist_ok=True)

    # -------------------------
    # All trials
    # -------------------------
    df = study.trials_dataframe()
    trials_path = os.path.join(output_dir, f"{study_name}_trials.csv")
    df.to_csv(trials_path, index=False)

    # -------------------------
    # Best params
    # -------------------------
    best_params = study.best_trial.params
    best_value = study.best_value

    best_df = pd.DataFrame([best_params])
    best_df["best_value"] = best_value

    best_path = os.path.join(output_dir, f"{study_name}_best.csv")
    best_df.to_csv(best_path, index=False)

    print(f"\nSaved trials to: {trials_path}")
    print(f"Saved best params to: {best_path}")

def build_filters(trial):
    base = trial.suggest_int("FILTER_BASE", 32, 64, step=32)

    depth = trial.suggest_int("FILTER_DEPTH", 2, 4)

    return [base * (2 ** i) for i in range(depth)]