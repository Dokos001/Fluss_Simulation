import os

import tensorflow as tf
import typer

from model.model import CBLSTM


def trainModel(model, modelpath, feature_files, label_files, feature_files_val, label_files_val, callbacks, cfg):


    #--------------------------------------------------------------------------------
    if not os.path.exists(modelpath):
        # Training mit den gewählten Parametern
        model.fit(
            feature_files, label_files,
            validation_data=(feature_files_val, label_files_val),
            batch_size=cfg["BATCH_SIZE"],
            epochs=cfg["EPOCHS"],
            verbose=1,
            callbacks=callbacks,
        )
        
        os.makedirs(os.path.dirname(modelpath), exist_ok=True)
        model.save(modelpath)
        typer.echo(f"Modell trained and saved: {modelpath}")
    else:
        typer.echo("The Modell already exists. Please delete the existing model to retrain. Or skip training.")
    return model