import os

import numpy as np

import tensorflow as tf
import typer

from model.model import CBLSTM


def trainModel(model, modelpath, feature_files, label_files, feature_files_val, label_files_val, callbacks, cfg):

    print("feature_files_shape", feature_files.shape)
    print("label_files_shape",label_files.shape)

    trainGenerator = ValueDataGenerator(feature_files, label_files, batch_size=cfg["BATCH_SIZE"])
    valGenerator = ValueDataGenerator(feature_files_val, label_files_val, batch_size=cfg["BATCH_SIZE"])

    #--------------------------------------------------------------------------------
    if not os.path.exists(modelpath):
        # Training mit den gewählten Parametern
        history = model.fit(
            trainGenerator,
            validation_data=valGenerator,
            epochs=cfg["EPOCHS"],
            verbose=1,
            callbacks=callbacks,
        )
        
        os.makedirs(os.path.dirname(modelpath), exist_ok=True)
        model.save(modelpath)
        typer.echo(f"Modell trained and saved: {modelpath}")
    else:
        typer.echo("The Modell already exists. Please delete the existing model to retrain. Or skip training.")
    return model, history

def evaluateModel(model, feature_files_test, label_files_test):
    
    results = model.evaluate(feature_files_test, label_files_test)
    results = dict(zip(model.metrics_names, results))

    y_pred = model.predict(feature_files_test)

    return y_pred,results

class ValueDataGenerator(tf.keras.utils.Sequence):
  def __init__(self, feature_files, label_files, batch_size=8,shuffle=True,):
    self.feature_files = feature_files
    self.label_files = label_files
    self.batch_size = batch_size
    self.feature_files = feature_files
    self.label_files = label_files
    self.shuffle = shuffle
    self.indices = np.arange(len(self.feature_files))
    self.on_epoch_end()
    
    
    
    
  def __len__(self):
    # returns the number of batches
    return int(len(self.feature_files) / self.batch_size)
            
      
  def __getitem__(self, index):
      'Generate one batch of data'
      # Generate file-indexes of the batch
      indexes = self.indices[index*self.batch_size:(index+1)*self.batch_size]
      #print('index',index)
      #print(indexes)
      # Find list of files
      feature_files_tmp = self.feature_files[indexes]
      label_files_tmp = self.label_files[indexes]
      
      # get one Batch of Data
      feature_files_tmp = feature_files_tmp[..., np.newaxis]
      #print("feature_filestmp_shape", feature_files_tmp.shape)
      #print("label_filestmp_shape", label_files_tmp.shape)

      return feature_files_tmp, label_files_tmp
    
      
  def on_epoch_end(self):
    #Daten werden nach Epoch neu gemischt
    'Updates indexes after each epoch'
    if self.shuffle == True:
        np.random.shuffle(self.indices)


#-------------------------------------------------------------------------------------------------
#       Testgenerator
#
#       Operiert Analog zum normalen Generator allerdings ohne Label files.
#       
#-------------------------------------------------------------------------------------------------
class TestGenerator(tf.keras.utils.Sequence):
    def __init__(self, feature_files, batch_size, preprocess_fn=None):
        """
        feature_files: numpy array (n_samples, H, W, C) ODER Pfadliste
        batch_size: int
        preprocess_fn: optionale Funktion: x -> x_preprocessed
        """
        self.feature_files = feature_files
        self.batch_size = batch_size
        self.preprocess_fn = preprocess_fn

    def __len__(self):
        return int(np.ceil(len(self.feature_files) / self.batch_size))

    def __getitem__(self, idx):
        batch_x = self.feature_files[
            idx * self.batch_size : (idx + 1) * self.batch_size
        ]

        if self.preprocess_fn:
            batch_x = self.preprocess_fn(batch_x)

        return batch_x