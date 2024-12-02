from model import CBLSTM
from DataGeneration import DataGenerator
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import tensorflow as tf
import os
import pandas as pd
from scikeras.wrappers import KerasClassifier


BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 10
EPOCHS = 10
NUMBER_OF_ARRAYS = 15
NUMBER_OF_BITS = 13

# Define the custom KerasClassifier subclass
class CustomKerasClassifier(KerasClassifier):
    def __init__(self, build_fn, **kwargs):
        super().__init__(build_fn=build_fn, **kwargs)
        self.build_fn = build_fn
        for key, value in kwargs.items():
            setattr(self, key, value)

    def create_model(self):
        # Call the build function with all custom parameters passed in __init__
        return self.build_fn(
            learning_rate=self.learning_rate,
            filters=self.filters,
            num_of_conv_Layers=self.num_of_conv_Layers,
            lstm_units=self.lstm_units,
            lstm_layers=self.lstm_layers
        )


def build_model(learning_rate=0.001, filters=[32, 64, 128, 256], num_of_conv_Layers=3, lstm_units=13, lstm_layers=3):
    model_instance = CBLSTM()
    return model_instance.create_model(
        learning_rate=learning_rate,
        filters=filters,
        num_of_conv_Layers=num_of_conv_Layers,
        lstm_units=lstm_units,
        lstm_layers=lstm_layers
    )

def main():
    

    [X_train, X_test, y_train, y_test, X_val, y_val , train_dataset, val_dataset, test_dataset] = create_Dataset(NUMBER_OF_ARRAYS, NUMBER_OF_BITS, 0.20, 42)

    batch_size = [10, 20, 30, 64, 128]
    epochs = [10, 30, 50]
    learning_rate = [0.005, 0.0025, 0.001, 0.0001]
    filters = [[32,64,128,256],[32,32,64,64,128,128,256,256]]
    num_of_conv_Layers = [3, 5]
    lstm_units = [13, 26]
    lstm_layers = [3, 5, 7]
    param_grid_training = dict(optimizer__learning_rate=learning_rate, batch_size=batch_size, epochs=epochs)
    param_grid_model = dict(filters=filters, num_of_conv_Layers=num_of_conv_Layers, lstm_units=lstm_units, lstm_layers=lstm_layers)

    model = CustomKerasClassifier(build_fn= build_model, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose = 1)

    #Search for Model Parameters
    grid_result_model = GridSearchCV(estimator=model, param_grid=param_grid_model, n_jobs=-1, cv=3)
    grid_result_model = grid_result_model.fit(X_train, y_train)
    print("Best: %f using %s" % (grid_result_model.best_score_, grid_result_model.best_params_))

    results_df = pd.DataFrame(grid_result_model.cv_results_)
    results_df.to_csv("grid_search_results_model.csv", index=False)

    model = CustomKerasClassifier(buidl_fn = build_model, verbose = 1)

    #Search for Training Parameters
    grid_result_model = GridSearchCV(estimator=model, param_grid=param_grid_training, n_jobs=-1, cv=3)
    grid_result_training = grid_result_model.fit(X_train, y_train)
    print("Best: %f using %s" % (grid_result_training.best_score_, grid_result_training.best_params_))

    results_df = pd.DataFrame(grid_result_training.cv_results_)
    results_df.to_csv("grid_search_results_training.csv", index=False)

    #model_instance.train_cnn_Model(train_dataset, val_dataset, batch_size=BATCH_SIZE, epochs=EPOCHS)



    #y_pred = model_instance.evaluate_model(feature_files=X_test)
    #y_pred = model_instance.evaluate_modelDataset(test_dataset)
    #bit_preds = (y_pred >= 0.5).astype(int)
    #print(y_pred[1])
    #print(bit_preds[1])
    #print(y_test[1])





def create_Dataset(number_of_Arrays, number_of_bits, test_size, random_state):
    Gen = DataGenerator()
    [t, dist_sequenzes, ideal_sequenzes,sequenzes] = Gen.createDataSet(number_of_Arrays, number_of_bits)
    X_train, X_test, y_train, y_test = train_test_split(dist_sequenzes,
                                                    sequenzes,
                                                    test_size=test_size,
                                                    random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.20, random_state=random_state)
    X_train, X_test, y_train, y_test, X_val, y_val = map(np.array, [X_train, X_test, y_train, y_test, X_val, y_val])

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    val_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    val_dataset = val_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    test_dataset = test_dataset.batch(BATCH_SIZE)

    return X_train, X_test, y_train, y_test, X_val, y_val , train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    main()
    