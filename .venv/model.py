
import os
import tensorflow.keras.layers as layers
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping


class CBLSTM:

    model = []
    trained = []
    learning_rate = []
    batch_size = 2


    #-------------------------------------------------------------------------------------------------
    #       Initialisierungsfunktion
    #
    #       Legt die Learning Rate fest, sollte keine angegeben worden sein.
    #       Verwaltet ebenfalls das Training über die GPU
    #
    #       Training:
    #               über die GPU: "os.environ['CUDA_VISIBLE_DEVICES'] = '0' "
    #               über die CPU: "os.environ['CUDA_VISIBLE_DEVICES'] = '-1'"
    # 
    #       Sollte keine Learning Rate angegeben sein wird diese auf 0.25*10^-3 gesetzt.
    #-------------------------------------------------------------------------------------------------
    def __init__(self,
                  learning_rate = None, batch_size = None):
                tf.get_logger().setLevel('DEBUG')
                os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
                os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
                gpus = tf.config.list_physical_devices('GPU')
                os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
                os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
                os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


                if batch_size == None:
                        self.batch_size = 2
                else:
                        self.batch_size = batch_size

                if learning_rate == None:
                        self.learning_rate = 0.0001
                else:
                        self.learning_rate = learning_rate

    def create_model(self,learning_rate = 0.001, filters = [32,64,128,256], num_of_conv_Layers = 3, lstm_units = 13, lstm_layers = 3):
        # Initialize the model
        first = True

        input_shape = (2000,1)

        filters = filters

        #backward_layer = layers.LSTM(10, activation='relu', return_sequences=True, go_backwards=True)

        inputs = tf.keras.Input(shape=input_shape)
        #masked_inputs = layers.Masking(mask_value=-1.0)(inputs)

        initializer = tf.keras.initializers.GlorotUniform()

        res = inputs
        for filt in filters:

            for x in range(num_of_conv_Layers):
                if first:
                    convlayer = layers.Conv1D(filt,3, input_shape = input_shape[1:],padding= 'causal', activation = tf.nn.leaky_relu, kernel_initializer=initializer)(inputs)
                    first = False
                else:   
                    if x == 0:
                        convlayer = layers.Conv1D(filt,3, input_shape = input_shape[1:],padding= 'causal', activation = tf.nn.leaky_relu, kernel_initializer=initializer)(res)
                    else:
                        convlayer = layers.Conv1D(filt,3, input_shape = input_shape[1:],padding= 'causal', activation = tf.nn.leaky_relu, kernel_initializer=initializer)(convlayer)
                convlayer = layers.BatchNormalization()(convlayer)
            poollayer = layers.MaxPooling1D(pool_size=(2), padding = "same")(convlayer)
            dropout = layers.Dropout(0.2)(poollayer)
            res = layers.Conv1D(filt ,kernel_size=[1], strides=[2])(res)
            res = layers.add([res, dropout])
                    

        x = res
        for i in range(lstm_layers):
            return_seq = True if i < lstm_layers - 1 else False
            x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=return_seq, activation='relu'))(x)
            x = layers.Dropout(0.2)(x)
        bidirec = x

        #glob = layers.GlobalAveragePooling1D()(bidirec)
        #flatten = layers.Flatten()(glob)
        softmax = layers.Dense(13, activation='sigmoid')(bidirec)

        mdl = tf.keras.Model(inputs=inputs, outputs=softmax)

        adam = tf.keras.optimizers.Adam(learning_rate = learning_rate)
        # Compile the model
        mdl.compile(optimizer=adam, loss="categorical_crossentropy", metrics=['accuracy'])
        #Save a picture of the model
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'CBLSTM.png')
        tf.keras.utils.plot_model(mdl, save_path, show_shapes=True)
        # Summary of the model
        mdl.summary()
        self.model = mdl
        return mdl
    
    #-------------------------------------------------------------------------------------------------
    #       Trainingsfunktion
    #
    #       Training des Modells über die durch die Übergabe der Verzeichnisse definierten Daten
    #-------------------------------------------------------------------------------------------------        
    def train_cnn_Model(self, feature_files, label_files,
                        feature_files_val, label_files_val,
                        batch_size=2,
                        epochs=30):
        
        self.batch_size = batch_size
        
        #Speichern der Güte während des Trainings.
        #csv_logger = csv_logger('log.csv', append= True, separator=';')
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        trained = self.model.fit(feature_files, label_files,
                                 validation_data = (feature_files_val, label_files_val),
                        epochs = epochs,
                        #steps_per_epoch = steps_per_epoch,
                        #validation_steps = validation_steps,
                        batch_size = batch_size,
                        verbose = 1,
                        callbacks = [reduce_lr,early_stop])
                        #callbacks = [csv_logger])
        
        self.trained = trained
        
        return self.model, self.trained
    
    #-------------------------------------------------------------------------------------------------
    #       Trainingsfunktion mit Datasets
    #
    #       Training des Modells über die durch die Übergabe der Verzeichnisse definierten Daten
    #-------------------------------------------------------------------------------------------------        
    def train_cnn_Model(self, train, val,
                        batch_size=2,
                        epochs=30):
        
        self.batch_size = batch_size
        
        #Speichern der Güte während des Trainings.
        #csv_logger = csv_logger('log.csv', append= True, separator=';')
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        trained = self.model.fit(train, 
                                 validation_data = val,
                        epochs = epochs,
                        #steps_per_epoch = steps_per_epoch,
                        #validation_steps = validation_steps,
                        batch_size = batch_size,
                        verbose = 1,
                        callbacks = [reduce_lr,early_stop])
                        #callbacks = [csv_logger])
        
        self.trained = trained
        
        return self.model, self.trained
    
    #-------------------------------------------------------------------------------------------------
    #       TestFunktion
    #
    #       Verwaltet das Testen des Modells
    #-------------------------------------------------------------------------------------------------
    def evaluate_model(self,feature_files):
            

            y_pred =  self.model.predict(feature_files, verbose = 1) 
            

            
            return y_pred
    
    #-------------------------------------------------------------------------------------------------
    #       TestFunktion Datensatz
    #
    #       Verwaltet das Testen des Modells
    #-------------------------------------------------------------------------------------------------
    def evaluate_modelDataset(self,test):
            

            self.model.evaluate(test, verbose = 1) 
            