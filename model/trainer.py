def train_cnn_Model(model, feature_files, label_files,
                        feature_files_val, label_files_val,
                        batch_size=2,
                        epochs=30):
        
        
        #Speichern der Güte während des Trainings.
        #csv_logger = csv_logger('log.csv', append= True, separator=';')
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        callbacks = [
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
            EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True),
            CSVLogger("log.csv", append=True, separator=";"),
            tf.keras.callbacks.TensorBoard(log_dir="./logs", histogram_freq=1),
        ]
        trained = model.fit(feature_files, label_files,
                                 validation_data = (feature_files_val, label_files_val),
                        epochs = epochs,
                        #steps_per_epoch = steps_per_epoch,
                        #validation_steps = validation_steps,
                        batch_size = batch_size,
                        verbose = 1,
                        callbacks = callbacks)#,csv_logger])
                        #callbacks = [csv_logger])
        
        
        return model, trained

def trainModel(config_path: str = "config/config.json"):

    cfg = load_config(config_path)
    #-------------------------- Random Parameter initialization ---------------------

    set_random_seed(cfg["RANDOM_SEED"])

    #--------------------------------------------------------------------------------
    X_train, X_test, y_train, y_test, X_val, y_val = prepare_data(cfg)
    model_instance = CBLSTM() # create an instance of the model class
    model_path = get_model_path(cfg)

    #--------------------------------------------------------------------------------

    
    if cfg["TRAIN_NEW_MODEL"]:
    # Modell erstellen
        model = model_instance.create_model(
            learning_rate=cfg["LEARNING_RATE"],
            filters=[32, 64, 128],
            num_of_conv_Layers=cfg["NUM_OF_CONV_LAYERS"],
            lstm_units=cfg["LSTM_UNITS"],
            lstm_layers=cfg["LSTM_LAYERS"],
            dropout_rate=cfg["DROPOUT"],
        )

        callbacks = [
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
            EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True),
            CSVLogger("log.csv", append=True, separator=";"),
            tf.keras.callbacks.TensorBoard(log_dir="./logs", histogram_freq=1),
        ]
        # Training mit den gewählten Parametern
        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=cfg["BATCH_SIZE"],
            epochs=cfg["EPOCHS"],
            verbose=1,
            callbacks=callbacks,
        )
        
        
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model.save(model_path)
        typer.echo(f"Modell gespeichert unter: {model_path}")
        display_train_val_loss()
    else:

        typer.echo("Lade existierendes Modell ...")
        model = tf.keras.models.load_model(model_path)