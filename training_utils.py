import keras
import os
import joblib
import json

def create_checkpoint_callback(checkpoint_folder):

    if not os.path.exists(checkpoint_folder):
      os.mkdir(checkpoint_folder)

    return keras.callbacks.ModelCheckpoint(filepath=checkpoint_folder,
                                                                        monitor='val_binary_accuracy',
                                                                        mode='max',
                                                                        save_best_only=True)

def get_saving_paths(root, batch_size=16, 
                          epochs=1, 
                          epoch_duration=10, 
                          optimizer_name="adam", 
                          loss_name="binary_crossentropy"):

      model_name = "batch_{}_epoch_{}_duration_{}_optim_{}_loss_{}".format(batch_size, 
                                                                                  epochs, 
                                                                                  epoch_duration,           
                                                                                  optimizer_name, 
                                                                                  loss_name)
      
      history_filename = "history_" + model_name + ".json"
      local_filename = model_name + ".pb"
      checkpoints_folder = "checkpoints_" + model_name

      model_filename = os.path.join(root, local_filename)
      history_filename = os.path.join(root, history_filename)
      checkpoint_folder = os.path.join(root, model_name)

      return model_filename, history_filename, checkpoint_folder


def build_train_save_model(X_train, y_train, save_folder_path,
                          batch_size=16, 
                          epochs=1, 
                          epoch_duration=10, 
                          optimizer=keras.optimizers.Adam(learning_rate=0.001), 
                          loss= keras.losses.BinaryCrossentropy()):

    # Paths to save the model, its history, and its checkpoints
    model_filename, history_filename, checkpoint_folder = get_saving_paths(save_folder_path, 
                                                                            batch_size=batch_size, 
                                                                            epochs=epochs, 
                                                                            epoch_duration=epoch_duration, 
                                                                            optimizer_name=optimizer.name, 
                                                                            loss_name=loss.name)
    if not os.path.exists(save_folder_path):
      os.mkdir(save_folder_path)

    if not os.path.exists(model_filename):

        # Create model
        model = Unet1D(backbone_name='resnet18_1d',
                      input_shape = (None, 11)
                      )
        model.compile(
                        optimizer=optimizer,
                        loss=loss,
                        metrics=[
                            keras.metrics.BinaryAccuracy(),
                            keras.metrics.FalseNegatives()],
                        )

        # Fit model
        checkpoint_callback = create_checkpoint_callback(checkpoint_folder)
        history = model.fit(X_train, y_train, verbose = 1,
                        callbacks = checkpoint_callback,
                        batch_size = batch_size,
                        epochs = epochs,
                        validation_split=0.2)

        # Save 
        with open(model_filename, 'wb') as file_pi:
              joblib.dump(model, file_pi)

        with open(history_filename, 'w') as file_pi:
              json.dump(history.history, file_pi)
        print(history.history)
