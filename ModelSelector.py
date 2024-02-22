import os
import numpy as np
import json
from glob import glob
import pickle

import matplotlib.pyplot as plt

class ModelSelector():

    """ Read all models history, from it get_best_model returns the best model, saved in checkpoint folder

    """
    def __init__(self, history_folder_path):

        assert os.path.exists(history_folder_path)

        self.history_folder_path = history_folder_path
        self.history_paths = []
        self.histories = []

        for path in glob(os.path.join(history_folder_path, "*.json")):

            self.history_paths.append(path)
            with open(path, "r") as f:
                history = json.load(f)
                self.histories.append(history)
                print(history.keys())

        self._compute_best_metrics()

    def _compute_best_metrics(self):
        self.best_losses = np.array([np.max(h["val_loss"]) for h in self.histories])
        self.best_accuracies= np.array([np.max(h["val_binary_accuracy"]) for h in self.histories])

    def get_best_model_name(self):

        history_path = self.history_paths[np.argmax(self.best_accuracies)]
        # model path is the same as history path, 
        # except it does not start with "history" nor end with ".json"
        return os.path.basename(history_path)[8:-5] 


    def get_best_model(self):

        model_name = self.get_best_model_name()
        model_path = os.path.join(self.history_folder_path, "checkpoint_" + model_name, "saved_model.pb")
        print(model_path)

        with open(model_path, "rb") as f:
            return pickle.load(f)

    def plot_history(self, index):

        assert index >= 0 and index < len(self.histories)

        print(self.histories[index]["loss"])

        plt.plot(self.histories[index]["loss"], "b")
        plt.plot(self.histories[index]["val_loss"], "orange")
        plt.show()

