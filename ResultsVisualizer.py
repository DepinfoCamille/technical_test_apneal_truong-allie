import keras
import matplotlib.pyplot as plt
from scikitplot.metrics import plot_confusion_matrix, plot_roc
from sklearn import metrics
import datetime
import numpy as np
import os 
# ---------------------------------------------------------------------
#  Utility functions
# ---------------------------------------------------------------------

def try_save_figure(folder_path, figure_name, figure):

    if folder_path is not None:   
        try: 
            figure.savefig(os.path.join(folder_path, figure_name))
        except OSError:
            print("Error when trying to save figure. Save folder does not exist")


# ---------------------------------------------------------------------
#  ResultsVisualizer class
# ---------------------------------------------------------------------


class ResultsVisualizer():

    def __init__(self, model, X_test, y_test, y_pred_threshold = 0.8):

        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        
        ## Get model predictions for X_test
        self.proba  = self.model.predict(self.X_test)
        self.y_pred = np.zeros(self.proba.shape)
        self.y_pred[np.where(self.proba>y_pred_threshold)] = 1

        ## Compute flattened arrays so we do not have to do it multiple times
        self._y_test_flattened = self.y_test.flatten()
        self._proba_flattened = self.proba.flatten()
        self._y_pred_flattened = self.y_pred.flatten()

    def compute_metrics(self):
        
        print("F1 score", metrics.f1_score(self._y_test_flattened, self._y_pred_flattened))
        print("Accuracy score", metrics.accuracy_score(self._y_test_flattened, self._y_pred_flattened))
        print("Precision score", metrics.precision_score(self._y_test_flattened, self._y_pred_flattened))
        print("Recall score", metrics.recall_score(self._y_test_flattened, self._y_pred_flattened))

    def display_figures(self, save_folder = None):

        # Plot and save confusion matrix
        fig, ax = plt.subplots(figsize=(16,12))
        plot_confusion_matrix(self._y_test_flattened, self._y_pred_flattened, ax=ax)
        if save_folder is not None:
            try_save_figure(save_folder, "confusion_matrix.pdf", fig)

        # Plot and save roc curve
        fig, ax = plt.subplots(figsize=(16,12))
        fpr, tpr, thresholds = metrics.roc_curve(self._y_test_flattened, self._y_pred_flattened)
        roc_auc = metrics.auc(fpr, tpr)
        display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
        display.plot(ax=ax)
        if save_folder is not None:
            try_save_figure(save_folder, "roc_curve.pdf", fig)


    def plot_data(self, X_row_id):
        """ Plot the features, groundtruth (red) and predictions (blue) at the given row index
        """

        assert X_row_id < self.X_test.shape[0]

        fig, axes = plt.subplots(nrows = self.X_test.shape[-1]+2, sharex=True, figsize=(30,12))

        ## Plot the features
        features = [self.X_test[X_row_id, :, i] for i in range(self.X_test.shape[-1])]
        for ax, feat in zip(axes[:-2], features):
            ax.plot(feat)

        ## Plot the groundtruth (red) and the model's predictions (blue)
        axes[-2].plot(self.y_test[X_row_id, :], "r")  
        axes[-1].plot(self.y_pred[X_row_id, :], "b")  
