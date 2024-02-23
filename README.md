# technical_test_apneal_truong-allie

## Code overview

*build_models.ipynb* builds multiple models from different hyperparameters.
For this, it uses the class DataLoader from *DataLoader.py* to shape the raw data into numpy arrays for training 
(with train_test_split, in which we can decide the duration of epochs)

*visualize_best_model.ipynb* selects the best model across all the ones built and saved with *build_models.ipynb*, with an instance of the class ModelSelector. This class also enables to display the loss and accuracy across the training. 
Then ResultsVisualizer enables to compute relevant metrics, ROC curve, and segments of the data with the predicted and actual values. 

*Report.pdf* presents the results and future work that I would have made if I had the time. 

## Issues

I was not able to load the models that I built. 
Consequently, I do not have results to present...
I hope the clean and structured code that I provide will still be of interest to you!


The model in the *models* folder and its checkpoints are dummies. 