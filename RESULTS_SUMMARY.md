# Results summary

## Model architecture

We chose to use the UNet-1d from url. 
We split the time recordings into recordings of duration 1min. 

## Hyperparameters choice

We run the model with a batch size of x, y epochs, bla optimizer and blou loss. 
These hyperparameters peformed best amongst the following choices:
LEARNING_RATE = [0.001,0.0001]
LOSSES = [keras.losses.BinaryCrossentropy(), 
          keras.losses.BinaryFocalCrossentropy(),
          keras.losses.KLDivergence()
          ]
OPTIMIZERS = [keras.optimizers.Adam(learning_rate=LEARNING_RATE),
             keras.optimizers.SGD(learning_rate=LEARNING_RATE)]
BATCH_SIZES = [16, 32, 64]
EPOCHS = [30]

## Model training

You can find below the loss and accuracy plots during the model fit. 
No obvious overfitting can be seen and validation accuracy is quite low (bla), we are therefore 
satisfied with the training. 

![alt text](https://github.com/[username]/[reponame]/blob/[branch]/image.jpg?raw=true)

## Results 

Here are the results obtained on the testing set with different metrics:

| F1 score      | Accuracy      | Balanced accuracy | Precision | Recall |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Content Cell  | Content Cell  | Content Cell  | Content Cell  | Content Cell  |

We can notice that. 

From the confusion matrix, we can notice that we have few false, blabla

 ROC curve shows good results, with area under the curve being . 