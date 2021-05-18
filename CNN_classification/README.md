

# Convolutional Neural Network Model for the Classification Task
## Table of Contents:
* [Project Requirements](#i-project-requirements)
* [Project Designs and Developments](#ii-project-designs-and-developments)
* [Discussions](#iii-discussions)
## I. Project Requirements:
In this project you will need to develop a CNN model for the classification task on fashion
MNIST dataset that has been corrupted with noise. The dataset is provided (“project-trainset.pkl” for training and project-trainlabel.pkl” for the labels). 
You will need to
develop the CNN as follows:
1. Develop a multilayered CNN classifier for classification of the fashion MNIST dataset.
While the textbook demonstrates a CNN model for doing just that, it involved training
over 1.4 million parameters. The goal of this project is to design a CNN network such
that it delivers reasonably high accuracy with weight parameters less than 50,000.
2. As usual, conduct validation tests on the model to ensure that it is not under/overfitting.
3. Provide a performance prediction of your model against the test set. The ability to
predict how well your model will do on unseen data is an important element in machine
learning model development
 
 
 
 
## II. Project Designs and Developments:
The project implementations started with loading the data set with pickle.load and performing some quick inspections on the given data as usual. The data set appeared to be clean without “NaN” data; therefore, no data cleaning was required. The data then was split into training, valid, and testing set. 
A quick inspection showed that the X_train[0] data has a shape of (28,28). Therefore, X_train, X_valid, and X_test needed to be reshaped in order to fit the training model. For the label set- y, one-hot encoder was used to categorize the label data in order to be used for training. A quick print out of y_train[0]  showed that y_train[0] was an array of 0,1: [0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]; which was what the implementations intended to encode. The example array indicated that the first digit label was 7 if we use argmax(axis=1) to translate the array.

![image](https://user-images.githubusercontent.com/72519491/118713273-b9f77080-b7ef-11eb-84d6-981c78057667.png)

After all the data processing had been done, the data set was ready to be trained and validated. The project development state moved to create a CNN model in order to be trained and perform prediction.
A function called create_mlp_model was used to generate the model for the Keras Classifier to use later. It was required that the parameter that passed into the Keras Classifier was a function to build model, not a pre-built model.
The implementations were played around with changes vary from optimizer, loss method to number of layers, filter size, kernel size, etc in order to get some senses of how the reduction in parameters would affect the overall implementation. Because the optimizer was picked as ‘adam’, loss method was picked as ‘categorical_crossentropy’ and number of layers was designed based on the overall outcomes of initial running trials, the parameters of the create_mlp_model only contained dropout rate 1 and dropout rate 2 that would be used in later GridSearch. 
The grid search parameters were generated as following:

param_grid = {

    'batch_size': (16, 32, 64),
    
    'epochs': (9,11,13,15,17,19,20,21,23,24,25,26,27,29,30,31,32,40,50),
    
    'dropout_rate1': (0.0, 0.10, 0.15, 0.20, 0.25, 0.35, 0.40),
    
    'dropout_rate2': (0.0, 0.10, 0.15, 0.20, 0.25, 0.35, 0.40), }


The Grid Search focused on searching the best dropout rate parameters for the create_mlp_model function and the most efficient batch size and epochs.
A model was also created to perform the grid search:
model = tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn=create_mlp_model, verbose=1)
The generated model was satisfied the requirements that total number of parameters are less than 50,000. The optimized model has the best fine-tuned parameters in the parameter lists with batch size = 64, epochs = 28, dropout rate 1 = 0.20, dropout rate 2 = 0.35

![image](https://user-images.githubusercontent.com/72519491/118713319-c7acf600-b7ef-11eb-9b49-edc1a41973a3.png)


Taking a deeper look into each layer, the non-trainable parameters number was from each layer batch normalization which was common and as intended in order to not overfit the model. These non-trainable parameters corresponded to the computed mean and standard deviation of the activations that was used during test time, and these parameters would never be trainable using gradient descent.
 
![image](https://user-images.githubusercontent.com/72519491/118713352-d09dc780-b7ef-11eb-8589-a1ddbb2eeee5.png)


The main methodologies were used to reduce the total number of parameters were using striding kernel with stride size = 1 and MaxPooling2D with the pool size = 2 as default.
In addition, each layer would also be normalized using BatchNormalization and then applied Dropout in order to prevent overfitting.
After the GridSearch was conducted and recorded, the code was then updated to create a model with the best tuned parameters. The model was then trained and saved to Model3.h5 using Keras.
In order to generate the Model3.h5 file, the checkpoint and early stopping methodologies were used in order to have an early stop when necessary after no progress on validation set for 5 consecutive number of epochs and to save the best available model.
After the model was loaded back from the Model.h5 file, a quick prediction was performed on the first 10 elements of the X_test to see how the design performed. In comparison to the first 10 elements of the label y_test. The designed model got 10/10 correct answers, indicating that its performance was outstanding. It raised the question that whether the model was overfitting.

![image](https://user-images.githubusercontent.com/72519491/118713393-deebe380-b7ef-11eb-84d1-c75a835e816d.png)

 

Looking back into the training process, the training accuracy was not lower than the validation accuracy; therefore, the model was not underfitting. The training loss has gone down and the validation loss also decreased while training accuracy is higher than the validation accuracy, indicating that the model seemed like not overfitting. To address this matter, an earlier design was implemented to ensure that the model would not drastically overfitting by using BatchNormalization and applied drop rate after input layer and hidden layers before passing parameters into the output layer. The model seemed to converge as the validation loss converged at around 0.3621.

![image](https://user-images.githubusercontent.com/72519491/118713459-f4f9a400-b7ef-11eb-8a09-ee233d47705b.png)

 
Model.evaluation was also used to evaluate the model’s performance.
mse_test = model.evaluate(X_test, y_test)
 
![image](https://user-images.githubusercontent.com/72519491/118713481-f9be5800-b7ef-11eb-8f3a-fe3ff29a97a8.png)

The test accuracy was lower than the validation accuracy (0.8649 < 0.8670) and it was expected.
Finally, after some necessary evaluations on the model, it seemed like the designed model was ready to go and the final test was performed. With the performance of 86.49% accuracy on the test set. My estimation for the model’s performance on a freshly new test set would be around 83.50%.
The confusion matrix was generated as well to evaluate the model’s performance on the test set.

![image](https://user-images.githubusercontent.com/72519491/118713502-004ccf80-b7f0-11eb-9b8b-a20e4eaa85c6.png)


From the above confusion matrix, it can be learned that the number 6 was the digit with the most misprediction (lowest correct classified score). In addition, it got misclassified as number 0 the most. This made sense since some handwritten number 6 looked very similar to 0.

## III. Discussions:
The model’s performance with accuracy was somewhat acceptable; however, there was still room for improvements. 
Given that the parameters were bounded by 50,000; if the total number of parameters were not limited, the model can be trained with a higher cap of accuracy because the difference in the number of combinations of CNN layers. In addition, the grid search was not performed precisely among all the available parameters. If the grid search were conducted more precise, the model parameters could be tuned finer and therefore, would likely generate a better prediction accuracy. 
