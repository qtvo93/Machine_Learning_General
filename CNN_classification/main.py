# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 16:21:12 2021

@author: QTVo
"""

import pickle
import pandas as pd
import numpy as np

# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"


# TensorFlow ≥2.0 is required
import tensorflow as tf
assert tf.__version__ >= "2.0"

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten,MaxPooling2D,Dropout,BatchNormalization

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV

#%%

def compute_model():
    with open('project3trainset.pkl', 'rb') as f:
        df= pickle.load(f)    
    X = df

    with open('project3trainlabel.pkl', 'rb') as f:
        df= pickle.load(f)    
    y = df 

    #%%



    X_train_full, X_test , y_train_full, y_test = train_test_split(X, y,test_size=0.20,random_state=0)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, 
                                                          test_size=0.20,random_state=0)
    #plot the first image in the dataset
    plt.imshow(X_train[0])
    #check image shape
    print(X_train[0].shape)

    #%%
    #reshape data to fit model
    X_train = X_train.reshape(35200,28,28,1)
    X_valid = X_valid.reshape(8800,28,28,1)
    X_test = X_test.reshape(11000,28,28,1)

    np.random.seed(42)
    tf.random.set_seed(42)

    #one-hot encode target column
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    y_valid= to_categorical(y_valid)
    print(y_train[0])

    #%%

    def create_mlp_model(dropout_rate1=0.20, dropout_rate2=0.35):
        #create model
        model = Sequential()
        #add model layers
        model.add(Conv2D(16, kernel_size=3, padding='valid' ,activation='relu', 
                         kernel_initializer='he_normal', input_shape=(28,28,1)))
        model.add(BatchNormalization())
        model.add(Dropout(rate=dropout_rate1))

        model.add(Conv2D(24, kernel_size=3,  padding='valid' ,activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D())    
        model.add(Dropout(rate=dropout_rate1))

        model.add(Conv2D(32, kernel_size=4, strides =  1, padding='valid' ,activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D())    
        model.add(Dropout(rate=dropout_rate1))

        model.add(Conv2D(64, kernel_size=4, strides =  1, padding='valid' ,activation='relu'))
        model.add(BatchNormalization())
        model.add(Flatten())
        model.add(Dropout(rate=dropout_rate2))   

        model.add(Dense(10, activation='softmax'))   
        #compile model using accuracy to measure model performance

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])   
        return model

    # Please uncomment this if_def to run grid search when Grading
    """
    # create model
    model = tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn=create_mlp_model, verbose=1)

    # define parameters and values for grid search 
    param_grid = {
        'batch_size': (16, 32, 64),
        'epochs': (9,11,13,15,17,19,20,21,23,24,25,26,27,28,29,30,31,32,40,50),
        'dropout_rate1': (0.0, 0.10, 0.15, 0.20, 0.25, 0.35, 0.40),
        'dropout_rate2': (0.0, 0.10, 0.15, 0.20, 0.25, 0.35, 0.40),
    }

    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=2)
    grid_result = grid.fit(X_train,y_train, validation_data=(X_valid, y_valid))   
    # print out results
    print(grid_result.best_params_)
    print(grid_result.best_score_)
    """
    #%%

    checkpoint_cb= tf.keras.callbacks.ModelCheckpoint("Model3.h5", save_best_only=True)
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=5,restore_best_weights=True)

    model = tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn=create_mlp_model, 
                                                           verbose=1, batch_size = 64, epochs=28)
    history = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), 
                        callbacks=[checkpoint_cb,early_stopping_cb] )

    #%%
    model = tf.keras.models.load_model("Model3.h5")

    mse_test = model.evaluate(X_test, y_test)

    model.summary()

    y_pred_10  = model.predict(X_test[:10])
    y_pred_10 = y_pred_10.argmax(axis=1)
    print(y_pred_10)
    print(y_test[:10])

    from sklearn.metrics import confusion_matrix
    y_pred= model.predict(X_test)
    cfm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
    print(cfm)

#%%%

if __name__ == '__main__':
    compute_model()
