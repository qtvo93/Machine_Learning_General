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
