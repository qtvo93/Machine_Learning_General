# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 22:57:58 2021
@author: QTVo
"""
import pickle
import numpy as np
import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"

# TensorFlow ≥2.0 is required
import tensorflow as tf
from tensorflow import keras
assert tf.__version__ >= "2.0"

# Common imports

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten,MaxPooling2D,Dropout,BatchNormalization,Conv2DTranspose

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV


#%%

with open('Xtrain', 'rb') as f:
    df= pickle.load(f)    
X = df

with open('Ytrain', 'rb') as f:
    df= pickle.load(f)    
y = df 

np.random.seed(42)
tf.random.set_seed(42)
#%%

plt.figure()
plt.imshow(X[0])
plt.colorbar()
plt.grid(False)
plt.show()