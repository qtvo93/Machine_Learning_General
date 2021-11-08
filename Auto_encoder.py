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

def compute_model():
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

    plt.figure()
    plt.imshow(y[0])
    plt.colorbar()
    plt.grid(False)
    plt.show()

    #%%

    X=X.reshape(60000,28,28,1)
    X=X/np.max(X)
    y = y.reshape(60000,28,28,1)

    X_train_full,X_test, y_train_full,y_test  = train_test_split(X,y,test_size=0.10,random_state=0)
    X_train,X_valid,y_train,y_valid = train_test_split(X_train_full,y_train_full,test_size=0.2,random_state=0)


    #%%
    input_img = keras.Input(shape=(28, 28, 1))
    def create_model():

        model = keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
        model = keras.layers.MaxPooling2D((2, 2), padding='same')(model )
        model = keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')(model) 
        model = keras.layers.MaxPooling2D((2, 2), padding='same')(model)
        model = keras.layers.Conv2D(4, (3, 3), activation='relu', padding='same')(model)
        encoded = keras.layers.MaxPooling2D((2, 2), padding='same')(model)

        #encode -> decode

        decoder = keras.layers.Conv2D(4, (3, 3), activation='relu', padding='same')(encoded)
        decoder = keras.layers.UpSampling2D((2, 2))(decoder)
        decoder = keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')(decoder)
        decoder = keras.layers.UpSampling2D((2, 2))(decoder)
        decoder = keras.layers.Conv2D(16, (3, 3), activation='relu')(decoder)
        decoder = keras.layers.UpSampling2D((2, 2))(decoder)
        decoded = keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(decoder)

        return decoded

    #%%
    decoded= create_model() 
    autoencoder = keras.Model(input_img,decoded)


    # For Grid Search purposes, please uncomment this block if_def running grid search 

    """
    batch_size = [32, 64, 128]
    optimizer = ['SGD', 'adam','nadam']
    epochs = [50,70,90,100,110,120,130,140,150,160,180,200,210]


    for b in batch_size:
        for o in optimizer:
           for e in epochs:
               autoencoder.compile(optimizer=o, loss = 'binary_crossentropy', metrics =['accuracy'])
               grid_result = autoencoder.fit(X_train, y_train, 
                                             validation_data=(X_valid, y_valid), epochs=e,batch_size = b)
    """
    # end_if Grid Search


    #%%

    checkpoint_cb= tf.keras.callbacks.ModelCheckpoint("Model4.h5", save_best_only=True)
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=5,restore_best_weights=True)
    autoencoder.compile(optimizer='adam', loss = 'binary_crossentropy', metrics =['accuracy'])
    history = autoencoder.fit(X_train, y_train, validation_data=(X_valid, y_valid),batch_size=128,epochs=100, 
                        callbacks=[checkpoint_cb,early_stopping_cb] )


    model = tf.keras.models.load_model("Model4.h5")
    model.summary()

    #%%%

    decoded_imgs = model.predict(X_test)

    def plot(x, p , labels = False): 
         plt.figure(figsize = (20,2)) 
         for i in range(10): 
             plt.subplot(1, 10, i+1) 
             plt.imshow(x[i].reshape(28,28), cmap = 'binary') 
             plt.xticks([]) 
             plt.yticks([]) 
             if labels: 
                 plt.xlabel(np.argmax(p[i])) 
                 plt.show() 
         return 

    plot(decoded_imgs,None)
    """
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = list(range(1,101))

    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

    """
if __name__ == '__main__':
    compute_model()
