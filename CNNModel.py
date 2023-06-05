# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 20:34:32 2023

@author: bvtp1
"""
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten,Conv1D
from tensorflow.keras.models import Sequential
class CNNModel:
    def __init__(self,X_train_stack,X_test_stack,y_train_stack,y_test_stack):
        self.X_train_stack=X_train_stack
        self.X_test_stack=X_test_stack
        self.y_train_stack=y_train_stack
        self.y_test_stack=y_test_stack
        self.pred = []
    def train(self):
        n_steps = 300
        n_features = 1
        cnn_model = Sequential()
        cnn_model.add(Conv1D(filters=64, kernel_size=1, activation='relu', input_shape=(n_features,n_steps)))
        cnn_model.add(Flatten())
        cnn_model.add(Dense(50, activation='relu'))
        cnn_model.add(Dense(1))
        cnn_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=['binary_accuracy'],
        )
        cnn_model.compile(
            optimizer=tf.keras.optimizers.Adam(epsilon=0.01),
            loss='binary_crossentropy',
            metrics=['binary_accuracy'],
        )        
        cnn_model.fit(self.X_train_stack, self.y_train_stack, epochs=10, verbose=0)
        # cnn_scores = cnn_model.evaluate(self.X_test_stack, self.y_test_stack, verbose=0)
        self.pred = cnn_model.predict(self.X_test_stack)
        # print(cnn_model.summary())
        # print("CNN Accuracy: %.2f%%" % (cnn_scores*100))
