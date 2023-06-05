# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 20:28:45 2023

@author: bvtp1
"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

class LSTMModel:
    def __init__(self,X_train_stack,y_train_stack, X_test_stack,y_test_stack):
        self.X_test_stack = X_test_stack
        self.y_train_stack = y_train_stack
        self.y_test_stack = y_test_stack
        self.X_train_stack = X_train_stack
        self.pred = []
    def train(self):
        lstm_model = Sequential()
        lstm_model.add(LSTM(100,return_sequences=True))
        lstm_model.add(Dense(1, activation='sigmoid'))
        lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        lstm_model.fit(self.X_train_stack,self.y_train_stack)
        lstm_scores = lstm_model.evaluate(self.X_test_stack, self.y_test_stack, verbose=0)
        self.pred = lstm_model.predict(self.X_test_stack)
        print(lstm_model.summary())
        print("LSTM Accuracy: %.2f%%" % (lstm_scores[1]*100))