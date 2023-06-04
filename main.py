# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 19:13:26 2023

@author: bvtp1
"""

import os
from dataPreprocessing import dataBuilder
from GAT import GATGNNModel
from GCNModel import GCNModel
from LSTMModel import LSTMModel
from CNNModel import CNNModel
from Classifiers import Classifiers
from graphAnalysis import network_analysis
#%% Create dataset object with current working directory
directory= os.getcwd().replace('\\','/')
print("Working directory: ", directory)
dataset = dataBuilder(directory)
#%% Preprocess data, '15' for Twitter15
dataset.create_dataset('16')
#%% Load saved data
dataset.load()
#%% Split 7-3 train-test
dataset.split_train_test(0.7)
#%% Retrieve graphs
data = dataset.data
#%% GAT GNN
GATmodel = GATGNNModel().train(data)
#%% Graph convolutional neural network
GCN = GCNModel(dataset.train_loader,dataset.test_loader).fit()
#%% LSTM
LSTM = LSTMModel(X_train_stack = dataset.X_train_stack,
                 y_train_stack=dataset.y_train_stack, 
                 X_test_stack=dataset.X_test_stack, 
                 y_test_stack=dataset.y_test_stack).train()
#%% CNN
CNN = CNNModel(X_train_stack = dataset.X_train_stack,
                 y_train_stack=dataset.y_train_stack, 
                 X_test_stack=dataset.X_test_stack, 
                 y_test_stack=dataset.y_test_stack).train()
#%% Classifiers XGBOOST,DECISION TREE, RANDOM FOREST, KNN, SVM
Others = Classifiers(X_train_np=dataset.X_train_np, y_train=dataset.y_train, X_test_np=dataset.X_test_np, y_test=dataset.y_test).train()
#%% Inspect 1st sample
inspecting_sample = 0
na_sample = network_analysis(data[inspecting_sample])
na_sample.draw_network()
cc = na_sample.closeness_centrality()
bc = na_sample.betweenness_centrality()
ec = na_sample.eigenvector_centrality()
dc = na_sample.degree_centrality()
import networkx as nx
cnc = nx.common_neighbor_centrality(network_analysis(data[inspecting_sample]).G)
for u, v, p in cnc:
    print(f"({u}, {v}) -> {p}")