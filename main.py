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
GATmodel = GATGNNModel()
GATmodel.train(data)
gat_pred = GATmodel.preds

#%% Graph convolutional neural network
GCN = GCNModel(dataset.train_loader,dataset.test_loader)
GCN.fit()
gcn_pred = GCN.pred
#%% LSTM
LSTM = LSTMModel(X_train_stack = dataset.X_train_stack,
                 y_train_stack=dataset.y_train_stack, 
                 X_test_stack=dataset.X_test_stack, 
                 y_test_stack=dataset.y_test_stack)
LSTM.train()
lstm_pred = LSTM.pred
#%% CNN
CNN = CNNModel(X_train_stack = dataset.X_train_stack,
                 y_train_stack=dataset.y_train_stack, 
                 X_test_stack=dataset.X_test_stack, 
                 y_test_stack=dataset.y_test_stack)
CNN.train()
cnn_pred = CNN.pred
#%% Classifiers XGBOOST,DECISION TREE, RANDOM FOREST, KNN, SVM
Others = Classifiers(X_train_np=dataset.X_train_np, y_train=dataset.y_train, X_test_np=dataset.X_test_np, y_test=dataset.y_test)
Others.train()
xg_pred = Others.xg_pred
rf_pred = Others.clrf_pred
knn_pred = Others.knn_pred
svc_pred = Others.svc_pred
dt_pred = Others.dt_pred
#%% Inspect 1st sample
inspecting_sample = 1
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
#%% Results and confusion matrixes
#GAT results
from sklearn import metrics
import matplotlib.pyplot as plt
def display_statistic(model_name,preds):   
    expected_y = dataset.y_test
    print(metrics.classification_report(expected_y, preds))
    cm = metrics.confusion_matrix(expected_y, preds)
    print(model_name," CLASSIFIER ACCURACY: ",metrics.accuracy_score(expected_y,preds))
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()
#%%
display_statistic('XGBoost', xg_pred)
#%%
display_statistic('Decision Tree', dt_pred)
#%%
display_statistic('RANDOM FOREST',rf_pred)
#%%
display_statistic('KNN',knn_pred)
#%%
display_statistic('SVC',svc_pred)
#%%
lstm_pred_format = [int(round(i,0)) for i in lstm_pred.flatten()]
display_statistic('LSTM',lstm_pred_format)
#%%
display_statistic('GAT',gat_pred)
#%%
gcn_pred_format = []
for i in gcn_pred:
    for j in i:
        gcn_pred_format.append(j)
display_statistic('GCN',gcn_pred_format)
#%%
display_statistic('CNN', [int(round(i,0)) for i in cnn_pred.flatten()] )