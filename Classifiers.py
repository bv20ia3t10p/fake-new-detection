# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 20:39:43 2023

@author: bvtp1
"""
import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
class Classifiers:
    def __init__(self,X_train_np,y_train,X_test_np,y_test):
        self.X_train_np = X_train_np
        self.X_test_np = X_test_np
        self.y_train = y_train
        self.y_test = y_test
        self.xg_pred = []
        self.clrf_pred = []
        self.svc_pred = []
        self.knn_pred = []
        self.dt_pred=[]
    def train(self):
        X_train_np = self.X_train_np
        X_test_np = self.X_test_np
        y_train= self.y_train
        y_test= self.y_test
        clf = DecisionTreeClassifier()
        clf = clf.fit(X_train_np,y_train)
        clf_pred = clf.predict(X_test_np)
        self.dt_pred= clf_pred
        print("Decision Tree Classifier Accuracy:",metrics.accuracy_score(y_test, clf_pred))
        #%%
        
        # creating a RF classifier
        clrf = RandomForestClassifier(n_estimators = 100)  
          
        # Training the model on the training dataset
        # fit function is used to train the model using the training sets as parameters
        clrf.fit(X_train_np, y_train)
          
        # performing predictions on the test dataset
        y_pred = clrf.predict(X_test_np)
        self.clrf_pred=y_pred
        print("ACCURACY OF THE RF MODEL: ", metrics.accuracy_score(y_test, y_pred))
        #%%
        clsv = SVC()  
        clsv.fit(X_train_np, y_train)
        y_pred = clsv.predict(X_test_np)
        self.svc_pred=y_pred
        print("ACCURACY OF THE SVM MODEL: ", metrics.accuracy_score(y_test, y_pred))
        #%%
        clknn = KNeighborsClassifier(n_neighbors=30)
        clknn.fit(X_train_np, y_train)
        y_pred = clknn.predict(X_test_np)
        self.knn_pred=y_pred
        print("ACCURACY OF THE KNN MODEL: ", metrics.accuracy_score(y_test, y_pred))
        #%%
        model = xgb.XGBClassifier()
        model.fit(X_train_np, y_train)
        expected_y  = y_test
        predicted_y = model.predict(X_test_np)
        self.xg_pred=predicted_y
        # print(metrics.classification_report(expected_y, predicted_y))
        # print(metrics.confusion_matrix(expected_y, predicted_y))
        print("XGBOOST CLASSIFIER ACCURACY: ",metrics.accuracy_score(expected_y,predicted_y))