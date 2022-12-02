# -*- coding: utf-8 -*-
"""
Created on Fri Feb  19 14:07:39 2021

@author: QTVo
"""

import pickle
import pandas as pd
import numpy as np

"""
def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)
"""

with open('mnist_X_train.pkl', 'rb') as f:
    df= pickle.load(f)    
X = df

with open('mnist_y_train.pkl', 'rb') as f:
    df= pickle.load(f)    
y = df 

#%%
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"

# to make this notebook's output stable across runs
np.random.seed(42)

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier


#%%

import numpy as np

X_train, X_test , y_train, y_test = train_test_split(X, y,test_size=0.25,random_state=0)


#%%
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#%%
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression

#%% KNN Classifier

pipe = Pipeline([
        ('sc', StandardScaler()),     
        ('knn', KNeighborsClassifier(algorithm='brute')) 
    ])
params = {
        'knn__n_neighbors': [1,11,13,15,17,57,59,209,211,213],
        'knn__leaf_size': [40,50],
        'knn__p': [6,7]}
        
neigh_clf = GridSearchCV(estimator=pipe,           
                      param_grid=params, 
                      cv=5) 
neigh_clf.fit(X_train, y_train)
#print(neigh_clf.best_params_)

#%% RandomForest
pipe = Pipeline([
        ('sc', StandardScaler()),     
        ('rnd', RandomForestClassifier(random_state=42)) 
    ])

param_grid = { 
    'rnd__n_estimators': [400, 500],
    'rnd__max_features': ['auto', 'sqrt', 'log2',1,2,4,6,7,8],
    'rnd__max_depth' : [None,1,2,3,4,5,6,7],
    'rnd__criterion' :['gini', 'entropy'],
    'rnd_bootstrap': [True, False],
    'rnd_max_leaf_nodes': [100,200,300,400]}
rnd_clf = GridSearchCV(estimator=pipe, param_grid=param_grid, cv= 5)
rnd_clf.fit(X_train, y_train)
#print(rnd_clf.best_params_)

#%% SVC
pipe = Pipeline([
        ('sc', StandardScaler()),     
        ('svc', SVC(gamma="scale",random_state=42)) ])

param_grid={'svc__C': [1, 10], 'svc__kernel': ('linear', 'rbf')}

svc_clf = GridSearchCV(estimator=pipe,param_grid=param_grid)
svc_clf.fit(X_train, y_train)
#print(svc_clf.best_params_)

#%% Final Classifiers with tuned parameters and the essemble model
neigh_clf = KNeighborsClassifier(n_neighbors = 13, leaf_size = 35, p = 2 ,algorithm='brute')
rnd_clf = RandomForestClassifier(bootstrap=True, criterion='gini', max_features=6, 
                                 min_samples_leaf =1, min_samples_split=2,n_estimators=100)
svc_clf = SVC(C=1, kernel = 'rbf',gamma='scale',random_state=42)

voting_clf = VotingClassifier(
    estimators=[('rf', rnd_clf), ('KNN', neigh_clf),('svc', svc_clf)],
    voting='hard')

for clf in (rnd_clf, neigh_clf, svc_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))

#%% Confustion matrix and pickle models
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)

filename = 'model2.pkl'
pickle.dump(voting_clf, open(filename, 'wb'))
filename = 'RandomForest.pkl'
pickle.dump(rnd_clf, open(filename, 'wb'))
filename = 'SVC.pkl'
pickle.dump(svc_clf, open(filename, 'wb'))
filename = 'Kneighbors.pkl'
pickle.dump(neigh_clf, open(filename, 'wb'))