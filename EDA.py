# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 16:23:31 2022

@author: user
"""
 
import os
import random
import csv
import time
import random
from random import shuffle, choice, sample, uniform
from operator import add, attrgetter
from tqdm import tqdm
from functools import reduce
from copy import deepcopy
import pandas as pd
import numpy as np
import tensorflow as tf
import time
from keras.models import Sequential
from keras.layers import Dense, Dropout, InputLayer
from keras import backend as K

import numpy as np
import random as python_random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt
import keras.backend as K

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss

from keras.models import Sequential
from keras import layers
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.utils import class_weight

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn import metrics
from sklearn.metrics import classification_report, roc_curve, auc

import shap

from operator import add
import IPython

# Defining the seed
SEED = 2022

# Define Game Commands
RIGHT_CMD = [0, 1]
LEFT_CMD = [1, 0]

# Define Reward Config
START_REWARD = 0
MIN_REWARD = 250

# Define Hyperparameters for NN
HIDDEN_LAYER_COUNT = [1, 2, 3, 4, 5]
HIDDEN_LAYER_NEURONS = [8, 16, 24, 32, 64, 128, 256, 512]
HIDDEN_LAYER_RATE = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] 
HIDDEN_LAYER_ACTIVATIONS = ['tanh', 'relu', "sigmoid", "softplus"]
HIDDEN_LAYER_TYPE = ['dense', 'dropout']
MODEL_OPTIMIZER = ['adam', "rmsprop"] # rmsprop
# NOT USED YET
MODEL_LOSS=[tf.keras.losses.BinaryCrossentropy(), 
                  tf.keras.losses.BinaryFocalCrossentropy(gamma = 3.0)]


# Define Genetic Algorithm Parameters
# Max Number of Generations to Apply the Genetic Algorithm
GENERATIONS = 2 
# Max Number of Individuals in Each Population
SIZE = 10  
# Number of Best Candidates to Use 
ELITE_SIZE = 2  
# 
CROSSOVER_PROBABILITY = 0.7
MUTATION_PROBABILITY = 0.0
#
OPTIM = "max"
#
PATIENCE=5
#
EPOCHS=100
#
THRESHOLD=0.5

def oversample(ratio, x_train, y_train):
    sm = SMOTE(random_state=SEED, sampling_strategy=ratio)
    return sm.fit_resample(x_train, y_train)


def undersample(ratio, x_train, y_train, version):
    sm = NearMiss(version=version, sampling_strategy=ratio)
    return sm.fit_resample(x_train, y_train)


def remove_correlated(X, threshold=0.9):
    corr = np.absolute(X.corr())
    i = 1
    for (index, row) in corr.iterrows():
        for col in corr.columns[i:]:
            if row[col] > 0.9:
                print(f"{index} vs. {col} are highly correlated {row[col]}")
        i += 1
        # Select upper triangle of correlation matrix
    corr_matrix = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))
    # Find features with correlation greater than threshold
    correlated_cols = [c for c in corr_matrix.columns if any(corr_matrix[c] > threshold)]
    # Drop correlated columns
    return correlated_cols


def remove_outliers(x_train, x_test, y_train, y_test, cols, estimators=100):
    ''' 
    Uses an isolation forest in order to detect and remove outliers.
    '''
    isolation_forest = IsolationForest(n_estimators=estimators, random_state=SEED)
    
    #identify outliers:
    y_train_pred = isolation_forest.fit_predict(x_train[cols])
    y_test_pred = isolation_forest.predict(x_test[cols])

    #Remove outliers where 1 represent inliers and -1 represent outliers:
    x_train = x_train[np.where(y_train_pred == 1, True, False)]
    x_test = x_test[np.where(y_test_pred == 1, True, False)]

    y_train = y_train[np.where(y_train_pred == 1, True, False)]
    y_test = y_test[np.where(y_test_pred == 1, True, False)] 
    
    return x_train, x_test, y_train, y_test

def f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)

ds = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
ds.head()

ds['TotalCharges'] = pd.to_numeric(ds['TotalCharges'], errors='coerce')

ds.dropna(inplace=True)

# Droping the column 'CustomerID' since it is not useful:
ds.drop(columns=['customerID'], inplace=True)

ds["TotalCharges"] = ds["TotalCharges"].astype(float)

ds["Churn"].value_counts(normalize=True)

ds["Churn"] = np.where(ds["Churn"] == 'Yes', 1, 0)
X = ds.copy(deep=True)

y = X["Churn"]
X = X.drop(columns=["Churn"])

# Split the dataset intro train and test sets.
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    shuffle=True, stratify=y, random_state=SEED)

correlated_cols = remove_correlated(x_train)
x_train = x_train.drop(columns=correlated_cols)
x_test = x_test.drop(columns=correlated_cols)

print(x_train.shape)
print(x_test.shape)

num_features = ["tenure", "MonthlyCharges", "TotalCharges"]
x_train, x_test, y_train, y_test = remove_outliers(
    x_train, x_test, y_train, y_test, num_features)

print(x_train.shape)
print(x_test.shape)

print(y_train.value_counts(normalize=True))
print(y_test.value_counts(normalize=True))

scaler = StandardScaler()
x_train[num_features] = scaler.fit_transform(x_train[num_features])
x_test[num_features] = scaler.transform(x_test[num_features])

cat_features = list(set(x_train.columns).difference(
    num_features).difference(["SeniorCitizen"]))

# I guess it is useless, will check it later.
for col in cat_features:
    x_train[col] = x_train[col].astype('category')
    x_test[col] = x_test[col].astype('category')

for col in cat_features:
    print(x_train[col].unique())

# We can perform one-hot encoding for categorical features:
encoder = OneHotEncoder(drop="if_binary", sparse=False)
x_train_cat = encoder.fit_transform(x_train[cat_features])
x_test_cat = encoder.transform(x_test[cat_features])

x_train = np.concatenate((x_train_cat, x_train[num_features].values), axis=1)
x_test = np.concatenate((x_test_cat, x_test[num_features].values), axis=1)