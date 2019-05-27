#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 11:53:35 2019

@author: fangyucheng
Email: 664947387@qq.com
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from matplotlib.ticker import StrMethodFormatter

#import sklearn libraries
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score, cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, auc, make_scorer, confusion_matrix, f1_score, fbeta_score
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelBinarizer
#import different methods
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm

train_df = pd.read_csv("train.csv")
X_train, X_test, y_train, y_test = train_test_split(train_df)