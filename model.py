# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 22:27:30 2021

@author: Siddhant Pant
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

df = pd.read_csv("bank-marketing.csv")
df.head()

df.info()
dfa = df.copy()
dfa.drop(dfa[dfa['pdays']<0].index, inplace = True)
dfa.replace({'response': {"yes":1, "no":0}}, inplace = True)
obj_col = []
num_col = []
for col in dfa.columns:
    if dfa[col].dtype=='O':
        obj_col.append(col)
    else:
        num_col.append(col)

print(obj_col)
print(num_col)

plt.figure(figsize = (8,6))
sns.heatmap(dfa.corr(), annot=True)
plt.title("Correlation of each numerical Features")
plt.show()

from sklearn.preprocessing import LabelEncoder

dfb = dfa[obj_col].apply(LabelEncoder().fit_transform)

dfc = dfb.join(dfa[num_col])

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

X = dfc.drop("response", axis = 1)
X.head()

y = dfc[['response']]
y.head()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

from sklearn.linear_model import LogisticRegression

import warnings 
warnings.filterwarnings("ignore")

lr = LogisticRegression()

lr.fit(X_train, y_train)

cv_score = cross_val_score(lr, X_train, y_train, cv = 5)
np.mean(cv_score)


y_pred = lr.predict(X_test)

from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

def Logistic_Regression():
    print(classification_report(y_test, y_pred))
    print('AUC Score:', roc_auc_score(y_test, y_pred))
    print('Accuracy :', accuracy_score(y_test, y_pred))
    print('Cross Validation:', cross_val_score(lr, X_train, y_train, cv=5))
    
Logistic_Regression()

pickle.dump(lr, open('model.pkl', 'wb'))

model = pickle.load(open('model.pkl', 'rb'))
