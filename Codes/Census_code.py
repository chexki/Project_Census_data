# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 15:51:54 2018

@author: Chexki
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 15:18:34 2018

@author: Chexki
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%% 
# Data Loading

adult_df= pd.read_csv(r'D:\DATA SCIENCE\Python\Pytho_docs\adult_data.csv',
                      header=None, delimiter= ' *, *',engine='python')
test_df= pd.read_csv(r'D:\DATA SCIENCE\Python\Pytho_docs\adult_test.csv',
                      header=None, delimiter= ' *, *',engine='python')
print(adult_df.head())
print(adult_df.shape)

#%%
# Adding Column names to the dataset

adult_df.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
'marital_status', 'occupation', 'relationship',
'race', 'sex', 'capital_gain', 'capital_loss',
'hours_per_week', 'native_country', 'income']

test_df.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
'marital_status', 'occupation', 'relationship',
'race', 'sex', 'capital_gain', 'capital_loss',
'hours_per_week', 'native_country', 'income']

print(adult_df.head())

#%%
# Create a copy of dataframe

adult_df_rev = pd.DataFrame.copy(adult_df)
print(adult_df_rev.describe(include= 'all'))


#%%

## finding missing values and Outliers

print(adult_df.isnull().sum())
adult_df.boxplot()     
plt.show() 
#%%
# '?' special character is used to denote missing values. hence,

for value in ['workclass', 'education', 'marital_status', 'occupation', 'relationship',
'race', 'sex', 'native_country', 'income']:
    print(value,":", sum(adult_df[value]== '?'))

for value in ['workclass', 'education', 'marital_status', 'occupation', 'relationship',
'race', 'sex', 'native_country', 'income']:
    print(value,":", sum(test_df[value]== '?'))
#%%
# Replacing missing values with mode of respective feature.

for value in ['workclass','occupation','native_country']:
    adult_df_rev[value].replace(['?'],
                [adult_df_rev.describe(include='all')[value][2]],
    inplace= True)
    
for value in ['workclass','occupation','native_country']:
    test_df[value].replace(['?'],
                [test_df.describe(include='all')[value][2]],
    inplace= True)

print(adult_df_rev.head(20))

#%%
# Verifying if the missing values were imputed.

for value in ['workclass', 'education', 'marital_status', 'occupation', 'relationship',
'race', 'sex', 'native_country', 'income']:
    print(value,":", sum(adult_df_rev[value]== '?'))

#%%
# In order to achieve better results by sklearn,
# setting numeric levels to categorical variables.
    
#%%
# Creating list of Categorical variables

colname =['workclass', 'education', 'marital_status', 'occupation', 'relationship',
'race', 'sex', 'native_country', 'income']

colname

#%%
# Data Preprocessing

from sklearn import preprocessing
le={}                                        # creating blank dictionary

for x in colname:
    le[x]=preprocessing.LabelEncoder()

for x in colname:
    adult_df_rev[x]= le[x].fit_transform(adult_df_rev.__getattr__(x))    
    
for x in colname:
    test_df[x]= le[x].fit_transform(test_df.__getattr__(x))   
#%%
    
adult_df_rev.head()
# 0 == <=50k
# 1 == >=50k

#%%

X = adult_df_rev.values[:,:-1]          #independent vars
Y = adult_df_rev.values[:,-1]           # dependent var
print(Y)

X_test_df = test_df.values[:,:-1]          #independent vars

#%%
# Scaling the data in normalized fashion

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
print(X)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_test_df)
X = scaler.transform(X_test_df)
print(X_test_df)


#%%
# Defining data type as an integer.
Y = Y.astype(int)

#%%

# Feature Selection
# CODE for REcursive feature selection

from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
classifier1 = RandomForestClassifier()
rfe = RFE(classifier1, 6)
model_rfe = rfe.fit(X,Y)
print("Num Features: ", model_rfe.n_features_)
print("Selected Features: ")
print(list(zip(colname, model_rfe.support_)))
print("Feature Ranking: ", model_rfe.ranking_)
#%%
# Splitting the data into testing and training

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3,
                                                    random_state=10)

#%%
# Traing the Logistic regression model 

from sklearn.linear_model import LogisticRegression
classifier2 = (LogisticRegression())

# fitting training data to the model

classifier2.fit(X_train,Y_train)
Y_pred = classifier2.predict(X_test)
print(list(zip(Y_test,Y_pred)))

print(Y_test)

#%%
# Evaluating the model
# Plotting Confusion Matrix (Y_test to Y_Pred)

from sklearn.metrics import confusion_matrix, accuracy_score,classification_report

cfm= confusion_matrix(Y_test,Y_pred)
print(cfm)

# [[7009  414]
#  [1318 1028]]

# Type-2 error is significantly high.
#############################################################################

print("Classification Report")
print(classification_report(Y_test,Y_pred))

#              precision    recall  f1-score   support

#          0       0.84      0.94      0.89      7423
#          1       0.71      0.44      0.54      2346

#avg / total       0.81      0.82      0.81     
#############################################################################

accuracy_score = accuracy_score(Y_test,Y_pred)
print("Accuracy of the Model:", accuracy_score)

# Accuracy of the Model: 0.822704473334

#%%
# storing the predicted probabilities

y_pred_prob = classifier2.predict_proba(X_test)
print(y_pred_prob)

#%%
# Adjusting The Threshold

y_pred_class=[]
for value in y_pred_prob[:,0]:
    if value < 0.72:
        y_pred_class.append(1)
    else:
        y_pred_class.append(0)
print(y_pred_class)

#%%
from sklearn.metrics import confusion_matrix, accuracy_score

cfm= confusion_matrix(Y_test.tolist(),y_pred_class)
print(cfm)

accuracy_score = accuracy_score(Y_test.tolist(),y_pred_class)
print("Accuracy of the Model:", accuracy_score)

# [[6089 1334]
# [ 689 1657]]
# Accuracy of the Model: 0.792916368103

#%%
# Cross Validating the model
from sklearn import cross_validation

# Performing kfold cross validation

kfold_cv = cross_validation.KFold(n=len(X_train),n_folds = 10)
print(kfold_cv)
# running the model using scoring metric accuracy
kfold_cv_result = cross_validation.cross_val_score( \
                                        estimator =classifier2,
                                        X=X_train,y=Y_train,
                                        scoring="accuracy",
                                        cv=kfold_cv)
print(kfold_cv_result)
# [ 0.81842105  0.81578947  0.81878017  0.8490566   0.82843352  0.82580079
#  0.8095656   0.81702501  0.83238262  0.82799473]

# Finding the mean
print(kfold_cv_result.mean())
## 0.824324957853
#####################################################################################
#%%
# Randomforest classifier
Y_pred = model_rfe.predict(X_test)
#print(list(zip(Y_test, Y_pred)))
#%%

from sklearn.metrics import confusion_matrix, accuracy_score,classification_report
cfm= confusion_matrix(Y_test,Y_pred)
print(cfm)

#[[7397   26]
 #[ 142 2204]]

print("Classification Report")
print(classification_report(Y_test,Y_pred))
accuracy_score = accuracy_score(Y_test,Y_pred)
print("Accuracy of the Model:", accuracy_score) 

# Accuracy of the Model: 0.982802743372
#%%

# Adjusting The Threshold
Y_pred_prob = rfe.predict_proba(X_test)
print(Y_pred_prob)

Y_pred_class=[]
for value in Y_pred_prob[:,0]:
    if value < 0.72:
        Y_pred_class.append(1)
    else:
        Y_pred_class.append(0)
print(Y_pred_class)
#%%
from sklearn.metrics import confusion_matrix, accuracy_score

cfm= confusion_matrix(Y_test.tolist(),Y_pred_class)
print(cfm)

accuracy_score = accuracy_score(Y_test.tolist(),Y_pred_class)
print("Accuracy of the Model:", accuracy_score)

#[[6916  507]
# [   0 2346]]
# Accuracy of the Model: 0.948101136247


######################################################################################
#%%
# Prediction on dataset

# Logistic
Y_pred_test_LR = classifier2.predict(X_test_df)

# RandomForest
Y_pred_test_RF = model_rfe.predict(X_test_df)
