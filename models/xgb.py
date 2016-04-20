# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
from __future__ import division

#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# from subprocess import check_output
# print(check_output(["ls", "data"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# load data
df_train = pd.read_csv('data/train.csv')
df_test = pd.read_csv('data/test.csv')

# remove constant columns
remove = []
for col in df_train.columns:
    if df_train[col].std() == 0:
        remove.append(col)

df_train.drop(remove, axis=1, inplace=True)
df_test.drop(remove, axis=1, inplace=True)

# remove duplicated columns
remove = []
c = df_train.columns
for i in range(len(c)-1):
    v = df_train[c[i]].values
    for j in range(i+1,len(c)):
        if np.array_equal(v,df_train[c[j]].values):
            remove.append(c[j])

df_train.drop(remove, axis=1, inplace=True)
df_test.drop(remove, axis=1, inplace=True)

important_features = ['var38', 'var15', 'saldo_var30', 'saldo_medio_var5_ult3', 'saldo_medio_var4_hace2', 'num_var45_hace3', 'saldo_medio_var5_ult1'
                          ,'num_var45_ult3', 'saldo_var42', 'num_var22_ult1', 'num_var22_hace3']

labels_train = df_train['TARGET'].values
features_train = df_train[important_features].values

id_test = df_test['ID']
features_test = df_test[important_features].values

scaler = StandardScaler()
features_train = scaler.fit_transform(features_train)
features_test = scaler.fit_transform(features_test)

# length of dataset
len_train = len(features_train)
len_test  = len(features_test)

# classifier
clf = xgb.XGBClassifier(missing=np.nan, max_depth=5, n_estimators=350, learning_rate=0.03, nthread=4, subsample=0.95, colsample_bytree=0.85, seed=4242)

features_fit, features_eval, labels_fit, labels_eval= train_test_split(features_train, labels_train, test_size=0.3)

# fitting
clf.fit(features_train, labels_train, early_stopping_rounds=20, eval_metric="auc", eval_set=[(features_eval, labels_eval)])

print('Overall AUC:', roc_auc_score(labels_train, clf.predict_proba(features_train)[:,1]))

# predicting
y_pred= clf.predict_proba(features_test)[:,1]

submission = pd.DataFrame({"ID":id_test, "TARGET":y_pred})
submission.to_csv("submission.csv", index=False)

print('Completed!')