import pandas as pd
import csv
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn import svm, grid_search
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt


def logistc_prediction(features_train, labels_train, features_test, ids):

    clf = LogisticRegression(C=10, class_weight=None, dual=False, fit_intercept=True,
                             intercept_scaling=1, penalty='l1', random_state=None, tol=0.0001)

    clf = clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)

    predictions_file = open("data/canivel_logist_regression.csv", "wb")
    predictions_file_object = csv.writer(predictions_file)
    predictions_file_object.writerow(["ID", "TARGET"])
    predictions_file_object.writerows(zip(ids, pred))
    predictions_file.close()
    print ("Done")

def decision_tree_prediction(features_train, labels_train, features_test, ids):
    clf = DecisionTreeClassifier(criterion='gini',
                                 min_samples_split=10,
                                 max_depth=10,
                                 max_leaf_nodes=16,
                                 max_features=2)


    clf = clf.fit(features_train, labels_train)

    pred = clf.predict_proba(features_test)[:,1]

    feature_importance = clf.feature_importances_

    print (feature_importance)

    predictions_file = open("data/canivel_decision_tree.csv", "wb")
    predictions_file_object = csv.writer(predictions_file)
    predictions_file_object.writerow(["ID", "TARGET"])
    predictions_file_object.writerows(zip(ids, pred))
    predictions_file.close()


def rf_prediction(features_train, labels_train, features_test, ids):

    clf = RandomForestClassifier(bootstrap=True,
            criterion='entropy', max_depth=None, max_features=2,
            max_leaf_nodes=16, min_samples_split=10, n_estimators=1000,
            n_jobs=-1, oob_score=False)

    clf = clf.fit(features_train, labels_train)

    pred = clf.predict_proba(features_test)[:,1]


    print (pred)

    # feature_importance = clf.feature_importances_
    #
    # print (feature_importance)

    predictions_file = open("data/rf_prediction.csv", "wb")
    predictions_file_object = csv.writer(predictions_file)
    predictions_file_object.writerow(["ID", "TARGET"])
    predictions_file_object.writerows(zip(ids, pred))
    predictions_file.close()



if __name__ == "__main__":

    df_train = pd.read_csv('data/train.csv')
    df_test = pd.read_csv('data/test.csv')
    ids = df_test['ID'].values


    # sum == 0
    remove = []
    for col in df_train.columns:
        if(df_train[col].sum() == 0):
            remove.append(col)

    df_train.drop(remove, axis=1, inplace=True)
    df_test.drop(remove, axis=1, inplace=True)

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

    labels_train = df_train['TARGET'].values
    features_train = df_train.drop(['ID','TARGET'], axis=1).values

    features_test = df_test.drop(['ID'], axis=1).values

    # scaler = StandardScaler()
    # features_train = scaler.fit_transform(features_train)
    # features_test = scaler.fit_transform(features_test)

    # features_train, features_test, labels_train, labels_test = train_test_split(features_train, labels_train, test_size=0.2, random_state=42)
    # logistc_prediction_testing(features_train, labels_train, features_test, labels_test, ids)

    #logistc_prediction(features_train, labels_train, features_test, ids)

    decision_tree_prediction(features_train, labels_train, features_test, ids)
