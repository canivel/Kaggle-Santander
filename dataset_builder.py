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
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import SelectPercentile, f_classif,chi2
from sklearn.preprocessing import Binarizer, scale, StandardScaler
from sklearn import cross_validation

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

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(features_train, labels_train, random_state=1301, stratify=labels_train, test_size=0.3)

    clf = DecisionTreeClassifier(criterion='gini',
                                 min_samples_split=10,
                                 max_depth=10,
                                 max_leaf_nodes=16,
                                 max_features=2)


    #clf_acc = clf.fit(X_train, y_train)
    # print(clf.best_estimator_)
    #feature_importance = clf.feature_importances_
    #print (feature_importance)

    #pred = clf_acc.predict_proba(X_test)[:,1]
    #print (y_test, pred)
    # acc = accuracy_score(y_test, pred)
    # print ("Acc {}".format(acc))

    clf = clf.fit(features_train, labels_train)

    pred = clf.predict_proba(features_test)[:,1]

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

    # feature_importance = clf.feature_importances_
    #
    # print (feature_importance)

    predictions_file = open("data/rf_prediction.csv", "wb")
    predictions_file_object = csv.writer(predictions_file)
    predictions_file_object.writerow(["ID", "TARGET"])
    predictions_file_object.writerows(zip(ids, pred))
    predictions_file.close()

def ada_prediction(features_train, labels_train, features_test, ids):

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(features_train, labels_train, random_state=1301, stratify=labels_train, test_size=0.3)

    clf = AdaBoostClassifier(RandomForestClassifier(bootstrap=True,
                                                    criterion='entropy', max_depth=None, max_features=2,
                                                    max_leaf_nodes=16, min_samples_split=10, n_estimators=1000,
                                                    n_jobs=-1, oob_score=False),
                              algorithm="SAMME",
                              n_estimators=200)


    #clf_acc = clf.fit(X_train, y_train)
    # print(clf.best_estimator_)
    #feature_importance = clf.feature_importances_
    #print (feature_importance)

    #pred = clf_acc.predict_proba(X_test)[:,1]
    #print (y_test, pred)
    # acc = accuracy_score(y_test, pred)
    # print ("Acc {}".format(acc))

    clf = clf.fit(features_train, labels_train)

    pred = clf.predict_proba(features_test)[:,1]

    predictions_file = open("data/canivel_ada_forest.csv", "wb")
    predictions_file_object = csv.writer(predictions_file)
    predictions_file_object.writerow(["ID", "TARGET"])
    predictions_file_object.writerows(zip(ids, pred))
    predictions_file.close()

def gradientboost_prediction(features_train, labels_train, features_test, ids):

    class RandomForestClassifier_compability(RandomForestClassifier):
        def predict(self, X):
            return self.predict_proba(X)[:, 1][:,np.newaxis]

    base_estimator = RandomForestClassifier_compability()

    clf = GradientBoostingClassifier(loss='deviance', learning_rate=0.1,
                             n_estimators=5, subsample=0.3,
                             min_samples_split=2,
                             min_samples_leaf=1,
                             max_depth=3,
                             init=base_estimator,
                             random_state=None,
                             max_features=None,
                             verbose=2,
                             learn_rate=None)

    clf = clf.fit(features_train, labels_train)

    pred = clf.predict_proba(features_test)[:,1]

    # feature_importance = clf.feature_importances_
    #
    # print (feature_importance)

    predictions_file = open("data/rf_prediction.csv", "wb")
    predictions_file_object = csv.writer(predictions_file)
    predictions_file_object.writerow(["ID", "TARGET"])
    predictions_file_object.writerows(zip(ids, pred))
    predictions_file.close()


if __name__ == "__main__":

    df_train = pd.read_csv('data/train.csv', index_col=0)
    df_test = pd.read_csv('data/test.csv', index_col=0)
    ids = df_test.index

    df_train = df_train.replace(-999999,2)

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


    X = df_train.iloc[:,:-1]
    labels_train = df_train['TARGET']

    # Add zeros per row as extra feature
    X['n0'] = (X == 0).sum(axis=1)
    X['n1'] = X['var38']/X['n0']
    X['n2'] = X['var38']/X['var15']*100

    p=75
    X_bin = Binarizer().fit_transform(scale(X))
    selectChi2 = SelectPercentile(chi2, percentile=p).fit(X_bin, labels_train)
    selectF_classif = SelectPercentile(f_classif, percentile=p).fit(X, labels_train)

    chi2_selected = selectChi2.get_support()
    chi2_selected_features = [ f for i,f in enumerate(X.columns) if chi2_selected[i]]
    print('Chi2 selected {} features {}.'.format(chi2_selected.sum(),
       chi2_selected_features))
    f_classif_selected = selectF_classif.get_support()
    f_classif_selected_features = [ f for i,f in enumerate(X.columns) if f_classif_selected[i]]
    print('F_classif selected {} features {}.'.format(f_classif_selected.sum(),
       f_classif_selected_features))
    selected = chi2_selected & f_classif_selected
    print('Chi2 & F_classif selected {} features'.format(selected.sum()))
    features = [ f for f,s in zip(X.columns, selected) if s]
    print (features)

    features_train = X[features]

    df_test['n0'] = (df_test == 0).sum(axis=1)
    df_test['n1'] = df_test['var38']/df_test['n0']
    df_test['n2'] = df_test['var38']/df_test['var15']*100

    features_test = df_test[features]


    # scaler = StandardScaler()
    # features_train = scaler.fit_transform(features_train)
    # features_test = scaler.fit_transform(features_test)

    # features_train, features_test, labels_train, labels_test = train_test_split(features_train, labels_train, test_size=0.2, random_state=42)
    # logistc_prediction_testing(features_train, labels_train, features_test, labels_test, ids)

    #logistc_prediction(features_train, labels_train, features_test, ids)

    ada_prediction(features_train, labels_train, features_test, ids)
