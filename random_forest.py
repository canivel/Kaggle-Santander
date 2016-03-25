import pandas
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.preprocessing import Binarizer, scale, StandardScaler
from sklearn.grid_search import GridSearchCV
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectPercentile, f_classif,chi2

df_train = pandas.read_csv("data/train.csv")
df_test  = pandas.read_csv("data/test.csv")

id_test = df_test['ID']
y_train = df_train['TARGET'].values
X_train = df_train.drop(['ID','TARGET'], axis=1)
X_test = df_test.drop(['ID'], axis=1)

#START FEATURES ENGINEERING

#remove constant columns
remove = []
for col in X_train.columns:
    if X_train[col].std() == 0:
        remove.append(col)

X_train.drop(remove, axis=1, inplace=True)
X_test.drop(remove, axis=1, inplace=True)

# remove duplicated columns
remove = []
c = X_train.columns
for i in range(len(c)-1):
    v = X_train[c[i]].values
    for j in range(i+1,len(c)):
        if np.array_equal(v,X_train[c[j]].values):
            remove.append(c[j])

X_train.drop(remove, axis=1, inplace=True)
X_test.drop(remove, axis=1, inplace=True)

# X_train['total'] = X_train.sum(axis=1)
X_train['n0'] = (X_train == 0).sum(axis=1)
# X_train['var38_by_n0'] = (X_train['var38'].mean()/X_train['n0'])*100
#print(X_train[['var38', 'n0', 'var15']])
# X_train['var38'] = np.isclose(X_train.var38, 117310.979016)
# X_train['logvar38'] = X_train.loc[~X_train['var38mc'], 'var38'].map(np.log)
# X_train.loc[X_train['var38mc'], 'logvar38'] = 0

# X_test['total'] = X_test.sum(axis=1)
X_test['n0'] = (X_train == 0).sum(axis=1)

# X_test['var38_by_n0'] = (X_test['var38'].mean()/X_test['n0'])*100
# X_test['var38mc'] = np.isclose(X_test.var38, 117310.979016)
# X_test['logvar38'] = X_test.loc[~X_test['var38mc'], 'var38'].map(np.log)
# X_test.loc[X_test['var38mc'], 'logvar38'] = 0

#print(X_train[['var38', 'n0', 'var15']])

#END FEATURES ENGINEERING

#START FEATURES SELECTION

#n_estimators=100
#p=75 #0.829381456745
#p=60 #0.829966462021
#p = 50 # 0.830174696959
#p = 45 # 0.831054096049
#p = 40 # 0.83145710484
p = 39 # 0.834694917771
#p = 38 # 0.830570176831
#p = 35 # 0.831077883599
#p = 30 # 0.830324484692

X_bin = Binarizer().fit_transform(scale(X_train))
selectChi2 = SelectPercentile(chi2, percentile=p).fit(X_bin, y_train)
selectF_classif = SelectPercentile(f_classif, percentile=p).fit(X_train, y_train)

chi2_selected = selectChi2.get_support()
chi2_selected_features = [ f for i,f in enumerate(X_train.columns) if chi2_selected[i]]
print('Chi2 selected {} features {}.'.format(chi2_selected.sum(),
   chi2_selected_features))
f_classif_selected = selectF_classif.get_support()
f_classif_selected_features = [ f for i,f in enumerate(X_train.columns) if f_classif_selected[i]]
print('F_classif selected {} features {}.'.format(f_classif_selected.sum(),
   f_classif_selected_features))
selected = chi2_selected & f_classif_selected
print('Chi2 & F_classif selected {} features'.format(selected.sum()))
features = [ f for f,s in zip(X_train.columns, selected) if s]
print (features)

#END FEATURES SELECTION

X_sel = X_train[features]
sel_test = X_test[features]

# set up RF_1
rf_1 = RandomForestClassifier(bootstrap=True,
                             criterion='entropy',
                             min_samples_split=20,
                             max_depth=17,
                             n_estimators=1000,
                             n_jobs=4,
                             oob_score=False,
                             random_state=1301,
                             verbose=1)

# set up RF_2
dt_1 = DecisionTreeClassifier(criterion='entropy',
                              max_depth=17,
                              max_leaf_nodes=16,
                              random_state=4242)

ab_1 = AdaBoostClassifier(
    n_estimators=20,
    learning_rate=0.75,
    base_estimator=ExtraTreesClassifier(
        n_estimators=400,
        max_features=30,
        max_depth=12,
        min_samples_leaf=100,
        min_samples_split=100,
        verbose=1,
        n_jobs=-1))

clfs = [('rf', rf_1), ('dt', dt_1), ('ab', ab_1)]
# set up ensemble of rf_1 and rf_2
clf = VotingClassifier(estimators=clfs, voting='soft', weights=[2, 1, 1])


# clf = DecisionTreeClassifier(criterion='entropy',
#                              max_depth=17,
#                              max_leaf_nodes=16,
#                              random_state=4242)

# param_grid = {
#     #'n_estimators': [100],
#     #'max_features': ['auto', 'sqrt', 'log2'],
#     'max_leaf_nodes': [2, 8, 16],
#     'min_samples_split': [2, 5, 10],
#     'max_depth': [2, 15, 20],
#     'criterion': ['entropy', 'gini'],
# }
#
# clf = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5, scoring='roc_auc', verbose=1)
#{'min_samples_split': 5, 'max_leaf_nodes': 16, 'criterion': 'entropy', 'max_depth': 15}
# clf = RandomForestClassifier(n_estimators=100, max_depth=17, random_state=1, verbose=2, max_leaf_nodes=16, min_samples_split=10)

scores = cross_validation.cross_val_score(clf, X_sel, y_train, scoring='roc_auc', cv=5)
print(scores.mean())

clf.fit(X_sel, y_train)

y_pred = clf.predict_proba(sel_test)

submission = pandas.DataFrame({"ID":id_test, "TARGET":y_pred[:,1]})
submission.to_csv("submission_rf.csv", index=False)