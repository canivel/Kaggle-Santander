import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg") #Needed to save figures
from sklearn import cross_validation
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import SelectPercentile, f_classif,chi2
from sklearn.preprocessing import Binarizer, scale, StandardScaler
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import VotingClassifier

training = pd.read_csv("data/train.csv", index_col=0)
test = pd.read_csv("data/test.csv", index_col=0)

print(training.shape)
print(test.shape)

# Replace -999999 in var3 column with most common value 2 
# See https://www.kaggle.com/cast42/santander-customer-satisfaction/debugging-var3-999999
# for details
training = training.replace(-999999,2)


# Replace 9999999999 with NaN
# See https://www.kaggle.com/c/santander-customer-satisfaction/forums/t/19291/data-dictionary/111360#post111360
# training = training.replace(9999999999, np.nan)
# training.dropna(inplace=True)
# Leads to validation_0-auc:0.839577

X = training.iloc[:,:-1]
y = training.TARGET

# Add zeros per row as extra feature
X['n0'] = (X == 0).sum(axis=1)
X['n1'] = X['var38']/X['n0']

test['n0'] = (test == 0).sum(axis=1)
test['n1'] = test['var38']/test['n0']

p = 75 # 261 features validation_1-auc:0.848642

X_bin = Binarizer().fit_transform(scale(X))
selectChi2 = SelectPercentile(chi2, percentile=p).fit(X_bin, y)
selectF_classif = SelectPercentile(f_classif, percentile=p).fit(X, y)

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

X_sel = X[features]

sel_test = test[features]

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_sel, y, random_state=1301, stratify=y, test_size=0.3)

# xgb_model = xgb.XGBClassifier(
#     objective='binary:logistic',
#     missing=9999999999,
#     nthread=4)

# parameters = {
#               'learning_rate': [0.02, 0.1], #so called `eta` value
#               'max_depth': [5, 12],
#               'min_child_weight': [1, 3],
#               'subsample': [0.7, 0.5],
#               'colsample_bytree': [0.7, 1]
#               }

# clf = GridSearchCV(xgb_model, parameters, n_jobs=4,
#                    cv=cross_validation.StratifiedKFold(y_train, n_folds=2, shuffle=True),
#                    verbose=2, refit=True,scoring='roc_auc')
# ('Raw AUC score:', 0.8309212299879124)
# colsample_bytree: 0.7
# learning_rate: 0.1
# max_depth: 5
# min_child_weight: 3
# subsample: 0.7

# clf.fit(X_train, y_train)
#
# best_parameters, score, _ = max(clf.grid_scores_, key=lambda x: x[1])
# print('Raw AUC score:', score)
# for param_name in sorted(best_parameters.keys()):
#     print("%s: %r" % (param_name, best_parameters[param_name]))

XGBClassifier1 = xgb.XGBClassifier(objective='binary:logistic',
                                   missing=9999999999,
                                   nthread=4,
                                   n_estimators=400,
                                   max_depth=5,
                                   learning_rate=0.1,
                                   subsample=0.85,
                                   colsample_bytree=0.7,
                                   colsample_bylevel=0.8,
                                   min_child_weight=5,
                                   gamma=1,
                                   seed=23)
XGBClassifier2 = xgb.XGBClassifier(objective='binary:logistic',
                                   missing=9999999999,
                                   max_depth=8,
                                   n_estimators=1000,
                                   learning_rate=0.05,
                                   nthread=4,
                                   subsample=0.8,
                                   colsample_bytree=0.5,
                                   min_child_weight=8,
                                   seed=1313)
#0.825461
classifier = VotingClassifier([('clf1', XGBClassifier1), ('clf2', XGBClassifier2)], voting='soft', weights=[1, 1])

classifier.fit(X_train, y_train)


testingPreds=classifier.predict_proba(sel_test);
submission = pd.DataFrame({"ID":test.index, "TARGET":testingPreds[:,1]})
submission.to_csv("XGBoostEnsembled.csv", index=False)


# mapFeat = dict(zip(["f"+str(i) for i in range(len(features))],features))
# ts = pd.Series(clf.booster().get_fscore())
# #ts.index = ts.reset_index()['index'].map(mapFeat)
# ts.sort_values()[-15:].plot(kind="barh", title=("features importance"))
#
# featp = ts.sort_values()[-15:].plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))
# plt.title('XGBoost Feature Importance')
# fig_featp = featp.get_figure()
# fig_featp.savefig('feature_importance_xgb.png', bbox_inches='tight', pad_inches=1)