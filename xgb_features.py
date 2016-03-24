import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg") #Needed to save figures
from sklearn import cross_validation
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import SelectPercentile, f_classif,chi2
from sklearn.preprocessing import Binarizer, scale, StandardScaler
from sklearn.grid_search import GridSearchCV

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
X['n1'] = (X['var38']/X['n0'])*100
X['n2'] = (X['var38']/X['var15'])*100
X['n3'] = X[["var15", "var38", "saldo_var30", "saldo_medio_var5_hace2", "saldo_medio_var5_hace3"]].mean(axis=1)
X['n4'] = (X['var15']/X['n0'])*100
X['n5'] = ((X['var15']+X['var38'])/X['n0'])*100
# mean_var38 = X.var38.mean()
# max_var38 = X.var38.max()
# min_var38 = X.var38.min()
# rich_factor_var38 = mean_var38+(max_var38-mean_var38)/2
# poor_factor_var38 = mean_var38-(mean_var38-min_var38)/2

# mean_n2 = X.var38.mean()
# max_n2 = X.n2.max()
# min_n2 = X.n2.min()
# rich_factor_n2 = mean_n2+(max_n2-mean_n2)/2
# poor_factor_n2 = mean_n2-(min_n2-mean_n2)/2

# X['rich_var38'] = 0
# X['rich_var38'].loc[(X.var38 > rich_factor_var38)] = 3 # rich
# X['rich_var38'].loc[(X.var38 <= rich_factor_var38) & (X.var38 >= poor_factor_var38)] = 2 # median
# X['rich_var38'].loc[(X.var38 < poor_factor_var38)] = 1 # poor
#p = 90 # 341 features validation_1-auc:0.848001
#p = 86 # 308 features validation_1-auc:0.848039
#p = 80 # 284 features validation_1-auc:0.848414
#p = 77 # 267 features validation_1-auc:0.848000
p = 75 # 261 features validation_1-auc:0.848642
# p = 73 # 257 features validation_1-auc:0.848338
# p = 70 # 259 features validation_1-auc:0.848588
# p = 69 # 238 features validation_1-auc:0.848547
# p = 67 # 247 features validation_1-auc:0.847925
# p = 65 # 240 features validation_1-auc:0.846769
# p = 60 # 222 features validation_1-auc:0.848581
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

X_train, X_test, y_train, y_test = \
  cross_validation.train_test_split(X_sel, y, random_state=1301, stratify=y, test_size=0.3)

clf = xgb.XGBClassifier(missing=9999999999,
                max_depth = 5,
                n_estimators=1000,
                learning_rate=0.02,
                nthread=4,
                subsample=0.7,
                colsample_bytree=0.7,
                seed=4242)


clf.fit(X_train, y_train, early_stopping_rounds=50, eval_metric="auc",
        eval_set=[(X_train, y_train), (X_test, y_test)])

print('Overall AUC:', roc_auc_score(y, clf.predict_proba(X_sel, ntree_limit=clf.best_iteration)[:,1]))

test['n0'] = (test == 0).sum(axis=1)
test['n1'] = (test['var38']/test['n0'])*100
test['n2'] = (test['var38']/test['var15'])*100
test['n3'] = test[["var15", "var38", "saldo_var30", "saldo_medio_var5_hace2", "saldo_medio_var5_hace3"]].mean(axis=1)
test['n4'] = (test['var15']/test['n0'])*100
test['n5'] = ((test['var15']+test['var38'])/test['n0'])*100
# mean_var38 = test.var38.mean()
# max_var38 = test.var38.max()
# min_var38 = test.var38.min()
# rich_factor_var38 = mean_var38+(max_var38-mean_var38)/2
# poor_factor_var38 = mean_var38-(mean_var38-min_var38)/2

# mean_n2 = test.var38.mean()
# max_n2 = test.n2.max()
# min_n2 = test.n2.min()
# rich_factor_n2 = mean_n2+(max_n2-mean_n2)/2
# poor_factor_n2 = mean_n2-(min_n2-mean_n2)/2
#
# test['rich_var38'] = 0
# test['rich_var38'].loc[(test.var38 > rich_factor_var38)] = 3 # rich
# test['rich_var38'].loc[(test.var38 <= rich_factor_var38) & (test.var38 >= poor_factor_var38)] = 2 # median
# test['rich_var38'].loc[(test.var38 < poor_factor_var38)] = 1 # poor


sel_test = test[features]    
y_pred = clf.predict_proba(sel_test, ntree_limit=clf.best_iteration)

submission = pd.DataFrame({"ID":test.index, "TARGET":y_pred[:,1]})
submission.to_csv("submission.csv", index=False)

mapFeat = dict(zip(["f"+str(i) for i in range(len(features))],features))
ts = pd.Series(clf.booster().get_fscore())
#ts.index = ts.reset_index()['index'].map(mapFeat)
ts.sort_values()[-15:].plot(kind="barh", title=("features importance"))

featp = ts.sort_values()[-15:].plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))
plt.title('XGBoost Feature Importance')
fig_featp = featp.get_figure()
fig_featp.savefig('feature_importance_xgb.png', bbox_inches='tight', pad_inches=1)