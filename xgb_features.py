import itertools
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
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

training = pd.read_csv("data/train.csv", index_col=0)
test = pd.read_csv("data/test.csv", index_col=0)

def remove_feat_constants(data_frame):
    # Remove feature vectors containing one unique value,
    # because such features do not have predictive value.
    print("")
    print("Deleting zero variance features...")
    # Let's get the zero variance features by fitting VarianceThreshold
    # selector to the data, but let's not transform the data with
    # the selector because it will also transform our Pandas data frame into
    # NumPy array and we would like to keep the Pandas data frame. Therefore,
    # let's delete the zero variance features manually.
    n_features_originally = data_frame.shape[1]
    selector = VarianceThreshold()
    selector.fit(data_frame)
    # Get the indices of zero variance feats
    feat_ix_keep = selector.get_support(indices=True)
    orig_feat_ix = np.arange(data_frame.columns.size)
    feat_ix_delete = np.delete(orig_feat_ix, feat_ix_keep)
    # Delete zero variance feats from the original pandas data frame
    data_frame = data_frame.drop(labels=data_frame.columns[feat_ix_delete],
                                 axis=1)
    # Print info
    n_features_deleted = feat_ix_delete.size
    print("  - Deleted %s / %s features (~= %.1f %%)" % (
        n_features_deleted, n_features_originally,
        100.0 * (np.float(n_features_deleted) / n_features_originally)))
    return data_frame

def remove_feat_identicals(data_frame):
    # Find feature vectors having the same values in the same order and
    # remove all but one of those redundant features.
    print("")
    print("Deleting identical features...")
    n_features_originally = data_frame.shape[1]
    # Find the names of identical features by going through all the
    # combinations of features (each pair is compared only once).
    feat_names_delete = []
    for feat_1, feat_2 in itertools.combinations(
            iterable=data_frame.columns, r=2):
        if np.array_equal(data_frame[feat_1], data_frame[feat_2]):
            feat_names_delete.append(feat_2)
    feat_names_delete = np.unique(feat_names_delete)
    # Delete the identical features
    data_frame = data_frame.drop(labels=feat_names_delete, axis=1)
    n_features_deleted = len(feat_names_delete)
    print("  - Deleted %s / %s features (~= %.1f %%)" % (
        n_features_deleted, n_features_originally,
        100.0 * (np.float(n_features_deleted) / n_features_originally)))
    return data_frame

# remove constant columns
remove = []
for col in training.columns:
    if training[col].std() == 0:
        remove.append(col)

training.drop(remove, axis=1, inplace=True)
test.drop(remove, axis=1, inplace=True)
#
# # remove duplicated columns
remove = []
c = training.columns
for i in range(len(c)-1):
    v = training[c[i]].values
    for j in range(i+1,len(c)):
        if np.array_equal(v,training[c[j]].values):
            remove.append(c[j])
#
training.drop(remove, axis=1, inplace=True)
test.drop(remove, axis=1, inplace=True)

# training = pd.read_csv("data/clean_train.csv", index_col=0)
# test = pd.read_csv("data/clean_test.csv", index_col=0)

X = training.iloc[:,:-1]
y = training.TARGET

# training = remove_feat_constants(training)
# training = remove_feat_identicals(training)
#
# test = remove_feat_constants(test)
# test = remove_feat_identicals(test)

# print(training.shape)
# print(test.shape)

# Replace -999999 in var3 column with most common value 2 
# See https://www.kaggle.com/cast42/santander-customer-satisfaction/debugging-var3-999999
# for details
X = X.replace(-999999,2)


# Replace 9999999999 with NaN
# See https://www.kaggle.com/c/santander-customer-satisfaction/forums/t/19291/data-dictionary/111360#post111360
# training = training.replace(9999999999, np.nan)
# training.dropna(inplace=True)
# Leads to validation_0-auc:0.839577

# Add zeros per row as extra feature
X['n0'] = (X == 0).sum(axis=1)
#
X['var38mc'] = np.isclose(X.var38, 117310.979016)
X['logvar38'] = X.loc[~X['var38mc'], 'var38'].map(np.log)
X.loc[X['var38mc'], 'logvar38'] = 0
#
# X['saldo_total'] = X[["saldo_var30",
#                            "saldo_var42",
#                            "saldo_var5",
#                            "saldo_medio_var5_hace2",
#                            "saldo_medio_var5_hace3",
#                            "saldo_medio_var5_ult3",
#                            "saldo_medio_var5_ult1"]].sum(axis=1)
#

# X['age_var36'] = 0
# X['age_var36'].loc[(X.var15 >= 23) & (X.var15 <= 50) & (X.var36 == 99)] = 1
#
# # test['total'] = test.sum(axis=1)
test['n0'] = (test == 0).sum(axis=1)
#
test['var38mc'] = np.isclose(test.var38, 117310.979016)
test['logvar38'] = test.loc[~test['var38mc'], 'var38'].map(np.log)
test.loc[test['var38mc'], 'logvar38'] = 0
#
# test['saldo_total'] = test[["saldo_var30",
#                            "saldo_var42",
#                            "saldo_var5",
#                            "saldo_medio_var5_hace2",
#                            "saldo_medio_var5_hace3",
#                            "saldo_medio_var5_ult3",
#                            "saldo_medio_var5_ult1"]].sum(axis=1)
#

# test['age_var36'] = 0
# test['age_var36'].loc[(test.var15 >= 23) & (test.var15 <= 50) & (test.var36 == 99)] = 1
#
# #extra drops
X = X.drop(['var38mc'], axis=1)
test = test.drop(['var38mc'], axis=1)


#p = 90 # 341 features validation_1-auc:0.848001
#p = 86 # 308 features validation_1-auc:0.848039
#p = 80 # 284 features validation_1-auc:0.848414
#p = 77 # 267 features validation_1-auc:0.848000
#p = 75 # 261 features validation_1-auc:0.848642
# p = 73 # 257 features validation_1-auc:0.848338
# p = 70 # 259 features validation_1-auc:0.848588
# p = 69 # 238 features validation_1-auc:0.848547
# p = 67 # 247 features validation_1-auc:0.847925
# p = 65 # 240 features validation_1-auc:0.846769
# p = 60 # 222 features validation_1-auc:0.848581
p = 75 # 261 features validation_1-auc:0.0.845549

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

# features = ['var15', 'saldo_var5', 'saldo_var12', 'saldo_var13_corto', 'saldo_var13_largo', 'saldo_var13', 'saldo_var14',
#             'saldo_var20', 'saldo_var24', 'saldo_var26', 'saldo_var25', 'saldo_var30', 'saldo_var33', 'saldo_var37',
#             'saldo_var40', 'saldo_var42', 'saldo_var44', 'var36', 'num_var22_hace2', 'num_var22_hace3',
#             'num_var22_ult1', 'num_var22_ult3', 'num_med_var45_ult3', 'num_meses_var5_ult3', 'num_meses_var8_ult3',
#             'num_meses_var12_ult3', 'num_meses_var13_corto_ult3', 'num_meses_var13_largo_ult3', 'num_meses_var17_ult3',
#             'num_meses_var33_ult3', 'num_meses_var39_vig_ult3', 'num_meses_var44_ult3', 'num_op_var39_comer_ult1',
#             'num_op_var40_comer_ult3', 'num_op_var40_efect_ult1', 'num_op_var40_efect_ult3', 'num_op_var41_comer_ult1',
#             'num_op_var41_comer_ult3', 'num_op_var41_efect_ult1', 'num_op_var41_efect_ult3', 'num_op_var39_efect_ult1',
#             'num_op_var39_efect_ult3', 'num_sal_var16_ult1', 'num_var43_emit_ult1', 'num_var43_recib_ult1',
#             'num_trasp_var11_ult1', 'num_trasp_var33_in_hace3', 'num_venta_var44_ult1', 'num_var45_hace3', 'num_var45_ult1',
#             'saldo_medio_var5_hace2', 'saldo_medio_var5_hace3', 'saldo_medio_var5_ult1', 'saldo_medio_var5_ult3',
#             'saldo_medio_var8_hace2', 'saldo_medio_var8_hace3', 'saldo_medio_var8_ult1', 'saldo_medio_var8_ult3',
#             'saldo_medio_var12_hace2', 'saldo_medio_var12_hace3', 'saldo_medio_var12_ult1', 'saldo_medio_var12_ult3',
#             'saldo_medio_var13_corto_hace2', 'saldo_medio_var13_corto_hace3', 'saldo_medio_var13_corto_ult1',
#             'saldo_medio_var13_corto_ult3', 'saldo_medio_var13_largo_hace2', 'saldo_medio_var13_largo_hace3',
#             'saldo_medio_var13_largo_ult1', 'saldo_medio_var13_largo_ult3', 'saldo_medio_var33_hace2', 'saldo_medio_var33_hace3',
#             'saldo_medio_var33_ult1', 'saldo_medio_var33_ult3', 'saldo_medio_var44_hace2', 'saldo_medio_var44_hace3',
#             'saldo_medio_var44_ult1', 'saldo_medio_var44_ult3', 'var38', 'n0', 'saldo_medio_low']


X_sel = X[features]
sel_test = test[features]

# scaler = StandardScaler()
# X_sel = scaler.fit_transform(X_sel)
# sel_test = scaler.fit_transform(sel_test)

# number_of_folds=10
# y_values = y.values
# kfolder = cross_validation.StratifiedKFold(y_values, n_folds=number_of_folds, shuffle=True, random_state=15)
#
#
#
# for train_index, test_index in kfolder:
#     print (train_index)
#     # X_train, X_test = X_sel[train_index], X_sel[test_index]
#     # y_train, y_test = y[train_index], y[test_index]


X_train, X_test, y_train, y_test = \
  cross_validation.train_test_split(X_sel, y, random_state=1301, stratify=y, test_size=0.3)

clf = xgb.XGBClassifier(missing=9999999999,
                        max_depth = 5,
                        n_estimators=1000,
                        learning_rate=0.05,
                        nthread=4,
                        subsample=0.8,
                        colsample_bytree=0.5,
                        min_child_weight = 8,
                        seed=1313)


clf.fit(X_train, y_train, early_stopping_rounds=50, eval_metric="auc",
        eval_set=[(X_train, y_train), (X_test, y_test)])

print('Overall AUC:', roc_auc_score(y, clf.predict_proba(X_sel, ntree_limit=clf.best_iteration)[:,1]))

y_pred = clf.predict_proba(sel_test, ntree_limit=clf.best_iteration)

submission = pd.DataFrame({"ID":test.index, "TARGET":y_pred[:,1]})
submission.to_csv("submission_1101.csv", index=False)

mapFeat = dict(zip(["f"+str(i) for i in range(len(features))],features))
ts = pd.Series(clf.booster().get_fscore())
#ts.index = ts.reset_index()['index'].map(mapFeat)
ts.sort_values()[-15:].plot(kind="barh", title=("features importance"))

featp = ts.sort_values()[-15:].plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))
plt.title('XGBoost Feature Importance')
fig_featp = featp.get_figure()
fig_featp.savefig('feature_importance_xgb.png', bbox_inches='tight', pad_inches=1)