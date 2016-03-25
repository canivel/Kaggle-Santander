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
# remove = []
# for col in training.columns:
#     if training[col].std() == 0:
#         remove.append(col)
#
# training.drop(remove, axis=1, inplace=True)
# test.drop(remove, axis=1, inplace=True)
#
# # remove duplicated columns
# remove = []
# c = training.columns
# for i in range(len(c)-1):
#     v = training[c[i]].values
#     for j in range(i+1,len(c)):
#         if np.array_equal(v,training[c[j]].values):
#             remove.append(c[j])
#
# training.drop(remove, axis=1, inplace=True)
# test.drop(remove, axis=1, inplace=True)

training = pd.read_csv("data/train.csv", index_col=0)
test = pd.read_csv("data/test.csv", index_col=0)

training = remove_feat_constants(training)
training = remove_feat_identicals(training)

test = remove_feat_constants(test)
test = remove_feat_identicals(test)

# print(training.shape)
# print(test.shape)

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
# X['sum_saldos'] = X[["saldo_var30",
#                            "saldo_var42",
#                            "saldo_var5",
#                            "saldo_medio_var5_hace2",
#                            "saldo_medio_var5_hace3",
#                            "saldo_medio_var5_ult3",
#                            "saldo_medio_var5_ult1"]].mean(axis=1)
#
# X['perc_saldos'] = X['sum_saldos']/X['total']

# X['var38_by_n0'] = (X['var38'].mean()/X['n0'])*100

# X['var38mc'] = np.isclose(X.var38, 117310.979016)
# X['logvar38'] = X.loc[~X['var38mc'], 'var38'].map(np.log)
# X.loc[X['var38mc'], 'logvar38'] = 0

# X['var38_by_var15'] = (X['var38']/X['var15'])*100
# X['mean_top_features'] = X[["var15", "var38", "saldo_var30", "saldo_medio_var5_hace2", "saldo_medio_var5_hace3"]].mean(axis=1)
# X['var15_by_n0'] = (X['var15']/X['n0'])*100
# X['var38_by_total'] = (X['var38']/X['total'])*100


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

X_sel = X[features]

test['n0'] = (test == 0).sum(axis=1)

test = test[features]
test_index = test.index

train = np.array(X_sel)
test = np.array(test)

# object array to float
train = train.astype(float)
test = test.astype(float)

# i like to train on log(1+x) for RMSLE ;)
# The choice is yours :)
label_log = np.log1p(y)

params = {}
params["missing"] = "9999999999"
params["objective"] = "reg:linear"
params["n_estimators"] = 1000
params["eta"] = 0.009
params["learning_rate"] = 0.05
params["min_child_weight"] = 5
params["subsample"] = 0.85
params["colsample_bytree"] = 0.8
params["scale_pos_weight"] = 1.0
params["silent"] = 0
params["max_depth"] = 7


plst = list(params.items())

xgtrain = xgb.DMatrix(train, label=label_log)
xgtest = xgb.DMatrix(test)


num_rounds = 400
model = xgb.train(plst, xgtrain, num_rounds)
preds1 = model.predict(xgtest)
model = xgb.train(plst, xgtrain, num_rounds)
preds2 = model.predict(xgtest)

preds = np.expm1((preds1+preds2)/2)

submission = pd.DataFrame({"ID":test_index, "TARGET":preds})
submission.to_csv("submission_xgb_matrix.csv", index=False)