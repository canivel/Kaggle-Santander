import itertools
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# import matplotlib
# import matplotlib.pyplot as plt
# matplotlib.use("Agg") #Needed to save figures
from sklearn import cross_validation
import xgboost as xgb
from sklearn.metrics import roc_auc_score
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



training = pd.read_csv("data/train.csv", index_col=0)
test = pd.read_csv("data/test.csv", index_col=0)

training = remove_feat_constants(training)
training = remove_feat_identicals(training)

test = remove_feat_constants(test)
test = remove_feat_identicals(test)

training.to_csv('data/clean_train.csv')
test.to_csv('data/clean_test.csv')

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
X['total'] = X.sum(axis=1)
X['n0'] = (X == 0).sum(axis=1)

# Normalize each feature to unit norm (vector length)
x_train_normalized = normalize(X, axis=0)

# Run PCA
pca = PCA(n_components=6)
x_train_projected = pca.fit_transform(x_train_normalized)


# p = 75 # 261 features validation_1-auc:0.836441
# X_bin = Binarizer().fit_transform(scale(x_train_projected))
# selectChi2 = SelectPercentile(chi2, percentile=p).fit(X_bin, y)
# selectF_classif = SelectPercentile(f_classif, percentile=p).fit(x_train_projected, y)
#
# chi2_selected = selectChi2.get_support()
# chi2_selected_features = [ f for i,f in enumerate(x_train_projected.columns) if chi2_selected[i]]
# print('Chi2 selected {} features {}.'.format(chi2_selected.sum(),
#    chi2_selected_features))
# f_classif_selected = selectF_classif.get_support()
# f_classif_selected_features = [ f for i,f in enumerate(x_train_projected.columns) if f_classif_selected[i]]
# print('F_classif selected {} features {}.'.format(f_classif_selected.sum(),
#    f_classif_selected_features))
# selected = chi2_selected & f_classif_selected
# print('Chi2 & F_classif selected {} features'.format(selected.sum()))
# features = [ f for f,s in zip(X.columns, selected) if s]
# print (features)

X_sel = x_train_projected

X_train, X_test, y_train, y_test = \
  cross_validation.train_test_split(X_sel, y, random_state=1301, stratify=y, test_size=0.3)

clf = xgb.XGBClassifier(missing=9999999999,
                        max_depth = 8,
                        n_estimators=1000,
                        learning_rate=0.05,
                        nthread=4,
                        subsample=0.8,
                        colsample_bytree=0.5,
                        min_child_weight = 8,
                        seed=4242)


clf.fit(X_train, y_train, early_stopping_rounds=50, eval_metric="auc",
        eval_set=[(X_train, y_train), (X_test, y_test)])

print('Overall AUC:', roc_auc_score(y, clf.predict_proba(X_sel, ntree_limit=clf.best_iteration)[:,1]))

test['total'] = test.sum(axis=1)
test['n0'] = (test == 0).sum(axis=1)
# test['n1'] = (test['var38']/test['n0'])*100
# test['n2'] = (test['var38']/test['var15'])*100
# test['n3'] = test[["var15", "var38", "saldo_var30", "saldo_medio_var5_hace2", "saldo_medio_var5_hace3"]].mean(axis=1)
# test['n4'] = (test['var15']/test['n0'])*100
# test['n5'] = (test['saldo_var30']/test['total'])*100
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


#sel_test = test[features]
x_test_normalized = normalize(test, axis=0)
sel_test = pca.transform(x_train_normalized)

y_pred = clf.predict_proba(sel_test, ntree_limit=clf.best_iteration)

submission = pd.DataFrame({"ID":test.index, "TARGET":y_pred[:,1]})
submission.to_csv("submission.csv", index=False)

# mapFeat = dict(zip(["f"+str(i) for i in range(len(features))],features))
# ts = pd.Series(clf.booster().get_fscore())
# #ts.index = ts.reset_index()['index'].map(mapFeat)
# ts.sort_values()[-15:].plot(kind="barh", title=("features importance"))
#
# featp = ts.sort_values()[-15:].plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))
# plt.title('XGBoost Feature Importance')
# fig_featp = featp.get_figure()
# fig_featp.savefig('feature_importance_xgb.png', bbox_inches='tight', pad_inches=1)