import pandas as pd
import glob

submissions = glob.glob('*submission*.csv')

dfs = []

for f in submissions:
    _df = pd.read_csv(f, index_col=0)
    dfs.append(_df)


df = pd.concat(dfs, axis=1)

df['FINAL'] = df.mean(axis=1)

submission = pd.DataFrame({"ID":df.index, "TARGET":df['FINAL'].values})
submission.to_csv("submission_average.csv", index=False)

# training = pd.read_csv("data/clean_train.csv", index_col=0)
# test = pd.read_csv("data/clean_test.csv", index_col=0)


