#%%
print("Script to check wether if the are differences between ways of splitting a dataset based on the alg")

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
import pandas as pd



def get_frec_df(y) -> pd.DataFrame:
    df = pd.DataFrame({"v": pd.Series(y).value_counts()})
    df["value"] = df.index
    df["%"] = df["v"]/y.shape[0]*100
    del df["v"]
    return df

def plot_dist(df: pd.DataFrame, shuffling_alg: str, fold_order: int) -> None:
    dfp = df.set_index("value")
    title = "Distribution for " + shuffling_alg + " with fold no: " + str(fold_order)
    print(title)
    dfp.plot.bar(title=title)
    

X = np.array([1, 1, 1, 4, 1, 2, 1, 4, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 2, 4, 1, 1, 2, 1, 1, 1, 2, 2])
y = np.array([0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 3, 2, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 2, 2, 3, 2, 1])
print(f"Original data X ---> {X},\n y ---> {y}", "\n")
initial_df = get_frec_df(y)
plot_dist(initial_df, "Original Data", 0)
initial_df["%_initial_dist"] = initial_df["%"]
print("Original DF target distribution: \n", initial_df)

n_splits = 5



kf = KFold(n_splits=n_splits)
print("\n Vanilla --> ", kf.get_n_splits(X, y))

print(kf)

for i, (train_index, test_index) in enumerate(kf.split(X, y)):
    print(f"Fold {i}:")
    print(f"  Train data X ---> {X[train_index]}, y ---> {y[train_index]}")
    print(f"  Test  data X ---> {X[test_index]}, y ---> {y[test_index]}")
    dist_train, dist_test = get_frec_df(y[train_index]), get_frec_df(y[test_index])
    print("New train DF target distribution for KFOLD: \n", dist_train)
    print("New test  DF target distribution for KFOLD: \n", dist_test)
    dfp = pd.concat([dist_train, initial_df["%_initial_dist"]], axis=1)
    plot_dist(dfp, "K_fold", i)
    print(" ")


# print(1)

#%%
print("\n  ----- ----- ----- \n")
    
skf = StratifiedKFold(n_splits=n_splits)
print("Stratified --> ", skf.get_n_splits(X, y))

print(skf)

for i, (train_index, test_index) in enumerate(skf.split(X, y)):
    print(f"Fold {i}:")
    print(f"  Train data X ---> {X[train_index]}, y ---> {y[train_index]}")
    print(f"  Test  data X ---> {X[test_index]}, y ---> {y[test_index]}")
    dist_train, dist_test = get_frec_df(y[train_index]), get_frec_df(y[test_index])
    print("New train DF target distribution for stratKFOLD: \n", dist_train)
    print("New test  DF target distribution for stratKFOLD: \n", dist_test)
    dfp = pd.concat([dist_train, initial_df["%_initial_dist"]], axis=1)
    plot_dist(dfp, "strat_K_fold", i)
    print(" ")
# print(2)