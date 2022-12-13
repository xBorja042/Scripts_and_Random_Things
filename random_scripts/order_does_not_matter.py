import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

print(" Script to test if order in a distribution does matter on the result. ")

experiment_length = 5000
unbalance_fraction = 0.1

list_0, list_1 = [0 for i in range(experiment_length)], [1 for i in range(int(experiment_length * unbalance_fraction))]
lt = list_0 +  list_1
df = pd.DataFrame({"v":lt})

plt.show(sns.countplot(df["v"]).set(title="Complete Data"))


    
df2 = df.sample(frac=1)

plt.show(sns.countplot(df2["v"]).set(title="Complete Data"))



for frac in [0.1, 0.3, 0.5, 0.8]:
    fig, axes = plt.subplots(1, 2, sharex=True, figsize=(10,5))
    fig.suptitle('Data fraction: ' + str(frac * 100) + " %")
    sns.countplot(df["v"].sample(frac=frac), ax=axes[0])
    axes[0].set_title('Random distribution drawn from ORDERED df')
    sns.countplot(df2["v"].sample(frac=frac), ax=axes[1])
    axes[1].set_title('Random distribution drawn from DISORDERED df')
    
