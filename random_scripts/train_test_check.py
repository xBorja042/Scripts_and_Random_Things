
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

unbalanced_list_0 = [0 for i in range(400)]

for ub in [100, 50, 20, 10]:
    unbalanced_list_1 = [1 for i in range(ub)]
    
    u_l = unbalanced_list_0 +  unbalanced_list_1
    df = pd.DataFrame({'y': u_l})
    
    mu, sigma = 0, 0.1 
    s = np.random.normal(mu, sigma, len(u_l))
    df['x'] = s
    
    for seed in [1, 2, 7, 11, 27, 42]:
        X_train, X_test, y_train, y_test = train_test_split(
                                    df['x'], df['y'], test_size=0.33, random_state=seed)
        
        f, axes = plt.subplots(1, 3)
        sns.countplot(df['y'], ax=axes[0])
        sns.countplot(y_train, ax=axes[1])
        sns.countplot(y_test,  ax=axes[2])
        plt.show()
        
        f, axes = plt.subplots(1, 3)
        sns.kdeplot(df['x'], ax=axes[0])
        sns.kdeplot(X_train, ax=axes[1])
        sns.kdeplot(X_test,  ax=axes[2])
        plt.show()
    
print(" We can see train test split from sklearn Tends to preserve distributions")