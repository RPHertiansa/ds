import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.cluster import KMeans
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler

#use kmean clustering algorithm
#4 steps, initialization, cluster assignment, centroid updating, repeat

#euclidian distance, shortest distance between 2 points
x1 = np.array([1, -1])
x2 = np.array([4, 3])
np.sqrt(((x1-x2)**2).sum())

data = load_wine()
wine = pd.DataFrame(data.data, columns=data.feature_names)

scatter_matrix(wine.iloc[:, [0,5]])
plt.show()

X = wine[['alcohol', 'total_phenols']]

scale = StandardScaler()
scale.fit(X)
print(scale.mean_)
print(scale.scale_)

X_scaled = scale.transform(X)