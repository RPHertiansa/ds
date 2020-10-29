import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.cluster import KMeans
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler

#use kmean clustering algorithm
#4 steps, initialization, cluster assignment, centroid updating, repeat

data = load_wine()
wine = pd.DataFrame(data.data, columns=data.feature_names)

X = wine[['alcohol', 'total_phenols']]

scale = StandardScaler()
scale.fit(X)
X_scaled = scale.transform(X)

#find optimal value of k
kmeans = KMeans(n_clusters=3)
kmeans.fit(X_scaled)
print(kmeans.inertia_)
#inertia decrease as the number of k increase, optimal k is when the inertia doesnt decrease rapidly

inertia = []
for i in np.arange(1, 11):
    km = KMeans(n_clusters=i)
    km.fit(X_scaled)
    inertia.append(km.inertia_)
    
plt.plot(np.arange(1,11), inertia, marker='o')
plt.xlabel('n of clusters')
plt.ylabel('inertia')
plt.show()

#k=3 is the optimal number