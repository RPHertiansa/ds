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

kmeans = KMeans(n_clusters=3)
kmeans.fit(X_scaled)
y_pred = kmeans.predict(X_scaled)
centroid = kmeans.cluster_centers_ #coordinates of 3 centroids

#plot centroid position
plt.scatter(X_scaled[:,0], X_scaled[:,1], c=y_pred)
plt.scatter(centroid[:,0], centroid[:,1], marker='*', s=250, c=[0, 1, 2], edgecolors='k')
plt.xlabel('alcohol')
plt.ylabel('total_phenols')
plt.title('k-means (k=3')
plt.show()

#new data to predict
X_new = np.array([[13, 2.5]])
X_new_scaled = scale.transform(X_new)
print(X_new_scaled)
print(kmeans.predict(X_new_scaled))

