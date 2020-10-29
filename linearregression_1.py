import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
boston_dataset = load_boston()

boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
boston['MEDV'] = boston_dataset.target #we will predict this value later
print(boston.shape)
# print(boston.head())

#correlation matric, measure linear relationships between variables
corr_matrix = boston.corr().round(2)
print(corr_matrix)

boston.plot(kind='scatter',
    x = 'RM',
    y = 'MEDV',
    figsize=(8,6))
plt.show()