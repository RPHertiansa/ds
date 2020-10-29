import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#buidling datasets
boston_dataset = load_boston()
boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
boston['MEDV'] = boston_dataset.target #being targeted
model = LinearRegression()

X = boston[['RM']] #turn RM into matrix, making sure it remains 2D array
Y = boston['MEDV']

#train test split function, 30% test, 70% training
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 1)

#dimension check
# print(X_train.shape)
# print(Y_train.shape)
# print(X_test.shape)
# print(Y_test.shape)

#mode fitting, fit the model to the training data and find coefficient (intercept and slope)
model.fit(X_train, Y_train)
print(model.intercept_.round(2)) #intercept
print(model.coef_.round(2)) #slope

# MEDV = -30.57 + 8.46 * RM

#prediction, 6.5 is input
new_RM = np.array([6.5]).reshape(-1,1) #make sure 2d array
print(model.predict(new_RM))

y_test_predicted = model.predict(X_test)