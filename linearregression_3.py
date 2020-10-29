import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#buidling datasets
boston_dataset = load_boston()
boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
boston['MEDV'] = boston_dataset.target #being targeted to be predicted
model = LinearRegression()

X = boston[['RM']] #turn RM into matrix, making sure it remains 2D array
Y = boston['MEDV']

#train test split function, 30% test, 70% training
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 1)


#mode fitting, fit the model to the training data and find coefficient (intercept and slope)
model.fit(X_train, Y_train)
model.intercept_.round(2) #intercept -30.57
model.coef_.round(2) #slope 8.46

# MEDV = -30.57 + 8.46 * RM

#prediction, 6.5 is input
new_RM = np.array([6.5]).reshape(-1,1) #make sure 2d array
model.predict(new_RM) # 24.42606323

y_test_predicted = model.predict(X_test)

plt.scatter(X_test, Y_test, label='Testing Data')
plt.plot(X_test, y_test_predicted, label='Prediction' )
plt.xlabel('RM')
plt.ylabel('MEDV')
plt.legend(loc='upper left')
plt.show()

# residual, distance between a point to the line. Difference between observed and predicted value, 0 is good
# residual is scattered along y=0, ideally should be symmetrical and randomly spaced
residual = Y_test - y_test_predicted
plt.scatter(X_test, residual)
plt.hlines(y=0, xmin=X_test.min(), xmax=X_test.max(), linestyles='--')
plt.xlim((4,9))
plt.xlabel('RM')
plt.ylabel('residuals')
plt.show()

# mean squared error, closer to 0, the better
print((residual**2).mean())
#or
from sklearn.metrics import mean_squared_error
print(mean_squared_error(Y_test, y_test_predicted))

#results is 36.51, compare to variance of 92.26, this is not bad

# R^2 raquared, proportion of total variation explained by model
print(model.score(X_test, Y_test)) #about 60% of variance in testing data is explained by the model, the higher the better

# multivariate prediction might produce better model, but sometimes to many variable just spoil the model