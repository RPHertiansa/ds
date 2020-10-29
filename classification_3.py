import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, plot_confusion_matrix

iris = pd.read_csv('https://sololearn.com/uploads/files/iris.csv')
iris.drop('id', axis=1, inplace=True) #drop id column

X = iris[['petal_len', 'petal_wd']]
Y = iris['species']

inv_name_dict = {
    'iris-setosa': 0,
    'iris-versicolor': 1,
    'iris-virginica': 2
}

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1, stratify=Y)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, Y_train) #model fitting

predict = knn.predict(X_test)


#probability of a prediction, 0 to 1 for each class
y_pred_prob = knn.predict_proba(X_test)


#accuracy, proportion of data points that match the prediction
knn.score(X_test, Y_test)

(predict==Y_test.values).sum()/Y_test.size

#confusion matrix, berapa nilai true yg benar terprediksi
matrix = confusion_matrix(Y_test, predict, labels=['iris-setosa', 'iris-versicolor', 'iris_virginica'])

#cross validation, split between test-train is random, k-fold cross validation overcome the randomness
# data is divided into k subsets, bigger k, smaller diferences in resampling, smaller the technique bias
knn_cv = KNeighborsClassifier(n_neighbors=3)
cv_score = cross_val_score(knn_cv, X, Y, cv=5) #5fold cross validation
print(cv_score)
print(cv_score.mean()) #accuracy on average

#tuning K (hyperparameter) using GridSearchCV
#create new knn model
knn2 = KNeighborsClassifier()
param_grid = {'n_neighbors': np.arange(2,10)}

knn_gscv = GridSearchCV(knn2, param_grid, cv=5)
knn_gscv.fit(X, Y)

#find best value for n_neighbors and best accuracy
print(knn_gscv.best_params_) #4
print(knn_gscv.best_score_) #0.96

#rebuild final model
knn_final = KNeighborsClassifier(n_neighbors=4)
knn_final.fit(X, Y)
y_pred = knn_final.predict(X)
print(knn_final.score(X, Y))

#feeding new data to the model
#predict new data
new_data = np.array([[3.76, 1.2], [5.25, 1.2], [1.58, 1.2]]) #make sure it is 2D array

result = knn_final.predict(new_data)
prob = knn_final.predict_proba(new_data)
print(result)
print(prob)