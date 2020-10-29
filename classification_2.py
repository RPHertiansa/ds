import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
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
print(predict[:5]) #first 5 predictions

#probability of a prediction, 0 to 1 for each class
y_pred_prob = knn.predict_proba(X_test)
print(y_pred_prob[:5]) #probability of first 5 predictions

#accuracy, proportion of data points that match the prediction
print(knn.score(X_test, Y_test))

print((predict==Y_test.values).sum()/Y_test.size)

#confusion matrix, berapa nilai true yg benar terprediksi
matrix = confusion_matrix(Y_test, predict, labels=['iris-setosa', 'iris-versicolor', 'iris_virginica'])
print(matrix)

plot_confusion_matrix(knn, X_test, Y_test, cmap=plt.cm.Blues)
plt.show()