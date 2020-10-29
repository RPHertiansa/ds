import pandas as pd
import matplotlib.pyplot as plt
iris = pd.read_csv('https://sololearn.com/uploads/files/iris.csv')
iris.drop('id', axis=1, inplace=True) #drop id column

print(iris.shape)
print(iris.describe())
print(iris['species'].value_counts()) #count unique value in Species column
# iris is balanced dataset because each unique value has same amount

iris.hist()
plt.show()

#index mapping
inv_name_dict = {
    'iris-setosa': 0,
    'iris-versicolor': 1,
    'iris-virginica': 2
}
#color index
colors = [inv_name_dict[item] for item in iris['species']]

scatter = plt.scatter(iris['sepal_len'], iris['sepal_wd'], c = colors)
plt.xlabel('sepal length(cm)')
plt.ylabel('sepal width(cm)')
plt.legend(handles=scatter.legend_elements()[0], labels=inv_name_dict.keys())
plt.show()

scatter = plt.scatter(iris['petal_len'], iris['petal_wd'], c=colors)
plt.xlabel('petal length(cm)')
plt.ylabel('petal width(cm)')
plt.legend(handles=scatter.legend_elements()[0], labels=inv_name_dict.keys())
plt.show()

#petal more accurate in classifying