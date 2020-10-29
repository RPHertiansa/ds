import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#read data from csv
presidents_df = pd.read_cresidents_df = pd.read_csv('https://sololearn.com/uploads/files/president_heights_party.csv', index_col='name')

height = presidents_df['height']
age = presidents_df['age']
party = presidents_df['party'].value_counts()

plt.scatter(height, age) #scatter plot
plt.xlabel('height')
plt.ylabel('age')
plt.title('US Presidents')
plt.show()

plt.hist(height) #histogram
plt.show()

plt.style.use('ggplot')
party.plot(kind='bar')
plt.show()