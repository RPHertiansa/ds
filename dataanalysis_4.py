import pandas as pd
import numpy as np

#read data from csv
presidents_df = pd.read_cresidents_df = pd.read_csv('https://sololearn.com/uploads/files/president_heights_party.csv', index_col='name')

#Groupby, find value based on a condition
print(presidents_df.groupby('party').mean())
print(presidents_df.groupby('party')['height'].median())

#Aggregations, multiple operations on groupby

print(presidents_df.groupby('party')['height'].agg(['min', np.median, max]))

print(presidents_df.groupby('party').agg({
    'height': [np.median, np.mean],
    'age': [min, max]}))