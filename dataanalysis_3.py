import pandas as pd

#read data from csv
presidents_df = pd.read_cresidents_df = pd.read_csv('https://sololearn.com/uploads/files/president_heights_party.csv', index_col='name')

#basic stats, max, min, mean
max = presidents_df.max()
print(max)

min = presidents_df['age'].min()
print(min)

mean = presidents_df['age'].mean()
print(mean)

#mean will be equal to median if the data is symmetric
median = presidents_df['age'].median()
print(median)

#quantile divide data into 4 parts 0.25 0.5 0.75 1
quantile = presidents_df['age'].quantile(0.5)
print(quantile)

#variance 
var = presidents_df['age'].var()
print(var)

#standard deviation, sqrt of vatiance, menggambarkan tingkat perbedaan data, semakin kecil, data semakin seragam dan mendekati mean
std = presidents_df['age'].std()
print(std)

#describe, print all summary statistics except variance
describe = presidents_df.describe()
print(describe)
