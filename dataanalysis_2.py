import pandas as pd

#read data from csv
presidents_df = pd.read_cresidents_df = pd.read_csv('https://sololearn.com/uploads/files/president_heights_party.csv', index_col='name')

print(presidents_df.head()) #peek first 5 rows
print('------------')
print(presidents_df.tail()) #peek lask 5 rows
print(presidents_df.loc['Abraham Lincoln']) #find by name
print(presidents_df.iloc[15:18]) #find by rows
print(presidents_df.info()) #get dataframe info