import pandas as pd
import numpy as np

wine_dict = {
    'red_wine': [3, 6, 5],
    'white_wine': [5, 0, 10]
}
sales = pd.DataFrame(wine_dict, index=['adam', 'bob', 'charles']) #indexing array
print(sales['white_wine'])
print(sales['red_wine'])
print(sales)

print('-------')

#indexing from np.array
print(pd.Series(np.array([1,2,3]), index=['a','b','c']))