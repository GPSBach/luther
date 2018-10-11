import pandas as pd
import numpy as np
import googlemaps
import pickle as pkl
%matplotlib inline

"""
Read in data
"""

# read in sale data
sale=pd.DataFrame()
for x in range (0,8000,100):
    data = pd.read_csv('./APIreturns/sale_{}.csv'.format(x))
    sale=sale.append(data,ignore_index=True)

"""
Parse google transit duration times into floats
"""
sale['duration_float'] = np.nan
for index, row in sale.iterrows():
    values=row['duration_trip']\
    .replace('[','')\
    .replace(']','')\
    .replace("'","")\
    .split()
#     if index % 1000 == 0:
#         print(index)
    if len(values)==2:
        sale['duration_float'].iloc[index] = float(values[0])
    elif len(values)==4:
        sale['duration_float'].iloc[index] = float(values[0])*60 + float(values[2])
        
with open('./luther_model_data_full.pkl', 'wb') as picklefile:
    pkl.dump(sale, picklefile)