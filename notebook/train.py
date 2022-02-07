import requests
import pandas as pd
import re
import numpy as np
import joblib

# algo
url = 'https://gist.githubusercontent.com/skywalker0803r/0b78f6f77a6ec72195cb6a7feb3f5a7c/raw/78b41b87f722953d080f91b5527cfa58f65bc518/Text%2520similarity%2520comparison%2520with%2520Bert%2520and%2520nmslib%2520and%2520clean%2520function.py'
exec(requests.get(url).text)

# input
x_col = ['45A','50','59','46A','47A','78']

# load combined_excel
df = pd.read_excel('../data/combined_excel.xlsx').astype(str)
df['x'] = df[x_col].sum(axis=1) # concat
df['x'] = df['x'].apply(clean_text) # preprocess

# save preprocessed data
df.to_csv('../data/combined_excel_addx.csv')

# training model
model = coresystem(data=df['x'].values.tolist()[:5],str_max_len=500)

# save index
model.index.saveIndex('../models/model_index')

# load index
model.index.loadIndex('../models/model_index')

# test model
x_test = df['x'].values.tolist()[:1]
y_pred = model.predict(x_test)
assert x_test == y_pred

print('all done!')

