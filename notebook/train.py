import requests
import pandas as pd
import re
import numpy as np
import joblib

# 導入演算法
url = 'https://gist.githubusercontent.com/skywalker0803r/14922b3cb7ca23e0de5eec59a69cec25/raw/fc99cfce652af5214867936363fc254fe68c7dc2/CountVectorizer_NearestNeighbors.py'
exec(requests.get(url).text)

# 輸入欄位
x_col = ['45A','50','59','46A','47A','78']

# EXPNO對應表
EXPNO對應表 = pd.read_csv('../data/對應表/EXPNO對應表.csv')
EXPNO對應表['X'] = EXPNO對應表[x_col].sum(axis=1) # 字串串起來

# 訓練資料
train_data = EXPNO對應表['X'].values.tolist()

# 開始訓練模型
model = coresystem()
model.fit(train_data)

# test
test_data = train_data
最相近鄰居 = model.predict(test_data)
for i,j in zip(test_data,最相近鄰居):
  assert i==j
print('test ok')

# save model
save_path = '../models/model'
joblib.dump(model,save_path)
print(f'save done save model in {save_path}')

