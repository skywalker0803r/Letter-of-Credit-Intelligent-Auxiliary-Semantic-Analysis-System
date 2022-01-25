import requests
import pandas as pd
import re
import numpy as np
import joblib

# 導入演算法
url = 'https://gist.githubusercontent.com/skywalker0803r/7d1c5c3e8f11b6713a7994f08b62fbe9/raw/51992fcd154ea85b99f9332165fb5b4275139138/knn_nlp.py'
exec(requests.get(url).text)


# 預處理函數
def preprocess(x):
    x = str(x).upper() # 轉大寫字串
    x = re.sub('[\u4e00-\u9fa5]', '', x) # 去除中文
    x = re.sub(r'[^\w\s]','',x) # 去除標點符號
    x = x.replace('\n','').replace('\r','').replace('\t','') # 換行符號去除
    return str(x)

# 輸入欄位
x_col = ['45A','50','59','46A','47A','78']
# database
EXPNO對應表 = pd.read_csv('../data/對應表/EXPNO對應表.csv')#.head()
# 對輸入字串預處理
for col in x_col:
    EXPNO對應表[col] = EXPNO對應表[col].apply(preprocess)
str_max_len = 300
EXPNO對應表['X'] = EXPNO對應表[x_col].sum(axis=1) # 字串串起來
EXPNO對應表['X'] = EXPNO對應表['X'].apply(lambda x:str(x)[:str_max_len]) # 截斷

# 訓練資料
train_data = EXPNO對應表['X'].values.tolist()

# 開始訓練模型
model = coresystem(train_data,str_max_len=str_max_len)

# save index
model.index.saveIndex('index')

# load index
model.index.loadIndex('index')

# test
test_data = np.random.choice(train_data,size=3).tolist()
最相近鄰居 = model.predict(test_data)
for i,j in zip(test_data,最相近鄰居):
  assert i==j

#
print('all ok')


