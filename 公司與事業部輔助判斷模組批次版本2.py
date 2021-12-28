import streamlit as st
import pandas as pd
import numpy as np
import itertools
from collections import Counter
from scipy import stats
import joblib
import torch
import random
from pytorch_lightning import seed_everything
import os
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForQuestionAnswering
from transformers import pipeline
import re
from IPython.display import HTML
import warnings;warnings.simplefilter('ignore')
import requests

train_df = pd.read_csv('./data/對應表/EXPNO對應表.csv')

# set seed 
def set_seed(seed = int):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    seed_everything(seed)
    return random_state
seed = set_seed(42)

# coresystem
url = 'https://gist.githubusercontent.com/skywalker0803r/823ecfa11147c918ce805409bf4d733f/raw/24d559f020ade561f5b4ba49e02d1634e7dfb067/Letter-of-Credit-Intelligent-Auxiliary-Semantic-Analysis-System.py'
exec(requests.get(url).text)
a = coresystem(['apple','banana'])
a.predict('applee')
x_col = '45A' #產品名
x_col2 = '50' #開狀人
x_col3 = '59' #公司名
銀行_col = ['46A','47A','78'] #銀行欄位
col = ['45A','50','59','46A']
s = {}
n = 100
for i in col:
    s[i] = coresystem(data = train_df[i].values.tolist()[:n])

# 展示正確率
def get_acc(df):
    correct ,correct_label = [] ,[]
    df['集成預測代號'] = df['EXPNO'].apply(lambda x:x[:2])
    for i in df.index:
        if df.loc[i,'推薦公司事業部'] == df.loc[i,'集成預測代號']:
            correct.append('yes')
        else:
            correct.append('no')
    result = pd.DataFrame({'correct':correct})
    try:
        return result['correct'].value_counts()['yes']/len(result)
    except:
        return 0

# 主UI設計
st.title('公司與事業部輔助判斷模組')
st.image('./bert.png')
# 上傳測試檔案
st.text('請上傳csv或xlsx格式的檔案')
test_df = st.file_uploader("upload file", type={"csv", "xlsx"})
# 判斷檔案是哪一種格式
if test_df is not None:
    try:
        test_df = pd.read_csv(test_df,index_col=0).reset_index(drop=True).head()
    except:
        test_df = pd.read_excel(test_df,index_col=0).reset_index(drop=True).head()
button = st.button('predict')

# 推論按鈕
if button:
    st.write('正在預測產品')
    result = []
    test_df['EXPNO'] = 'not find'
    for idx in tqdm(test_df.index[:5]):
        predict = s['45A'].predict(test_df.loc[idx,'45A'])
        EXPNO = train_df.loc[train_df[col[0]]==predict,'EXPNO']
        test_df.loc[idx,'EXPNO'] = EXPNO.values[0]
    
    st.write(test_df)
    st.write(f'正確率:{get_acc(test_df)}')
    錯誤筆數 = len(text_output.loc[test_df['正確與否']=='no',:])
    st.write(f'錯誤筆數:{錯誤筆數}')

