import streamlit as st
import time
import re 
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
from simhash import Simhash
from sklearn.metrics import accuracy_score

# 核心算法
# MSKTS文本相似度比對算法
url = 'https://gist.githubusercontent.com/skywalker0803r/7c00d680d731b99ab549dd40a96006ce/raw/d9bb060a42285053faa5227df521b43f381f1a0b/MSKTS.py'
exec(requests.get(url).text)

#help function
def 根據受益人限縮database(database,受益人,公司寶典):
    公司英文名稱2代號 = dict(zip(公司寶典['公司英文名稱'],公司寶典['代號']))
    代號 = str(公司英文名稱2代號[受益人])
    expno第一碼 = 代號[:1]
    cond1 = database['EXPNO'].apply(lambda x:str(x)[0]) == expno第一碼
    cond2 = database['受益人'].apply(lambda x:x[0]) == 受益人
    cond = cond1 & cond2
    return database.loc[cond,:]
 
# 保留英文字母
def keep_alpha(str1): 
  char = "" 
  for x in str(str1):
    if x.isalpha(): 
      char = "".join([char, x])
  return char 

# 基於規則之關鍵字匹配算法
def matching(sentence,database):
  candidate_list = []
  for word in database:
    if word in sentence: 
      candidate_list.append(word)
  return candidate_list

# string_list中的string若為其他string的"子集"則剔除
def substringSieve(string_list):
    string_list.sort(key=lambda s: len(s), reverse=True)
    out = []
    for s in string_list:
      if not any([s in o for o in out]):
        out.append(s)
    return out

# 輸入sentence前處理
def preprocess_raw_sentence(x):
  x = str(x).upper() # 轉大寫字串
  x = re.sub('[\u4e00-\u9fa5]', '', x) # 去除中文
  x = re.sub(r'[^\w\s]','',x) # 去除標點符號
  x = x.replace('\n', '').replace('\r', '').replace('\t', '') # 去除換行符號
  str.strip(x) # 移除左右空白
  x = x.replace(' '*3,' '*1)# 去除"三重"空白
  x = x.replace(' '*2,' '*1)# 去除"雙重"空白
  x = ' ' + x + ' '# 出現在頭的 就不可能對到前後加空格的 這種情形要想想怎麼對照(加上左右空白)
  return x

# 產品後處理
def product_name_postprocess(x):
  x = str(x).upper() # 轉大寫字串
  x = x.replace('-','')# 去除標點符號
  x = x.replace('.','')# 去除標點符號
  x = x.replace(',','')# 去除標點符號
  x = x.strip() # 去除空白
  return x

# 基於關鍵字比對方法的預測函數
def predict_keyword(title,test_df,Unrecognized,input_col,database,output_col):
  result = []
  for i in tqdm(test_df.index):
    candidate_list = matching(
        sentence = test_df.loc[i,input_col],
        database = set(database) - set(Unrecognized) - set(['',' '*1,' '*2,' '*3])
        )
    result.append(substringSieve(candidate_list))
  test_df[output_col] = result
  return test_df

# 載入數據      
# 歷史資料庫
database = pd.read_excel('./data/combined_excel.xlsx')
# 新的測試數據
test_data = pd.read_csv('./data/測試數據/0927到2022.csv')

# 讀取"產品名"寶典
品名寶典 = pd.read_excel('./data/寶典/寶典人工處理後/寶典.v8.202111202.xlsx',engine='openpyxl')[['CODIV','DIVNM','ITEMNM']]
品名寶典 = 品名寶典.rename(columns={'ITEMNM':'品名','DIVNM':'公司事業部門','CODIV':'公司代號'})
品名寶典['品名'] = 品名寶典['品名'].apply(lambda x:product_name_postprocess(x))

# 讀取"開狀人"寶典
開狀人寶典 = pd.read_csv('./data/寶典/開狀人寶典.csv')

# 讀取"公司"寶典
公司寶典 = pd.read_csv('./data/寶典/公司寶典加尾綴.csv')

# 按照時間排序資料
def sort_by_form(df):
    df = df.sort_values(by='from')
    df = df.reset_index(drop=True)
    return df
database = sort_by_form(database)
test_data = sort_by_form(test_data)

# 定義欄位
產品名輸入 = '45A' #產品名
開狀人輸入 = '50' #開狀人
受益人輸入 = '59' #受益人
開狀銀行輸入 = 'LTADDRESS.1' #銀行輸入
輸出欄位 = ['產品名','開狀人','受益人','開狀銀行']
輸入欄位 = ['45A','50','59','LTADDRESS.1']

# 預處理函數
# 針對模型輸入做預處理
def 預處理(df):
  產品名輸入 = '45A' #產品名
  開狀人輸入 = '50' #開狀人
  受益人輸入 = '59' #受益人
  開狀銀行輸入 = 'LTADDRESS.1' #銀行輸入
  for i in [產品名輸入,開狀人輸入,受益人輸入]:
    df[i] = df[i].apply(lambda x:preprocess_raw_sentence(x))
  return df

# 抽特徵函數
def 抽特徵(df):
  # 1預測產品
  df = predict_keyword(
      title = '正在預測產品',
      test_df = df,
      Unrecognized = ['PE','MA','EA','GRADE','INA','PACK','PP','PA','',' '*1,' '*2],
      input_col = 產品名輸入,
      database = 品名寶典['品名'].values.tolist(),
      output_col = '產品名',
      )

  # 2預測開狀人
  df = predict_keyword(
      title = '正在預測開狀人',
      test_df = df,
      Unrecognized = ['',' '*1,' '*2],
      input_col = 開狀人輸入,
      database = 開狀人寶典['開狀人'].values.tolist(),
      output_col = '開狀人',
      )
  # 3預測公司(受益人)
  df = predict_keyword(
      title = '正在預測受益人',
      test_df = df,
      Unrecognized = ['',' '*1,' '*2],
      input_col = 受益人輸入,
      database = 公司寶典['公司英文名稱'].values.tolist(),
      output_col = '受益人',
      )
  # 4預測開狀銀行
  df['開狀銀行'] = df[開狀銀行輸入].apply(lambda x:str(x)[:8])
  return df

# 根據特定欄位和索引給出候選答案清單
def 根據特定欄位和索引給出候選答案清單(col,idx,k,database_size=100,database=None,test_data=None):
  # 預處理
  database['處理過的資料'] = (database[col]).apply(keep_alpha)
  test_data['處理過的資料'] = (test_data[col]).apply(keep_alpha)
  # 建立模型
  model = MSKTS()
  model.fit(list(set(database['處理過的資料'].sample(database_size).values.tolist()) - set(['',' '*1,' '*2]))) #去除空白
  # 產生預測答案清單
  output = [i[0] for i in model.predict(test_data['處理過的資料'][idx],k=k)]
  預測答案清單 = database.loc[database['處理過的資料'].isin(output),'EXPNO'].dropna().apply(lambda x:str(x)[:2]).values.tolist()
  return 預測答案清單

# 推論函數
def 推論函數(database_size,test_data):
  answer_list = []
  # 遍歷整個test_data
  for idx in tqdm(range(len(test_data))):
    # 先根據受益人限縮database
    try:
      restricted_database = 根據受益人限縮database(database,test_data.loc[idx,'受益人'][0],公司寶典)
    except:
      restricted_database = database
    # 根據col和idx做推論
    o1 = 根據特定欄位和索引給出候選答案清單(
      col='產品名',idx=idx,k=3,
      database_size = min(database_size,len(restricted_database)),
      database = restricted_database,
      est_data = test_data)
    o2 = 根據特定欄位和索引給出候選答案清單(
      col='開狀人',idx=idx,k=3,
      database_size = min(database_size,len(restricted_database)),
      database = restricted_database,
      test_data = test_data)
    o3 = 根據特定欄位和索引給出候選答案清單(
      col='受益人',idx=idx,k=3,
      database_size = min(database_size,len(restricted_database)),
      database = restricted_database,
      test_data = test_data)
    o4 = 根據特定欄位和索引給出候選答案清單(
      col='開狀銀行',idx=idx,k=3,
      database_size = min(database_size,len(restricted_database)),
      database = restricted_database,
      test_data = test_data)
    # 集成投票
    o = o1#+o2+o3+o4
    try:
      ensemble_output = max(o,key=o.count)
    except:
      ensemble_output = 'not find'
    # 加入答案清單
    answer_list.append(ensemble_output)
  return answer_list

# 使用者介面 UI
st.title('公司與事業部輔助判斷模組')
st.image('./bert.png')
tag = st.text_input('請輸入預測結果保存檔案名稱')

# 上傳檔案 csv or excel file
st.text('請上傳csv或xlsx格式的檔案')
test_df = st.file_uploader("upload file", type={"csv", "xlsx"})
if test_df is not None:
    try:
        test_df = pd.read_csv(test_df,index_col=0).reset_index(drop=True)
    except:
        test_df = pd.read_excel(test_df,index_col=0).reset_index(drop=True)

# 渲染輸入資料
st.write('渲染輸入資料')
st.write(test_data)

# 預測程序
if st.button('predict'):
    # 開始記錄時間
    start_time = time.time()
    
    # 準備訓練資料
    database = 抽特徵(預處理(database))
    
    # 準備測試資料
    test_data = 抽特徵(預處理(test_data))

    # 尋找最相近前案的EXPNO
    test_data['預測EXPNO'] = 推論函數(database_size=len(database),test_data=test_data)

    # 計算消費時間
    cost_time = time.time() - start_time
    st.write(f'推論花費時間:{cost_time}')

    # 渲染輸出資料
    st.write('渲染輸出資料')
    st.write(test_data)

    # 渲染正確率
    acc = accuracy_score(
        test_data['推薦公司事業部'].values.tolist(),
        test_data['預測EXPNO'].values.tolist()
        )
    st.write(f'正確率:{acc}')
    
    # 保存結果到指定資料夾
    save_path = f'./predict_result/{tag}.xlsx'
    test_data.to_excel(save_path)
    st.write(f'檔案已自動保存至{save_path}裡面')
