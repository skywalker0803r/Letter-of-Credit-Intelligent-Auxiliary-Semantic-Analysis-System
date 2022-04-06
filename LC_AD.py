# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 12:23:16 2022

@author: aiuser
"""

# pip install openpyxl
# pip install ipywidgets

import re 
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook as tqdm
from simhash import Simhash
import matplotlib.pyplot as plt
import editdistance

def seed_everything(seed: int):
    import random, os
    import numpy as np
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
seed_everything(42)


# 核心算法
# MSKTS文本相似度比對算法
# url = 'https://gist.githubusercontent.com/skywalker0803r/7c00d680d731b99ab549dd40a96006ce/raw/d9bb060a42285053faa5227df521b43f381f1a0b/MSKTS.py'
# exec(requests.get(url).text)

#from simhash import Simhash

class MSKTS(object):
  '''
  most similar k text search
  '''
  def __init__(self):
    self.name = 'most similar k text search'
  
  def fit(self,database):
    self.database = map(lambda x:str(x).upper(), database)
  
  def predict(self,input_data,k=3):
    input_data = input_data.upper()
    score = {}
    for history_data in self.database:
      score[history_data] = Simhash(input_data).distance(Simhash(history_data))
    return sorted(score.items(), key=lambda x:x[1],reverse=False)[:k]



# help function
# 只保留英文字母
def keep_alpha(str1): 
  char = "" 
  for x in str(str1):
    if x.isalpha(): 
      char = "".join([char, x])
  return char

# 基於規則之關鍵字匹配算法
def matching(sentence,database,use_X000D=False):
  candidate_list = []
  for word in database:
    if word in sentence: 
      candidate_list.append(word)
  if (use_X000D == True) and (len(candidate_list) == 0):
    candidate_list.append(sentence.split('_X000D')[0])
  if len(candidate_list) == 0:
    candidate_list.append('matching函數失效')#1
  return candidate_list

# # string_list中的string若為其他string的"子集"則剔除
# def substringSieve(string_list):
#     string_list.sort(key=lambda s: len(s), reverse=True)
#     out = []
#     for s in string_list:
#       if not any([s in o for o in out]):
#         out.append(s)
#     return out

# string_list中的string若為其他string的"子集"則剔除
def substringSieve(string_list):
    string_list = [item.strip() for item in string_list]
    string_list.sort(key=lambda s: len(s), reverse=True)
    out = []
    for s in string_list:
      if not any([s in o for o in out]):
        out.append(s)
    return out

# 去除多重空白
def remove_multiple_blanks(x):
  for i in range(2,10,1):
    x = x.replace(' '*i,' ')
  return x

# 輸入sentence前處理
def preprocess_raw_sentence(x):
  x = str(x).upper() # 轉大寫字串
  x = re.sub('[\u4e00-\u9fa5]', '', x) # 去除中文
  x = re.sub(r'[^\w\s]','',x) # 去除標點符號
  x = x.replace('\n', '').replace('\r', '').replace('\t', '') # 去除換行符號
  str.strip(x) # 移除左右空白
  x = remove_multiple_blanks(x) # 去除多重空白
  x = ' ' + x + ' '# 出現在頭的 就不可能對到前後加空格的 這種情形要想想怎麼對照(加上左右空白)
  return x

# 字串長度小於3的單詞前後加空白
def add_space(x):
    if (' ' not in x) and (len(x)<=3): 
        return ' ' + x + ' '
    else:
        return x

# 產品後處理
def product_name_postprocess(x):
  x = str(x).split('(')[0] # 擷取括號前面的字串
  x = str(x).upper() # 轉大寫字串
  x = re.sub(r'[^\w\s]','',x) # 去除標點符號
  x = x.strip() # 去除空白
  x = add_space(x)# 字串長度小於3的單詞前後加空白
  return x

# 基於關鍵字比對方法的預測函數
def predict_keyword(title,test_df,Unrecognized,input_col,database,output_col,use_X000D=False):
  result = []
  for i in tqdm(test_df.index):
    candidate_list = matching(
        sentence = test_df.loc[i,input_col],
        database = set(database) - set(Unrecognized),
        use_X000D = use_X000D
        )
    result.append(substringSieve(candidate_list))
  test_df[output_col] = result
  return test_df

# 取得dataframe的空列表索引
def get_empty_list_idx(df,col):
  error_idx = []
  for idx,name in enumerate(df[col].values.tolist()):
    if len(name) == 0:
      error_idx.append(idx)
  return error_idx

# 公司英文名稱模糊比對函數
def 公司英文名稱模糊比對函數(input_data,公司寶典):
    # 去[]
    input_data = input_data[0]
    # 去尾綴
    for 尾綴 in 公司寶典['尾綴']:
      input_data = input_data.replace(尾綴,'')
    # 去空白
    input_data = input_data.strip()
    # 幾種意外情況
    if input_data == 'not find2':#2
      return [input_data]
    if input_data.encode('utf-8').isalpha() == False:
      return [input_data]
    # 模糊搜索最相似公司
    score = {}
    for history_data in 公司寶典['公司英文名稱']:
        score[history_data] = editdistance.eval(input_data,history_data)
    return min(score,key=score.get)

# 公司英文名稱2代號函數
def 公司英文名稱2代號函數(input_data,公司寶典):
    score = {}
    for history_data in 公司寶典['公司英文名稱']:
      score[history_data] = editdistance.eval(input_data,history_data)
    return 公司寶典.loc[公司寶典['公司英文名稱']==min(score,key=score.get),'代號']





# 載入數據
# 歷史資料庫
database = pd.read_excel('data/combined_excel.xlsx')
# 新的測試數據
test_data = pd.read_csv('data/測試數據/0927到2022.csv')

# 讀取"產品名"寶典
# 品名寶典 = pd.read_excel('data/寶典/寶典人工處理後/寶典.v8.202111202.xlsx',engine='openpyxl')[['CODIV','DIVNM','ITEMNM']]
品名寶典 = pd.read_excel('data/寶典/寶典人工處理後/寶典.v9.xlsx',engine='openpyxl')[['CODIV','DIVNM','ITEMNM']]
品名寶典 = 品名寶典.rename(columns={'ITEMNM':'品名','DIVNM':'公司事業部門','CODIV':'公司代號'})
品名寶典['品名'] = 品名寶典['品名'].apply(lambda x:product_name_postprocess(x))

# 讀取"開狀人"寶典
開狀人寶典 = pd.read_csv('data/寶典/開狀人寶典.csv')

# 讀取"公司"寶典
#公司寶典 = pd.read_csv('data/寶典/公司寶典加尾綴.csv',index_col=0).astype(str).reset_index(drop=True)
公司寶典 = pd.read_csv('data/寶典/公司寶典加尾綴v2.csv',index_col=0, encoding='ANSI').astype(str).reset_index(drop=True)






大公司列表 = []
for i in 公司寶典['代號']:
    if (len(i) == 1):
        大公司列表.append(i)
大公司列表


小公司列表 = []
for i in 公司寶典['代號']:
    if (len(i) == 2) and (i[0].isalpha()) and (i[1].isalpha()):
        小公司列表.append(i)
#小公司列表
小公司列表.append('J7')




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
def 抽特徵(df,品名寶典=None,開狀人寶典=None,公司寶典=None):
  # 預測產品(利用品名寶典)
  df = predict_keyword(
      title = '正在預測產品',
      test_df = df,
      Unrecognized = ['PE','MA','EA','GRADE','INA','PACK','PP','PA','']+[' '*i for i in range(1,10,1)],
      input_col = 產品名輸入,
      database = list(map(lambda x:str(x).upper(),品名寶典['品名'].values.tolist())),
      output_col = '產品名',
      use_X000D = False,
      )

  # 預測開狀人(善用X000D)
  df = predict_keyword(
      title = '正在預測開狀人',
      test_df = df,
      Unrecognized = ['']+[' '*i for i in range(1,10,1)],
      input_col = 開狀人輸入,
      database = list(map(lambda x:str(x).upper(),開狀人寶典['開狀人'].values.tolist())),
      output_col = '開狀人',
      use_X000D = True,
      )

  # 搜索法預測公司(受益人)
  df = predict_keyword(
      title = '正在預測受益人',
      test_df = df,
      Unrecognized = ['']+[' '*i for i in range(1,10,1)],
      input_col = 受益人輸入,
      database = list(map(lambda x:str(x).upper(),公司寶典['公司英文名稱'].values.tolist())),
      output_col = '受益人',
      use_X000D = True,
      )
  # 受益人(公司)模糊比對,確保跟寶典上寫的一致
  df['受益人'] = df['受益人'].apply(lambda x:公司英文名稱模糊比對函數(x,公司寶典))

  # 預測開狀銀行靠規則比對篩選前8碼即可
  df['開狀銀行'] = df[開狀銀行輸入].apply(lambda x:str(x)[:8])
  return df



# 準備訓練資料
database = 抽特徵(預處理(database),品名寶典=品名寶典,開狀人寶典=開狀人寶典,公司寶典=公司寶典)
for i in 輸出欄位:
    number = get_empty_list_idx(df=database,col=i)
    print(i,'空列表數量:',len(number))
database[輸出欄位].tail(5)

# 準備測試資料
test_data = 抽特徵(預處理(test_data),品名寶典=品名寶典,開狀人寶典=開狀人寶典,公司寶典=公司寶典)
for i in 輸出欄位:
    number = get_empty_list_idx(df=test_data,col=i)
    print(i,'空列表數量:',len(number))
test_data[輸出欄位].tail(5)










# 模型測試
def 根據特定欄位和索引給出候選答案清單(col,idx,k,database_size=100,database=None,test_data=None):
  # 判斷是否為空列表
  if len(test_data[col][idx]) == 0:
    return []
  # 預處理
  database['處理過的資料'] = (database[col]).apply(keep_alpha)
  test_data['處理過的資料'] = (test_data[col]).apply(keep_alpha)
  # 建立模型
  model = MSKTS()
  model.fit(list(set(database['處理過的資料'].sample(database_size).values.tolist()) - set(['']+[' '*i for i in range(1,10,1)])))
  # 產生預測答案清單
  predict_answer = [i for i in model.predict(test_data['處理過的資料'][idx],k=k)]
  # 預測最相似文本
  預測最相似文本 = [i[0] for i in predict_answer]
  預測EXPNO前兩碼 = database.loc[database['處理過的資料'].isin(預測最相似文本),'EXPNO'].dropna().apply(lambda x:str(x)[:2]).values.tolist()
  # 相似度距離
  相似度距離 = [i[1] for i in predict_answer]
  # 預測完整EXPNO
  預測完整EXPNO = database.loc[database['處理過的資料'].isin(預測最相似文本),'EXPNO'].dropna().apply(lambda x:str(x)[:]).values.tolist()
  # 最相似前案
  最相似前案 = database.loc[database['處理過的資料'].isin(預測最相似文本),col].dropna().apply(lambda x:str(x)[:]).values.tolist()
  # 最相似前案時間
  最相似前案時間 = database.loc[database['處理過的資料'].isin(預測最相似文本),'from'].dropna().apply(lambda x:str(x)[:]).values.tolist()
  return 預測EXPNO前兩碼,相似度距離,預測完整EXPNO,最相似前案,最相似前案時間
issue_idx = test_data['受益人'][test_data['受益人'].apply(lambda x:str(x)[0])=='not find3'].index#3
test_data.loc[issue_idx,['59','受益人']]

test_data[輸出欄位].head()




# 模型測試
def 根據特定欄位和索引給出候選答案清單v2(col,idx,k,database_size=100,database=None,test_data=None):
  # 判斷是否為空列表
  if len(test_data[col][idx]) == 0:
    return []
  # 預處理
  database['處理過的資料'] = (database[col]).apply(keep_alpha)
  test_data['處理過的資料'] = (test_data[col]).apply(keep_alpha)
  # 建立模型
  model = MSKTS()
  model.fit(list(set(database['處理過的資料'].sample(database_size).values.tolist()) - set(['']+[' '*i for i in range(1,10,1)])))
  # 產生預測答案清單
  predict_answer = [i for i in model.predict(test_data['處理過的資料'][idx],k=k)]
  # 預測最相似文本
  預測最相似文本 = [i[0] for i in predict_answer]
  預測EXPNO前兩碼 = database.loc[database['處理過的資料'].isin(預測最相似文本),'EXPNO'].dropna().apply(lambda x:str(x)[:2]).values.tolist()
  # 相似度距離
  相似度距離 = [i[1] for i in predict_answer]
  # 預測完整EXPNO
  預測完整EXPNO = database.loc[database['處理過的資料'].isin(預測最相似文本),'EXPNO'].dropna().apply(lambda x:str(x)[:]).values.tolist()
  # 最相似前案
  最相似前案 = database.loc[database['處理過的資料'].isin(預測最相似文本),col].dropna().apply(lambda x:str(x)[:]).values.tolist()
  # 最相似前案時間
  最相似前案時間 = database.loc[database['處理過的資料'].isin(預測最相似文本),'from'].dropna().apply(lambda x:str(x)[:]).values.tolist()
  return 預測EXPNO前兩碼,相似度距離,預測完整EXPNO,最相似前案,最相似前案時間




def 根據受益人限縮database(database,受益人,公司寶典):
    代號 = 公司英文名稱2代號函數(受益人,公司寶典).values[0]
    cond = database['EXPNO'].apply(lambda x:str(x)[0]) == str(代號)[0]
    return database.loc[cond,:]

A = test_data.loc[0,'受益人'][0]
print(A)
公司英文名稱2代號函數(A,公司寶典).values[0]
根據受益人限縮database(database,A,公司寶典)[['受益人','EXPNO']]

# def 根據受益人限縮品名寶典(品名寶典,受益人,公司寶典):
#     代號 = 公司英文名稱2代號函數(受益人,公司寶典).values[0]
#     cond = 品名寶典['公司代號'].apply(lambda x:str(x)[0]) == str(代號)[0]
#     return 品名寶典.loc[cond,:]

# A = test_data.loc[1,'受益人'][0]
# print(A)
# 公司英文名稱2代號函數(A,公司寶典).values[0]
# 根據受益人限縮品名寶典(品名寶典,A,公司寶典)




def 目標函數(database_size,database,品名寶典,公司寶典,test_data,test_n=20,k=1):
  # 初始化'預測EXPNO'和correct
  test_data['預測EXPNO'] = None
  test_data['正確'] = None
  correct = []
  # 遍歷test_data做推論
  for idx in tqdm(range(test_n)):
    # 先用品名映射到代號
    
    try:
      受益人 = test_data.loc[idx,'受益人'][0]
      代號_受益人 = 公司英文名稱2代號函數(受益人,公司寶典).values[0]
      # 若為小公司就直接給公司事業部代碼
      if 小公司列表.count(代號_受益人) > 0:
        代號 = 代號_受益人
      else: # 若為大公司
        # 先根將萃取之品名與公司事業部代碼抓出來，再判斷其公司事業部代碼的第一碼是否屬該受益人
        #restricted_品名寶典 = 根據受益人限縮品名寶典(品名寶典,受益人,公司寶典)
        # 品名可能會有多個，取其公司事業部代碼的眾數
        品名_tmp = test_data.loc[idx,'產品名']
        代號_tmp = []
        品名 = []
        for i in range(len(品名_tmp)):
            #代號.append(dict(zip(restricted_品名寶典['品名'],restricted_品名寶典['公司代號']))[品名[i]])
            #if dict(zip(品名寶典['品名'],品名寶典['公司代號']))[add_space(品名_tmp[i])][:1] == 代號_受益人:
                #代號_tmp.append(dict(zip(品名寶典['品名'],品名寶典['公司代號']))[品名_tmp[i]])
                #品名.append(品名_tmp[i]) 
            代號_tmp2 = pd.DataFrame(品名寶典.loc[np.where(品名寶典['品名']==add_space(品名_tmp[i])),'公司代號']).reset_index()
            for j in range(len(代號_tmp2)):
                if str(代號_tmp2.loc[j,'公司代號'])[:1] == str(代號_受益人):
                    代號_tmp.append(代號_tmp2.loc[j,'公司代號'])
                    品名.append(品名_tmp[i])           
        代號 = max(set(代號_tmp), key=代號_tmp.count)
        # 有無可能將"品名"這個物件取代特徵萃取所得的test_data.at[idx,'產品名']?或另存在另外一欄"產品名_受益人限縮"
        # if 品名_tmp != 品名:
        #     #test_data.at[idx,'產品名'] = None
        #     test_data.at[idx,'產品名'] = set(品名)
    except:
      代號 = None
      
    # 先根據受益人限縮database
    try:
      restricted_database = 根據受益人限縮database(database,test_data.loc[idx,'受益人'][0],公司寶典)
    except:
      restricted_database = database
    # 根據四個欄位預測答案
    o1,d1,e1,n1,t1 = 根據特定欄位和索引給出候選答案清單(
      col='產品名',idx=idx,k=k,
      database_size = min(database_size,len(restricted_database)),
      database = restricted_database,
      test_data = test_data)
    o2,d2,e2,n2,t2 = 根據特定欄位和索引給出候選答案清單(
      col='開狀人',idx=idx,k=k,
      database_size = min(database_size,len(restricted_database)),
      database = restricted_database,
      test_data = test_data)
    o3,d3,e3,n3,t3 = 根據特定欄位和索引給出候選答案清單(
      col='受益人',idx=idx,k=k,
      database_size = min(database_size,len(restricted_database)),
      database = restricted_database,
      test_data = test_data)
    o4,d4,e4,n4,t4 = 根據特定欄位和索引給出候選答案清單(
      col='開狀銀行',idx=idx,k=k,
      database_size = min(database_size,len(restricted_database)),
      database = restricted_database,
      test_data = test_data)
    # 判斷是否為小公司
    if len(set(o3) & set(小公司列表)) > 0:
      o = list(set(o3) & set(小公司列表))
    # 判斷是否為大公司
    elif len(set([str(i[0]) for i in o3]) & set(大公司列表)) > 0:
      if len([i for i in o1 if str(i[0]) in 大公司列表]) > 0: 
        o = [i for i in o1 if str(i[0]) in 大公司列表]
      else:
        o = list(set(o3) & set(大公司列表))
    # 其他情況
    else:
      o = o1 + o2 + o3 + o4
    # 對o取眾數得到ensemble_output
    try:
      try:
        ensemble_output = max(o,key=o.count)
      except:
        ensemble_output = max(o1+o2+o3+o4,key=(o1+o2+o3+o4).count)
    except:
      ensemble_output = 'not find4'#4
    # 如果代號 != None 指派代號至ensemble_output
    if str(代號) != 'nan':
      ensemble_output = 代號
    # 指派前案特徵至test_data
    for i in [n1,n2,n3,n4]:
      if len(i) == 0:
        i.append('最相似前案dropna後是空值')#5
    test_data.loc[idx,'前案產品名'] = max(n1,key=n1.count)
    test_data.loc[idx,'前案開狀人'] = max(n2,key=n2.count)
    test_data.loc[idx,'前案受益人'] = max(n3,key=n3.count)
    test_data.loc[idx,'前案開狀銀行'] = max(n4,key=n4.count)
    # 指派前案時間至test_data
    try:
      test_data.loc[idx,'最相似前案時間'] = max(t1+t2+t3+t4,key=(t1+t2+t3+t4).count)
    except:
      test_data.loc[idx,'最相似前案時間'] = '最相似前案時間dropna後是空值'
    # 指派預測值至test_data
    test_data.loc[idx,'預測EXPNO'] = ensemble_output
    # 指派距離至test_data
    test_data.loc[idx,'相似度距離'] = np.sum(d1+d2+d3+d4)
    # 指派完整EXPNO至test_data
    完整EXPNO候選清單 =[]
    for expno in e1+e3+e3+e4:
      if expno[:2] == ensemble_output:
        完整EXPNO候選清單.append(expno)
    try:
      test_data.loc[idx,'預測完整EXPNO'] = max(完整EXPNO候選清單,key=完整EXPNO候選清單.count)
    except:
      try:
        test_data.loc[idx,'預測完整EXPNO'] = database.loc[database['EXPNO'].apply(lambda x:str(x)[:2]).isin(o1+o2+o3+o4),'EXPNO'].dropna().sample(1).values.tolist()
      except:
        try:
          test_data.loc[idx,'預測完整EXPNO'] = database.loc[database['EXPNO'].apply(lambda x:str(x)[:2]).isin([ensemble_output]),'EXPNO'].dropna().sample(1).values.tolist()
        except:
          test_data.loc[idx,'預測完整EXPNO'] = None
    # 當"預測EXPNO"是空、且"預測完整EXPNO"非空，用"預測完整EXPNO"前兩碼去補空的"預測EXPNO"
    #if str(test_data.loc[idx,'預測EXPNO']) == 'nan':
      #test_data.loc[idx,'預測EXPNO'] = test_data.loc[idx,'預測完整EXPNO'].apply(lambda x:str(x)[:2]).values.tolist()[0]
    if (test_data.loc[idx,'預測EXPNO'] == None and test_data.loc[idx,'預測完整EXPNO'] != None):
      test_data.loc[idx,'預測EXPNO'] = test_data.loc[idx,'預測完整EXPNO'][:2]
    # # 用預測完整EXPNO去補預測EXPNO
    # test_data['預測EXPNO'] = test_data['預測EXPNO'].astype(str)
    # test_data['預測完整EXPNO'] = test_data['預測完整EXPNO'].astype(str)
    # error_idx = (test_data['預測EXPNO'].apply(lambda x:x[:1]) != test_data['預測完整EXPNO'].apply(lambda x:x[:1])).values
    # test_data.loc[error_idx,'預測EXPNO'] = test_data.loc[error_idx,'預測完整EXPNO'].apply(lambda x:x[:2])
    # 計算正確率
    if ensemble_output == test_data['推薦公司事業部'][idx]:
      correct.append(True)
      test_data.loc[idx,'正確'] = True
    else:
      correct.append(False)
      test_data.loc[idx,'正確'] = False
  return np.mean(correct),test_data


acc,test_data = 目標函數(database_size=len(database),database=database,品名寶典=品名寶典,公司寶典=公司寶典,test_data=test_data,test_n=len(test_data))#test_n=len(test_data))
print('正確率:',acc)


最終所有必須欄位 = 輸入欄位+輸出欄位+['相似度距離','from','20','預測EXPNO','預測完整EXPNO','推薦公司事業部',
    '最相似前案時間','前案產品名','前案開狀人','前案受益人','前案開狀銀行']
# test_data.head(20).loc[test_data['預測EXPNO']!=test_data['推薦公司事業部'],最終所有必須欄位]



#輸出
test_data[最終所有必須欄位].to_excel('predict_result/預測結果.xlsx')




#錯誤確認
test_data.loc[test_data['正確']==False,最終所有必須欄位].to_excel('predict_result/錯誤預測結果.xlsx')
test_data.loc[test_data['正確']==False,最終所有必須欄位]
