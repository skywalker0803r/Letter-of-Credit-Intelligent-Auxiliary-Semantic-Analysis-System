import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 讀取押匯文件CSV
df_押匯文件 = pd.read_csv('data/preprocess_for_SQUAD_押匯文件.csv')

# 製作寶典
df = pd.read_excel('./data/20210831 押匯文件清單- 202007-202106.老師.xlsx')[['文件類別:','可能的說法']]

for i in df.columns:
    df[i] = df[i].apply(lambda x:x.replace('\n',''))

def string2list(s):
    return [ i.replace('\n','') for i in s.split('、')]

df['可能的說法'] = df['可能的說法'].apply(string2list)

def create_mapping(df):
    寶典 = {}
    for idx in df.index:
        d = {}
        for p in df.loc[idx,'可能的說法']:
            if p != '':
                d[p.strip()] = df.loc[idx,'文件類別:'].replace(' ','')
        寶典.update(d)
    return 寶典
寶典 = create_mapping(df)
del 寶典['I/PMARINE INSURANCE POLICY']

# 段落切分函數
def split_x(x,n=20):
    results = []
    if '+1)' in x:
        for i in range(1,n):
            text = x[x.find(f'+{i})'):x.find(f'+{i+1})')]
            if len(text) > 5:
                results.append(str(text))
    
    elif '1)' in x:
        for i in range(1,n):
            text = x[x.find(f'{i})'):x.find(f'{i+1})')]
            if len(text) > 5:
                results.append(str(text))
    
    elif '1.' in x:
        for i in range(1,n):
            text = x[x.find(f'{i}.'):x.find(f'{i+1}.')]
            if len(text) > 5:
                results.append(str(text))
    
    elif '+' in x:
        for text in x.split('+'):
            if len(text) > 5:
                results.append('+'+str(text))
    
    elif '(A)' in x:
        for i,j in zip(string.ascii_uppercase,string.ascii_uppercase[1:]+'A'):
            text = x[x.find(f'({i})'):x.find(f'({j})')]
            if len(text) > 5:
                results.append(str(text))
            
    return results

# 寶典搜索法
from tqdm import tqdm_notebook as tqdm
def Collection_method(x,寶典):
    labels = {}
    for i in tqdm(x):
        products = []
        for p in 寶典:
            if p in i:
                products.append(p) 
            elif ('/' in p) & (len(p)>len('P/L')):
                if p.split('/')[0] in i:
                    products.append(p) 
                if p.split('/')[1] in i:
                    products.append(p) 
        try:
            labels[i] = max(products,key=len)
        except:
            labels[i] = 'not find'
    predict = pd.DataFrame(columns=['string_X_train','predict'])
    predict['string_X_train'] = labels.keys()
    predict['predict'] = labels.values()
    return predict

# 字串轉索引
def str2index(context,string):
    if type(string) != str:
        print(string)
    ys = context.find(string)
    ye = ys + len(string)
    return ys,ye

# 上色函數
def color_output(text_input,text_output):
    ys,ye = str2index(text_input,text_output)
    left = text_input[:ys]
    mid = text_output
    right = text_input[ye:]
    st.markdown(f'<font>{left}</font> <font color="#FF0000">{mid}</font> <font>{right}</font>', 
    unsafe_allow_html=True)

# UI設計
st.title('押匯文件模組')
init_df = df_押匯文件[df_押匯文件['LCNO'] == '6281MLC00000321']
init_value = ' '.join(init_df['string_X_train'].values.tolist())
text_input = st.text_area('輸入文字',value=init_value)
init_value_sp = split_x(text_input)

# 展示切分結果
st.header('段落切分')
for text in init_value_sp:
    st.text(text)

# 預測按鈕
if st.button('預測'):
    st.write('LCNO = {}'.format(init_df.LCNO.values[0]))
    x = init_value_sp
    answer = Collection_method(x,list(寶典.keys()))
    answer['class'] = answer['predict'].map(寶典)
    for i in range(len(answer)):
        text_input = answer.iloc[i]['string_X_train']
        text_output = answer.iloc[i]['predict']
        st.text(answer.iloc[i]['class'])
        color_output(text_input,text_output)
    table = answer[answer != 'not find'].dropna(axis=0)

    # 將關鍵字前N個字元找出來
    def g(x,keyword,n=2):
        crazystring = x[x.find(keyword)-n:x.find(keyword)] # 高機率在關鍵詞前面有數量
        new_crazy = filter(str.isdigit,crazystring) # 只保留數字
        result = ''.join(list(new_crazy))
        if result != '':
            return result
        else:
            return 1 

    # 搜索正本和影本數量
    table['正本'] = table['string_X_train'].apply(lambda x:g(x,'ORIGINALS'))
    table['正本2'] = table['string_X_train'].apply(lambda x:g(x,'ORIGINAL'))
    table['影本'] = table['string_X_train'].apply(lambda x:g(x,'COPIES'))
    table['影本2'] = table['string_X_train'].apply(lambda x:g(x,'COPIE'))
    table['影本3'] = table['string_X_train'].apply(lambda x:g(x,'PHOTOCOPY'))
    
    # update
    table['正本'].update(table['正本2'])
    table['影本'].update(table['影本2'])
    table['影本'].update(table['影本3'])
    
    # 類別 正本 影本
    table2 = answer[['class']]
    table2['正本'] = table['正本']
    table2['影本'] = table['影本']
    st.table(table2.astype(str))

