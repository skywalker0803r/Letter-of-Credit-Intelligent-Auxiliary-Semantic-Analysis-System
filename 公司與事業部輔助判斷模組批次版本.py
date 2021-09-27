import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
import random
from pytorch_lightning import seed_everything
import os
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForQuestionAnswering
from transformers import pipeline
import warnings;warnings.simplefilter('ignore')

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

# bert 預測法
def model_predict(nlp,df,question='What is the product name?',start_from0=False):
    table = pd.DataFrame()
    idx_list = sorted(df.index.tolist())
    my_bar = st.progress(0)
    for percent_complete,i in enumerate(idx_list):
        my_bar.progress(percent_complete/len(idx_list))
        sample = df.loc[[i]]
        string_X_train = sample['string_X_train'].values[0]
        QA_input = {
            'question': question,
            'context': string_X_train
        }
        res = nlp(QA_input)
        if start_from0 == False:
            predict = QA_input['context'][res['start']:res['end']]
        else:
            predict = QA_input['context'][0:res['end']]
        row = pd.DataFrame({'predict':predict},index=[i])
        table = table.append(row)
    return table['predict'].values[0]

# 暴力搜索法
def Collection_method(df,產品集合):
    labels = {}
    my_bar = st.progress(0)
    for percent_complete,i in enumerate(df.index):
        my_bar.progress(percent_complete/len(df))
        products = []
        for p in 產品集合:
            if p in df.loc[i,'string_X_train']:
                products.append(p) # 加入候選清單
        try:
            labels[i] = max(products,key=len) # 候選清單中取最長的
        except:
            labels[i] = 'not find'
    predict = pd.DataFrame(index=labels.keys(),columns=['predict'])
    predict['predict'] = labels.values()
    predict['method'] = 'rule'
    return predict

# 載入模型
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")
model.load_state_dict(torch.load('./models/Product_Data_SQuAD_model_product.pt'))
model.eval()
nlp = pipeline('question-answering', model=model.to('cpu'), tokenizer=tokenizer)

# 上傳測試檔案
st.text('請上傳csv或xlsx格式的檔案')
test_df = st.file_uploader("upload file", type={"csv", "xlsx"})
x_col = st.text_input('請輸入X欄位名稱 提供給模型推論使用 例如:45A')
tag = st.text_input('請輸入預測結果保存檔案名稱')
if test_df is not None:
    try:
        test_df = pd.read_csv(test_df)
    except:
        test_df = pd.read_excel(test_df)
test_df = test_df.rename(columns={x_col:'string_X_train'})
st.text('測試資料')
st.write(test_df)
# 讀取訓練資料
train_df = pd.read_csv('./data/preprocess_for_SQUAD_產品.csv')[['string_X_train','Y_label']]

# 讀取寶典
df1 = pd.read_excel('./data/台塑企業_ 產品寶典20210303.xlsx',engine='openpyxl').iloc[:,:-1]
df2 = pd.read_excel('./data/寶典.v3.台塑網.20210901.xlsx',engine='openpyxl')
df2.columns = df1.columns
df = df1.append(df2)
df['品名'] = df['品名'].apply(lambda x:x.strip())

# 製作產品集合
產品集合 = set(df['品名'].values.tolist() + train_df['Y_label'].values.tolist())
品名2部門 = dict(zip(df['品名'],df['公司事業部門']))
品名2代號 = dict(zip(df['品名'],df['公司代號']))

# UI設計
st.title('公司與事業部輔助判斷模組')
st.image('./bert.png')
button = st.button('predict')

# 推論按鈕
if button:
    # 先用規則
    text_output = Collection_method(test_df, 產品集合)
    # 若規則無解則用bert
    not_find_idx = text_output.loc[text_output['predict'] == 'not find',:].index
    if len(not_find_idx) > 0:
        text_output.loc[not_find_idx,'predict'] = model_predict(nlp,test_df.loc[not_find_idx])
        text_output.loc[not_find_idx,'method'] = 'bert'
    
    # 對應部門別和代號
    def g(x):
        try:
            return str(品名2部門[x])
        except:
            return '找不到對應部門'
    def f(x):
        try:
            return str(品名2代號[x])
        except:
            return '找不到對應代號'
    
    text_output['部門'] = [g(i) for i in text_output['predict'].values]
    text_output['代號'] = [f(i) for i in text_output['predict'].values]
    text_output.insert(0, x_col, test_df['string_X_train'].values.tolist())

    # 改顏色
    def change_color(a):
        d = {}
        for k in text_output['predict'].values.tolist():
            d[k] = 'red'
        d1 = {k: 'background-color:' + v for k, v in d.items()}
        tdf = pd.DataFrame(index=a.index, columns=a.columns)
        tdf = a.applymap(d1.get).fillna('')
        return tdf
    
    def str2index(context,string):
        if type(string) != str:
            print(string)
        ys = context.find(string)
        ye = ys + len(string)
        return ys,ye

    def color_output(text_input,text_output):
        ys,ye = str2index(text_input,text_output)
        left = text_input[:ys]
        mid = text_output
        right = text_input[ye:]
        st.markdown(f'<font>{left}</font> <font color="#FF0000">{mid}</font> <font>{right}</font>', 
        unsafe_allow_html=True)
    
    for i in text_output.index:
        color_output(text_output.loc[i,x_col], text_output.loc[i,'predict'])

    # 展示結果
    st.write(text_output)
    
    # 保存結果
    folder = './predict_result/'
    if not os.path.exists(folder):
        os.makedirs(folder)
    save_path = f'./predict_result/{tag}.xlsx'
    text_output.to_excel(save_path)
    st.write(f'檔案已自動保存至{save_path}裡面')


    


