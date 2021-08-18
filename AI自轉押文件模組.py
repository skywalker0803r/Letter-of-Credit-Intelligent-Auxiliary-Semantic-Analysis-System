import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings;warnings.simplefilter('ignore')
from utils import set_seed,model_predict,Collection_method
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForQuestionAnswering
from transformers import pipeline
import torch
import plotly.express as px
from sklearn.decomposition import PCA
import numpy as np
import re
seed = set_seed(42)

# 函數庫
def preprocess(x):
    x = str(x)
    x = re.sub('[\u4e00-\u9fa5]', '', x) # 1.去除中文
    x = re.sub('[’!"#$%&\'()*+,/:;<=>?@[\\]^_`{|}~，。,.]', '', x) # 2.去除標點符號
    x = x.replace('\n', '').replace('\r', '').replace('\t', '') # 3.去除換行符號
    x = str.strip(x) # 4.移除左右空白
    if 'x000D' in x:
        x = x.replace('x000D','')
    return x

def get_bank(text):
    text = str(text)
    text = preprocess(text)
    keywords = ['TO ORDER OF','TO THEORDER OF','TO THE ORDER OF','TOTHE ORDER OF','TO THE ORDER+OF','TOORDER OF']
    for i in keywords:
        if i in text:
            idx = text.split(i)[1].find('BANK')
            result = preprocess(text.split(i)[1][:idx+len('BANK')])
            if 'BANK' in result:
                return result
            else:
                return None
        else:
            return None

def load_model(url):
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")
    model.load_state_dict(torch.load(url))
    model.eval()
    qa_nlp = pipeline('question-answering', model=model.to('cpu'), tokenizer=tokenizer)
    fe_nlp = pipeline("feature-extraction", model=model.to('cpu'), tokenizer=tokenizer)
    return qa_nlp ,fe_nlp

def get_embedding_batch(train_df,fe_nlp):
    max_text = max(train_df['string_X_train'].values,key=len)
    train_text = train_df['string_X_train'].tolist()
    train_text.append(max_text)
    train_emb = fe_nlp(train_text, padding="longest", truncation=True, max_length=40)
    pca = PCA(n_components=2)
    pca.fit(train_emb)
    train_emb = pca.transform(train_emb)
    return train_emb[:-1,:],pca,max_text

def get_embedding_one(text,fe_nlp,max_text,pca):
    df = pd.DataFrame(index=[0])
    df.loc[0,'string_X_train'] = text
    test_emb = fe_nlp([text,max_text], padding="longest", truncation=True, max_length=40)
    test_emb = pca.transform(test_emb)
    return test_emb[:-1,:]

def product_predict(df,nlp):
    # 自動切換規則或bert
    text_output = Collection_method(df, 產品集合)
    mode = 'rule'
    if str(text_output.values[0][0])== "not find":
        mode = 'bert'
        text_output = model_predict(nlp,df,question='What is the product name?')
    try:
        return text_output.iloc[0,0],mode
    except:
        return text_output,mode

def applicant_predict(df,nlp):
    # 自動切換規則或bert
    print(df)
    if 'x000D' in df['string_X_train'].values[0]:
        text_output = str.strip(df['string_X_train'].values[0].split('x000D')[0])
        mode = 'rule'
    else:
        mode = 'bert'
        text_output = model_predict(nlp,df,question='What is the Applicant?')
    try:
        return text_output.iloc[0,0],mode
    except:
        return text_output,mode

def bank_predict(df,nlp):
    text_output = get_bank(df['string_X_train'].values[0])
    mode = 'rule'
    if text_output == None:
        mode = 'bert'
        text_output = model_predict(nlp,df,question='What is the bank name?')
    try:
        return text_output.iloc[0,0],mode
    except:
        return text_output,mode

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

def get_bert_space(csv,fe,text_input,text_output): 
    
    test_row = csv.head(1).copy()
    test_row['string_X_train'] = text_input
    test_row['Y_label'] = text_output
    
    csv = csv.append(test_row).reset_index(drop=True)
    
    train_emb,pca,max_text = get_embedding_batch(csv,fe)
    
    csv['x'],csv['y'] = train_emb[:,0],train_emb[:,1]
    csv['type'] = ['trainset' for i in range(len(csv)-1)] + ['testset']
    
    return csv

# 載入資料
產品csv = pd.read_csv('./data/preprocess_for_SQUAD_產品.csv',index_col=0).rename(columns={'45A':'string_X_train'})[['string_X_train','Y_label']].head(20)
銀行csv = pd.read_csv('./data/preprocess_for_SQUAD_銀行.csv',index_col=0).rename(columns={'string_X_train':'string_X_train'})[['string_X_train','Y_label']].head(20)
開狀人csv = pd.read_csv('./data/preprocess_for_SQUAD_開狀人.csv',index_col=0).rename(columns={'string_X':'string_X_train'})[['string_X_train','Y_label']].head(20)

# 載入模型
產品qa ,產品fe = load_model('./models/Product_Data_SQuAD_model_產品.pt')
銀行qa ,銀行fe = load_model('./models/Product_Data_SQuAD_model_銀行.pt')
開狀人qa ,開狀人fe = load_model('./models/Product_Data_SQuAD_model_開狀人.pt')

# 載入產品寶典集合
df = pd.read_excel('./data/台塑企業_ 產品寶典20210303.xlsx',engine='openpyxl')
產品集合 = set(df['品名'].values)
品名2部門,品名2代號 = dict(zip(df['品名'],df['公司事業部門'])),dict(zip(df['品名'],df['公司代號']))

# 使用者 輸入區
st.title('AI自轉押文件模組')
st.subheader('產品欄位輸入區')
text_input_45 = st.text_area('45A欄位輸入區')
st.subheader('開狀人欄位輸入區')
text_input_50 = st.text_area('50欄位輸入區')
st.subheader('銀行欄位輸入區')
text_input_46A = st.text_area('46A欄位輸入區')
text_input_47A = st.text_area('47A欄位輸入區')
text_input_78A = st.text_area('78A欄位輸入區')


# 圖片
st.image('./bert.png')

# 預測按鈕
if st.button('predict'):
    
    # 產品
    st.subheader('產品預測結果')
    product,mode = product_predict(pd.DataFrame({'string_X_train':text_input_45},index=[0]),產品qa)
    st.text('預測結果')
    color_output(text_input_45,product)
    st.text(f'預測方式:{mode}')
    try:
        st.text(f'部門:{品名2部門[product]}')
        st.text(f'部門代號:{品名2代號[product]}')
    except:
        st.text(f'找不到所屬部門')

    # 開狀人
    st.subheader('開狀人預測結果')
    applicant,mode = applicant_predict(pd.DataFrame({'string_X_train':text_input_50},index=[0]),開狀人qa)
    st.text('預測結果')
    color_output(text_input_50,applicant)
    st.text(f'預測方式:{mode}')

    # 銀行
    st.subheader('銀行預測結果')
    bank,mode = bank_predict(pd.DataFrame({'string_X_train':text_input_46A},index=[0]),銀行qa)
    st.text('預測結果')
    color_output(text_input_46A,bank)
    st.text(f'預測方式:{mode}')

    # 針對這三個部分各自取出2維表示
    產品plot_data = get_bert_space(產品csv,產品fe,text_input_45,product) # 2dim x,y
    st.write(產品plot_data)
    開狀人plot_data = get_bert_space(開狀人csv,開狀人fe,text_input_50,applicant) # 2dim x,y
    st.write(開狀人plot_data)
    銀行plot_data = get_bert_space(銀行csv,銀行fe,text_input_46A,bank) # 2dim x,y
    st.write(銀行plot_data)
    
    # 更改欄位名稱 避免合併衝突
    產品plot_data.columns = ['產品string_X_train','產品Y_label','產品x','產品y','type']
    開狀人plot_data.columns = ['開狀人string_X_train','開狀人Y_label','開狀人x','開狀人y','type']
    銀行plot_data.columns = ['銀行string_X_train','銀行Y_label','銀行x','銀行y','type']

    #合併變成六維 x1,x2,x3,y1,y2,y3
    merge_plot_data = pd.concat([
        產品plot_data.iloc[:,[2,3]],
        開狀人plot_data.iloc[:,[2,3]],
        銀行plot_data.iloc[:,[2,3]]],axis=1) 

    #六維轉二維
    pca = PCA(n_components=2)
    pca.fit(merge_plot_data)
    xy = pca.transform(merge_plot_data)
    plot_data = pd.DataFrame() 
    plot_data['x'] ,plot_data['y'] = xy[:,0] ,xy[:,1]
    plot_data['產品string_X_train'] = 產品plot_data['產品string_X_train']
    plot_data['產品Y_label'] = 產品plot_data['產品Y_label']
    plot_data['開狀人string_X_train'] = 開狀人plot_data['開狀人string_X_train']
    plot_data['開狀人Y_label'] = 開狀人plot_data['開狀人Y_label']
    plot_data['銀行string_X_train'] = 銀行plot_data['銀行string_X_train']
    plot_data['銀行Y_label'] = 銀行plot_data['銀行Y_label']
    plot_data['type'] = ['trainset' for i in range(len(plot_data)-1)] + ['testset']
    st.write(plot_data)
    
    # 繪圖
    st.plotly_chart(px.scatter(plot_data, x="x", y="y", color="type", hover_data=['產品Y_label','開狀人Y_label','銀行Y_label']))
    







