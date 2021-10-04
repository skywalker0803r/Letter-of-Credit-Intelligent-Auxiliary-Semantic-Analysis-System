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
import re
from IPython.display import HTML
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

# jaccard文本相似度
def get_jaccard_sim(str1, str2): 
    a = set(str1.split()) 
    b = set(str2.split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

# 針對模型輸入做預處理
def preprocess(x):
    x = str(x)
    x = re.sub('[\u4e00-\u9fa5]', '', x) # 去除中文
    x = re.sub(r'[^\w\s]','',x) # 去除標點符號
    x = x.replace('\n', '').replace('\r', '').replace('\t', '') # 換行符號去除
    return str.strip(x) # 移除左右空白

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
    return table['predict']

# 寶典比對法
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

def add_space(x):
    if (' ' not in x)&(len(x)<=5):
        return f' {x} '
    else:
        return x

# 載入訓練好的模型
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

# 判斷檔案是哪一種格式
if test_df is not None:
    try:
        test_df = pd.read_csv(test_df,index_col=0)
    except:
        test_df = pd.read_excel(test_df,index_col=0)
test_df = test_df.rename(columns={x_col:'string_X_train'})
test_df['string_X_train'] = test_df['string_X_train'].apply(preprocess) # 針對輸入做預處理
st.text('測試資料')
st.write(test_df)

# 讀取訓練資料
train_df = pd.read_csv('./data/preprocess_for_SQUAD_產品.csv')[['string_X_train','Y_label','EXPNO']]
train_df['Y_label'] = train_df['Y_label'].apply(lambda x:x.strip()) #針對SPEC去除左右空白
train_df['Y_label'] = train_df['Y_label'].apply(lambda x:add_space(x))

# 讀取台塑網提供之寶典
df = pd.read_excel('./data/寶典.v4.20211001.xlsx',engine='openpyxl')
df = df.rename(columns={'ITEMNM':'品名','DIVNM':'公司事業部門','CODIV':'公司代號'})
df['品名'] = df['品名'].apply(lambda x:x.strip()) # 針對品名去除左右空白
df['品名'] = df['品名'].apply(lambda x:add_space(x))

# 製作產品集合(寶典+SPEC)
產品集合 = set(df['品名'].values.tolist() + train_df['Y_label'].values.tolist())

# 製作對應表(寶典對部門和代號)
品名2部門寶典 = dict(zip(df['品名'],df['公司事業部門']))
品名2代號寶典 = dict(zip(df['品名'],df['公司代號']))

# 製作對應表(訓練資料對代號)
品名2代號訓練資料 = dict(zip(train_df.dropna(subset=['EXPNO'],axis=0)['Y_label'],train_df.dropna(subset=['EXPNO'],axis=0)['EXPNO']))

# 根據品名從訓練資料搜索EXPNO(代號),然後把EXPNO(代號)代入寶典裡找公司部門
def find_department(x):
    try:
        return df.loc[df['公司代號']==train_df.loc[train_df.Y_label==x,'EXPNO'].dropna().value_counts().sort_values().index[-1],'公司事業部門'].value_counts().sort_values().index[-1]
    except:
        return 'not from_pretrained'

# 主UI設計
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
    
    # 對應部門別和代號,就算匹配不到一模一樣的,取最相似的,少了品名2部門訓練資料 使用find_department函數取代之
    def map2部門(x):
        if x in 品名2部門寶典.keys(): #先從寶典找
            return str(品名2部門寶典[x])
        elif x in train_df['Y_label'].values.tolist(): #找不到從訓練資料找
            return find_department(x)
        else: #再找不到算jac最接近的產品
            jacs = {}
            for p in 產品集合:
                jacs[p] = get_jaccard_sim(x,p)
            x = max(jacs,key=jacs.get)
            # 然後重複一樣動作
            map2部門(x)
    
    def map2代號(x):
        if  x in 品名2代號寶典.keys(): #先從寶典找
            return str(品名2代號寶典[x])
        elif x in 品名2代號訓練資料.keys(): #找不到從訓練資料找
            return str(品名2代號訓練資料[x])
        else: #再找不到算jac最接近的產品
            jacs = {}
            for p in 產品集合:
                jacs[p] = get_jaccard_sim(x,p)
            x = max(jacs,key=jacs.get)
            # 然後重複一樣動作
            map2代號(x)
    
    # 整理一下輸出結果
    text_output['部門'] = [map2部門(i) for i in text_output['predict'].values]
    text_output['代號'] = [map2代號(i) for i in text_output['predict'].values]
    text_output['部門'].fillna('寶典裡沒有',inplace=True)
    text_output['代號'].fillna('寶典裡沒有',inplace=True)
    text_output.insert(0, x_col, test_df['string_X_train'].values.tolist())
    text_output = pd.concat([test_df,text_output.iloc[:,:]],axis=1)
    text_output = text_output.drop(['string_X_train'],axis=1)
    col_45A = text_output['45A'].values.tolist()
    text_output = text_output.drop(['45A'],axis=1)
    text_output.insert(0, x_col, col_45A)
    correct = [ i==j for i,j in zip(text_output['代號'].values.tolist(),text_output['推薦公司事業部'].values.tolist())]
    st.write(text_output)
    text_output['代號'].apply(lambda x:x.replace('nan','not find'))
    text_output['正確與否'] = [ 'yes' if i == True else 'no' for i in correct]
    text_output['錯誤原因'] = '無錯誤'
    text_output.loc[text_output['正確與否']=='no','錯誤原因'] = '訓練使用的數據跟此份測試資料的代號不一致(可能還需釐清廠方提供數據是否有錯誤)'
    text_output.loc[text_output['部門']=='寶典裡沒有','錯誤原因'] = '寶典裡找不到,因此調用bert預測,預測出的產品在寶典裡沒有'

    # 改顏色
    def change_color(a):
        d = {}
        for k in text_output['predict'].values.tolist():
            d[k] = 'red'
        d1 = {k: 'background-color:' + v for k, v in d.items()}
        tdf = pd.DataFrame(index=a.index, columns=a.columns)
        tdf = a.applymap(d1.get).fillna('')
        return tdf
    
    # 在文本中找子字串
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

    def save_color_df(df,save_path):
        writer = pd.ExcelWriter(save_path, engine='xlsxwriter')
        
        # 在第一個row add 欄位名稱
        new_df = pd.DataFrame()
        for i in df.columns:
            new_df[i] = [i] + df[i].values.tolist() 
        df = new_df
        
        df.to_excel(writer, sheet_name='Sheet1', header=False, index=False)
        workbook  = writer.book
        worksheet = writer.sheets['Sheet1']
        cell_format_red = workbook.add_format({'font_color': 'red'})
        cell_format_default = workbook.add_format({'bold': False})
        worksheet.write_row('A1',df.columns.tolist())
        for row in range(0,df.shape[0]):
            word = df.iloc[row,:]['predict']
            detect_col_idx = 0
            try:
                # 1st case, wrong word is at the start and there is additional text
                if (df.iloc[row,detect_col_idx].index(word) == 0) \
                and (len(df.iloc[row,detect_col_idx]) != len(word)):
                    worksheet.write_rich_string(row, detect_col_idx, cell_format_red, word,
                                                cell_format_default,
                                                df.iloc[row,detect_col_idx][len(word):])

                # 2nd case, wrong word is at the middle of the string
                elif (df.iloc[row,detect_col_idx].index(word) > 0) \
                and (df.iloc[row,detect_col_idx].index(word) != len(df.iloc[row,detect_col_idx])-len(word)) \
                and ('Typo:' not in df.iloc[row,detect_col_idx]):
                    starting_point = df.iloc[row,detect_col_idx].index(word)
                    worksheet.write_rich_string(row, detect_col_idx, cell_format_default,
                                        df.iloc[row,detect_col_idx][0:starting_point],
                                        cell_format_red, word, cell_format_default,
                                        df.iloc[row,detect_col_idx][starting_point+len(word):])

                # 3rd case, wrong word is at the end of the string
                elif (df.iloc[row,detect_col_idx].index(word) > 0) \
                and (df.iloc[row,detect_col_idx].index(word) == len(df.iloc[row,detect_col_idx])-len(word)):
                    starting_point = df.iloc[row,detect_col_idx].index(word)
                    worksheet.write_rich_string(row, detect_col_idx, cell_format_default,
                                                df.iloc[row,detect_col_idx][0:starting_point],
                                                cell_format_red, word)

                # 4th case, wrong word is the only one in the string
                elif (df.iloc[row,detect_col_idx].index(word) == 0) \
                and (len(df.iloc[row,detect_col_idx]) == len(word)):
                    worksheet.write(row, detect_col_idx, word, cell_format_red)

            except ValueError:
                continue
        writer.save()

    # 展示結果在網頁上
    #for i in text_output.index:
    #    color_output(text_output.loc[i,x_col], text_output.loc[i,'predict'])
    #st.write(text_output)
    
    # 展示正確率
    def get_acc(df):
        correct = []
        correct_label = []
        for i in df.index:
            target = df.loc[i,'推薦公司事業部']
            predict = df.loc[i,'代號']
            if target == predict:
                correct.append('yes')
            else:
                correct.append('no')
        result = pd.DataFrame({'correct':correct})
        return result['correct'].value_counts()['yes']/len(result)
    st.write(f'正確率:{get_acc(text_output)}')
    ignore_error_text_output = text_output.loc[text_output['錯誤原因'] != '訓練使用的數據跟此份測試資料的代號不一致(可能還需釐清廠方提供數據是否有錯誤)']
    st.write(f'忽略訓練使用的數據跟此份測試資料的代號不一致的問題後正確率:{get_acc(ignore_error_text_output)}')
    
    # 保存結果到資料夾
    folder = './predict_result/'
    if not os.path.exists(folder):
        os.makedirs(folder)
    save_path = f'./predict_result/{tag}.xlsx'
    save_color_df(text_output,save_path)
    #text_output.to_excel(save_path)
    st.write(f'檔案已自動保存至{save_path}裡面')





    


