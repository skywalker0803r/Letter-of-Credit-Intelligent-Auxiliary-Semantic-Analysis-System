import streamlit as st
import pandas as pd
import numpy as np
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
# 0927到1105測試結果正確率:0.926923076923077錯誤筆數:76

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

# rule對出來的產品名若為其他產品名的子集則剔除
def remove_subsets_lists(l):
    max_len_string = max(l) #最長字串
    l2 = l[:]
    for m in l:
        for n in l:
            if set(m).issubset(set(n)) and m != n:
                l2.remove(m)
                break
    # 確保不為空list
    if len(l2) == 0:
        l2 = [max_len_string]
    return l2

# jaccard文本相似度
def get_jaccard_sim(str1, str2):
    a = set(str1.split()) 
    b = set(str2.split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

# Levenshtein Edit Distance PYTHON
def levenshtein(seq1, seq2):
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros ((size_x, size_y))
    for x in range(size_x):
        matrix [x, 0] = x
    for y in range(size_y):
        matrix [0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix [x,y] = min(
                    matrix[x-1, y] + 1,
                    matrix[x-1, y-1],
                    matrix[x, y-1] + 1
                )
            else:
                matrix [x,y] = min(
                    matrix[x-1,y] + 1,
                    matrix[x-1,y-1] + 1,
                    matrix[x,y-1] + 1
                )
    return (matrix[size_x - 1, size_y - 1])

# 針對模型輸入做預處理
def preprocess_45(x):
    x = str(x).upper() # 轉大寫字串
    x = re.sub('[\u4e00-\u9fa5]', '', x) # 去除中文
    x = re.sub(r'[^\w\s]','',x) # 去除標點符號
    x = x.replace('\n', '').replace('\r', '').replace('\t', '') # 換行符號去除
    str.strip(x) # 移除左右空白
    # 去除多重空白
    x = x.replace('   ', ' ')
    x = x.replace('  ', ' ')
    # 出現在頭的 就不可能對到前後加空格的 這種情形要想想怎麼對照(加上左右空白)
    x = ' ' + x + ' '
    return x

# bert 預測法
def model_predict(nlp,df,question='What is the product name?',start_from0=False,x_col='45A',y_col='預測產品'):
    table = pd.DataFrame()
    idx_list = sorted(df.index.tolist())
    my_bar = st.progress(0)
    for percent_complete,i in enumerate(idx_list):
        my_bar.progress(percent_complete/len(idx_list))
        sample = df.loc[[i]]
        string_X_train = sample[x_col].values[0]
        QA_input = {
            'question': question,
            'context': string_X_train
        }
        res = nlp(QA_input)
        if start_from0 == False:
            predict = QA_input['context'][res['start']:res['end']]
        else:
            predict = QA_input['context'][0:res['end']]
        row = pd.DataFrame({y_col:predict},index=[i])
        table = table.append(row)
    table[y_col] = table[y_col].apply(lambda x:[bert_postprocess(x)])
    return [ i[0] for i in table[y_col].values.tolist()] # list of string

# 寶典比對法
def Collection_method(df,產品集合,x_col):
    labels = {}
    labels_max = {}
    my_bar = st.progress(0)
    for percent_complete,i in enumerate(df.index):
        my_bar.progress(percent_complete/len(df))
        products = []
        for p in 產品集合:
            if (str(p) in str(df.loc[i,x_col])) | (get_jaccard_sim(str(p),str(df.loc[i,x_col]))>=0.9): # 模糊比對
                products.append(str(p)) # 加入候選清單
        if len(products) > 0: # 如果有找到產品 
            labels[i] = products # 複數個產品,之後配合公司去篩選出一個
            labels_max[i] = max(products,key=len) # 取長度最長的產品
        else:
            labels[i] = 'not find'
            labels_max[i] = 'not find'
    predict = pd.DataFrame(index=labels.keys(),columns=['預測產品'])
    predict['預測產品'] = labels.values()
    predict['預測產品(取長度最長)'] = labels_max.values()
    predict['預測產品使用方式'] = 'rule'
    return predict

def add_space(x):
    if (' ' not in x)&(len(x)<=5):
        return ' ' + x + ' '
    else:
        return x

def bert_postprocess(x):
    x = str(x)
    x = x.replace('QUANTITY','')
    if 'PACKING' in x: #像這個 有辦法將 packing之後的都幹掉嗎
        x = x[:x.find('PACKING')+len('PACKING')]
    return x

def product_name_postprocess(x):
    x = str(x)
    x = x.replace('-',' ')
    x = x.strip()
    x = add_space(x)
    return x

# 載入訓練好的模型(產品) 簡稱 nlp
# 載入訓練好的模型(開狀人) 簡稱 nlp2
def load_nlp(path,model,tokenizer):
    model.load_state_dict(torch.load(path))
    model.eval()
    nlp = pipeline('question-answering', model=model.to('cpu'), tokenizer=tokenizer)
    return nlp
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")
nlp = load_nlp('./models/Product_Data_SQuAD_model_product.pt',model,tokenizer)
nlp2 = load_nlp('./models/Product_Data_SQuAD_model_開狀人.pt',model,tokenizer)
nlp3 = load_nlp('./models/Product_Data_SQuAD_model_公司.pt',model,tokenizer)
nlp4 = load_nlp('./models/Product_Data_SQuAD_model_銀行.pt',model,tokenizer)

# 上傳測試檔案
st.text('請上傳csv或xlsx格式的檔案')
test_df = st.file_uploader("upload file", type={"csv", "xlsx"})
x_col = '45A' #產品名
x_col2 = '50' #開狀人
x_col3 = '59' #公司名
銀行_col = ['46A','47A','78'] #銀行欄位
tag = st.text_input('請輸入預測結果保存檔案名稱')

# 判斷檔案是哪一種格式
if test_df is not None:
    try:
        test_df = pd.read_csv(test_df,index_col=0).reset_index(drop=True)
    except:
        test_df = pd.read_excel(test_df,index_col=0).reset_index(drop=True)

# 針對45欄位輸入做預處理
st.write(f'test_df.shape:{test_df.shape}')
test_df['45A'] = test_df['45A'].apply(lambda x:preprocess_45(x))
st.text('測試資料')
st.write(test_df)

# 讀取訓練資料(SPEC)
train_df = pd.read_csv('./data/preprocess_for_SQUAD_產品.csv')[['string_X_train','Y_label','EXPNO']]
train_df_不加空白版本 = train_df.copy()
train_df['Y_label'] = train_df['Y_label'].apply(lambda x:product_name_postprocess(x))
train_df_不加空白版本['Y_label'] = train_df_不加空白版本['Y_label'].apply(lambda x:str(x).replace('-',' ').strip()) #品名後處理

# 讀取台塑網提供之(寶典人工手動修正過刪除線問題)
root = './data/寶典/寶典人工處理後/'

df5 = pd.read_excel(root+'寶典.v6.20211020.xlsx',engine='openpyxl')[['CODIV','DIVNM','ITEMNM']]
df5 = df5.rename(columns={'ITEMNM':'品名','DIVNM':'公司事業部門','CODIV':'公司代號'})

# 我做的寶典
df_by_ricky = pd.read_excel(root+'寶典_by_ricky.xlsx',engine='openpyxl')[['CODIV','DIVNM','ITEMNM']]
df_by_ricky = df_by_ricky.rename(columns={'ITEMNM':'品名','DIVNM':'公司事業部門','CODIV':'公司代號'})

# 廠區回饋
feedback = pd.read_excel(root+'寶典_feedback.xlsx',engine='openpyxl')[['公司代號','公司事業部門','品名']]

#df = df1.append(df2).append(df3).append(df4).append(df5) # 合併所有寶典
df = df5.append(df_by_ricky) # 合併官方寶典和我做的寶典和廠區回饋
df_不加空白版本 = df.copy()
df['品名'] = df['品名'].apply(lambda x:product_name_postprocess(x)) #品名後處理
df_不加空白版本['品名'] = df_不加空白版本['品名'].apply(lambda x:str(x).replace('-',' ').strip()) #品名後處理

# 讀取開狀人寶典,尾綴
開狀人寶典 = pd.read_csv('./data/寶典/開狀人寶典.csv')
開狀人尾綴 = pd.read_csv('./data/寶典/開狀人尾綴.csv')

# 讀取公司寶典,尾綴
公司寶典 = pd.read_csv('./data/寶典/公司寶典加尾綴.csv')
assert len(公司寶典) == 28 #公司名寶典不要擴充

# 製作產品集合(寶典+SPEC)
產品集合 = set(df['品名'].values.tolist() + train_df['Y_label'].values.tolist())
產品集合_不加空白版本 = set(df_不加空白版本['品名'].values.tolist() + train_df_不加空白版本['Y_label'].values.tolist())

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

# 讀取銀行列表
銀行列表 = np.load('./data/寶典/銀行寶典.npy')

# 主UI設計
st.title('公司與事業部輔助判斷模組')
st.image('./bert.png')
button = st.button('predict')

# 推論按鈕
if button:
    debug_mode = False
    
    # 先用規則
    text_output = Collection_method(test_df, 產品集合 ,x_col)
    
    # 若規則無解則改一下產品集合(不加空白)
    not_find_idx = text_output.loc[text_output['預測產品'] == 'not find',:].index
    if len(not_find_idx) > 0:
        text_output.loc[not_find_idx,text_output.columns] = Collection_method(test_df.loc[not_find_idx], 產品集合_不加空白版本 ,x_col)
    
    # 若還是無解則用bert
    not_find_idx = text_output.loc[text_output['預測產品'] == 'not find',:].index
    if len(not_find_idx) > 0:
        bert_predict = model_predict(nlp,test_df.loc[not_find_idx])
        text_output.loc[not_find_idx,'預測產品'] = [ [i] for i in bert_predict]
        text_output.loc[not_find_idx,'預測產品(取長度最長)'] = bert_predict
        text_output.loc[not_find_idx,'預測產品使用方式'] = 'bert'
    
    # 對應部門別和代號,就算匹配不到一模一樣的,取最相似的,少了品名2部門訓練資料 使用find_department函數取代之
    def map2部門(x):
        if x in 品名2部門寶典.keys(): #先從寶典找
            return str(品名2部門寶典[x])
        elif x in train_df['Y_label'].values.tolist(): #找不到從訓練資料找
            return str(find_department(x))
        else:# 模糊比對
            jacs = {}
            for i in 品名2部門寶典.keys():
                jacs[i] = get_jaccard_sim(x,i)
            x = max(jacs,key=jacs.get) # 模糊比對
            return map2部門(x)
    
    def map2代號(x):
        if  x in 品名2代號寶典.keys(): #先從寶典找
            return str(品名2代號寶典[x])
        elif x in 品名2代號訓練資料.keys(): #找不到從訓練資料找
            return str(品名2代號訓練資料[x])
        else:# 模糊比對
            jacs = {}
            for i in 品名2代號寶典.keys():
                jacs[i] = get_jaccard_sim(x,i)
            x = max(jacs,key=jacs.get) # 模糊比對
            return map2代號(x)
    
    # 利用產品名去對應部門跟代號
    text_output['預測產品'] = text_output['預測產品'].apply(remove_subsets_lists)#對出來的產品名若為其他產品名的子集則剔除
    text_output['根據產品預測部門'] = [[map2部門(i) for i in lst] for lst in text_output['預測產品'].values]
    text_output['根據產品預測代號'] = [[map2代號(i) for i in lst] for lst in text_output['預測產品'].values]
    text_output = pd.concat([test_df,text_output.iloc[:,:]],axis=1)
    col_45A = text_output['45A'].values.tolist()
    text_output = text_output.drop(['45A'],axis=1)
    text_output.insert(0, x_col, col_45A)

    # 開狀人的預測結果可以先試著也加進去====================================================================================
    def preprocess_50(x):
        x = str(x)
        x = re.sub('[\u4e00-\u9fa5]', '', x) # 去除中文
        x = re.sub(r'[^\w\s]','',x) # 去除標點符號
        x = x.replace('\n', '').replace('\r', '').replace('\t', '') # 換行符號去除
        return str.strip(x) # 移除左右空白
    
    def predict_Applicant(df=text_output,x_col=x_col2):
        df['50'] = df['50'].apply(lambda x:preprocess_50(x))
        df['預測開狀人'] = 'not find'
        for i in df.index:
            x = df.loc[i,x_col]
            # 1寶典匹配法
            for a in 開狀人寶典['開狀人'].values.tolist():
                if (a in x) & (df.loc[i,'預測開狀人']=='not find'):
                    df.loc[i,'預測開狀人'] = a
            # 2尾綴匹配法
            for b in 開狀人尾綴['尾綴'].values.tolist():
                if (str(b) in str(x)) & (df.loc[i,'預測開狀人']=='not find'):
                    df.loc[i,'預測開狀人'] = x[:x.find(b)+len(b)]
        # 若 1,2 方法都不行則用bert
        not_find_idx = df.loc[df['預測開狀人'] == 'not find',:].index
        if len(not_find_idx) > 0:
            bert_predict = model_predict(
                nlp2,
                df.rename(columns={x_col:'string_X_train'}).loc[not_find_idx],
                question='What is the Applicant name?',
                start_from0=True)
            df.loc[not_find_idx,'預測開狀人'] = bert_predict
        return df
    text_output = predict_Applicant(df=text_output,x_col=x_col2)
    # 開狀人的預測結果可以先試著也加進去====================================================================================

    # 公司的預測結果也可以試著加進去========================================================================================
    def preprocess_59(x): # 公司59欄位預處理
        x = str(x) #轉str
        x = re.sub('[\u4e00-\u9fa5]', '', x) # 去除中文
        x = re.sub(r'[^\w\s]','',x) # 去除標點符號
        x = x.replace('\n', '').replace('\r', '').replace('\t', '') # 去除換行符號
        x = x.replace('r','').replace('n','')
        return str.strip(x)
    
    def predict_divisions(df=text_output,x_col=x_col3):
        df['59'] = df['59'].apply(lambda x:preprocess_59(x))
        div表 = pd.read_csv('./data/寶典/DIVSION映射表.csv')
        df['DIVSION'] = 'not find'
        for i in df.index:
            x = df.loc[i,x_col]
            for a in div表['59'].values:
                if (str(a) in str(x)) & (df.loc[i,'DIVSION'] == 'not find'):
                    df.loc[i,'DIVSION'] = a
        def div映射代號(x):
            jacs = {'not find':-999}
            for i in div表.index:
                jacs[div表.loc[i,'EXPNO']] = get_jaccard_sim(x,div表.loc[i,'59'])
            return max(jacs,key=jacs.get)
        df['DIVSION預測代號'] = [ div映射代號(i) for i in df['DIVSION'].values]
        df.loc[df['DIVSION']=='not find','DIVSION預測代號'] = 'not find'
        return df


    def predict_company(df=text_output,x_col=x_col3):
        df['59'] = df['59'].apply(lambda x:preprocess_59(x))
        df['受益人'] = 'not find'
        for i in df.index:
            x = df.loc[i,x_col]
            # 1寶典匹配法
            for a in 公司寶典['公司英文名稱'].values.tolist():
                if (a in x) & (df.loc[i,'受益人'] == 'not find'):
                    df.loc[i,'受益人'] = a
            # 2尾綴匹配法
            for b in 公司寶典['尾綴'].values.tolist():
                if (str(b) in str(x)) & (df.loc[i,'受益人'] == 'not find'):
                    df.loc[i,'受益人'] = x[:x.find(b)+len(b)]
        # 若 1,2 方法都不行則用bert
        not_find_idx = df.loc[df['受益人'] == 'not find',:].index
        if len(not_find_idx) > 0:
            bert_predict = model_predict(
                nlp3, #nlp3(公司)
                df.rename(columns={x_col:'string_X_train'}).loc[not_find_idx],
                question = 'What is the company name?',
                start_from0 = True)
            df.loc[not_find_idx,'受益人'] = bert_predict
        
        #模糊比對
        def 公司映射代號(公司英文名稱):
            levens = {}
            for idx in 公司寶典.index:
                levens[公司寶典.loc[idx,'代號']] = levenshtein(公司英文名稱,公司寶典.loc[idx,'公司英文名稱']) #公司的模糊比對
            Threshold = 3*2 # 代表替換"n"次字元可以讓兩個字串一致
            if min(levens.values()) <= Threshold:
                return min(levens,key=levens.get)
            else:
                return 'not find'
        df['利用公司名稱預測公司代號'] = [公司映射代號(公司英文名稱) for 公司英文名稱 in df['受益人'].values]
        return df
    
    text_output = predict_company(df=text_output,x_col=x_col3)
    text_output = predict_divisions(df=text_output,x_col=x_col3)

    text_output['集成預測代號'] = 'not find'
    for idx in text_output.index:
        公司預測代號 = str(text_output.loc[idx,'利用公司名稱預測公司代號'])
        產品預測代號列表 = text_output.loc[idx,'根據產品預測代號'].copy()
        DIVSION預測代號 = str(text_output.loc[idx,'DIVSION預測代號'])
        DIVSION = str(text_output.loc[idx,'DIVSION'])
        try:
            if 公司預測代號.isalpha(): # 例如"RS"
                text_output.loc[idx,'集成預測代號'] = 公司預測代號
                continue

            if 公司預測代號 == 'not find': # 直接取眾數
                text_output.loc[idx,'集成預測代號'] = stats.mode(產品預測代號列表)[0][0]
                continue

            # 判斷第一碼做初步篩選,再取眾數
            for 產品預測代號 in 產品預測代號列表:
                assert (len(產品預測代號) == 2) & (type(產品預測代號) == type('string'))
                if 產品預測代號[0] != 公司預測代號[0]: #看產品代號第一碼跟公司預測代號第一碼有沒有一致
                    產品預測代號列表 = list( set(產品預測代號列表)-set([產品預測代號]))
            if len(產品預測代號列表) != 0: # 如果有找到產品
                text_output.loc[idx,'集成預測代號'] = stats.mode(產品預測代號列表)[0][0] # 從候選清單取眾數
            else: # 否則用公司代號assign
                text_output.loc[idx,'集成預測代號'] = 公司預測代號
        except Exception as e: #異常處理
            st.write(e)
            text_output.loc[idx,'集成預測代號'] = 公司預測代號
    
    #==================銀行預測部分==================================================================
    def preprocess_銀行(x):
        x = str(x) # 0.轉字串
        x = re.sub('[\u4e00-\u9fa5]', '', x) # 1.去除中文
        x = re.sub('[’!"#$%&\'()*+,/:;<=>?@[\\]^_`{|}~，。,.]', '', x) # 2.去除標點符號
        x = x.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ') # 3.去除換行符號
        x = x.replace('x000D','') # 4.移除'x000D'
        x = ' ' + str.strip(x) + ' ' # 5.移除左右空白 在左右各加一格空白
        return x
    # 根據寶典找銀行
    def get_bank_寶典(x,寶典):
        for p in 寶典:
            if p in x:
                return p
        return 'not find'

    def predict_bank(df=text_output,x_col=銀行_col):
        df['銀行輸入'] = df[銀行_col[0]] + ' ' + df[銀行_col[1]] + ' ' + df[銀行_col[2]]
        df['開狀銀行'] = 'not find'
        for i in df.index:
            x = df.loc[i,'銀行輸入']
            # 先試寶典匹配法
            for a in 銀行列表:
                if (a in x) & (df.loc[i,'開狀銀行'] == 'not find'):
                    df.loc[i,'開狀銀行'] = a
        # 若寶典匹配不到則用bert
        not_find_idx = df.loc[df['開狀銀行'] == 'not find',:].index
        if len(not_find_idx) > 0:
            bert_predict = model_predict(
                nlp4, #nlp4(銀行)
                df.rename(columns={'銀行輸入':'string_X_train'}).loc[not_find_idx],
                question = 'What is the bank name?',
                start_from0 = False)
            df.loc[not_find_idx,'開狀銀行'] = bert_predict
        return df
    text_output = predict_bank(df=text_output,x_col=銀行_col)
    #==================銀行預測部分==================================================================
    # 計算正確與否
    correct = [ i==j for i,j in zip(text_output['集成預測代號'].values.tolist(),text_output['推薦公司事業部'].values.tolist())]
    text_output['正確與否'] = [ 'yes' if i == True else 'no' for i in correct]
    text_output['錯誤原因'] = '無錯誤'
    text_output.loc[text_output['正確與否']=='no','錯誤原因'] = '訓練使用的數據跟此份測試資料的代號不一致(可能還需釐清廠方提供數據是否有錯誤)'
    text_output.loc[text_output['根據產品預測部門']=='寶典裡沒有','錯誤原因'] = '寶典裡找不到,因此調用bert預測,預測出的產品在寶典裡沒有'

    #======================找對應的EXPNO==========================================================
    if debug_mode == False:
        text_output['EXPNO'] = 'not find'
        EXPNO對應表 = pd.read_csv('.\data\對應表\EXPNO對應表.csv')
        my_bar = st.progress(0)
        for percent_complete,i in enumerate(text_output.index):
            my_bar.progress(percent_complete/len(text_output))
            產品 = text_output.loc[i,'預測產品(取長度最長)']
            開狀人 = text_output.loc[i,'預測開狀人']
            受益人 = text_output.loc[i,'受益人']
            開狀銀行 = text_output.loc[i,'開狀銀行']
            jac = {}
            for j in EXPNO對應表.index:
                jac[j] = get_jaccard_sim(str(產品),str(EXPNO對應表.loc[j,'產品名']))+\
                    get_jaccard_sim(str(開狀人),str(EXPNO對應表.loc[j,'開狀人']))+\
                        get_jaccard_sim(str(受益人),str(EXPNO對應表.loc[j,'受益人']))+\
                            get_jaccard_sim(str(開狀銀行),str(EXPNO對應表.loc[j,'開狀銀行']))
            max_jac_idx = max(jac,key=jac.get)
            text_output.loc[i,'EXPNO'] = str(EXPNO對應表.loc[max_jac_idx,'EXPNO'])
    #==================================================================================================

    # 展示結果
    if debug_mode == True:
        st.write('==================================')
        st.write(text_output.loc[text_output['錯誤原因']!='無錯誤',['受益人','預測產品','預測產品(取長度最長)',
        '推薦公司事業部','根據產品預測代號','利用公司名稱預測公司代號','DIVSION','DIVSION預測代號','集成預測代號']])
        st.write('==================================')
    else:
        st.write('==================================')
        st.write(text_output)
        st.write('==================================')


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

    def save_color_df(df,save_path,x_cols=['45A','50','59','銀行輸入'],y_cols=['預測產品(取長度最長)','預測開狀人','受益人','開狀銀行']):
        # 建立writer
        writer = pd.ExcelWriter(save_path, engine='xlsxwriter')
        # 將 df 第一個 row 變成欄位名稱
        new_df = pd.DataFrame()
        for i in df.columns:
            new_df[i] = [i] + df[i].values.tolist() 
        df = new_df
        # 存檔
        df.to_excel(writer, sheet_name='Sheet1', header=False, index=False)
        # 參數設定
        workbook  = writer.book
        worksheet = writer.sheets['Sheet1']
        cell_format_red = workbook.add_format({'font_color': 'red'})
        cell_format_default = workbook.add_format({'bold': False})
        worksheet.write_row('A1',df.columns.tolist())
        # 定義反紅功能函數
        def add_word_color(df,x_col,y_col):
            for row in range(0,len(df)):
                word = df.iloc[row,:][y_col]
                detect_col_idx = df.columns.tolist().index(x_col)
                try:
                    # 1st case, wrong word is at the start and there is additional text
                    if (df.iloc[row,detect_col_idx].index(word) == 0) \
                    and (len(df.iloc[row,detect_col_idx]) != len(word)):
                        worksheet.write_rich_string(row,detect_col_idx,cell_format_red,
                            word,cell_format_default,df.iloc[row,detect_col_idx][len(word):])

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
        
        # 執行多次反紅功能函數
        for x,y in zip(x_cols,y_cols):
            add_word_color(df,x,y)
        
        # 存檔
        writer.save()
    
    # 展示正確率
    def get_acc(df):
        correct ,correct_label = [] ,[]
        for i in df.index:
            if df.loc[i,'推薦公司事業部'] == df.loc[i,'集成預測代號']:
                correct.append('yes')
            else:
                correct.append('no')
        result = pd.DataFrame({'correct':correct})
        return result['correct'].value_counts()['yes']/len(result)
    st.write(f'正確率:{get_acc(text_output)}')
    錯誤筆數 = len(text_output.loc[text_output['正確與否']=='no',:])
    st.write(f'錯誤筆數:{錯誤筆數}')
    ignore_error_text_output = text_output.loc[text_output['錯誤原因'] != '訓練使用的數據跟此份測試資料的代號不一致(可能還需釐清廠方提供數據是否有錯誤)']
    st.write(f'忽略訓練使用的數據跟此份測試資料的代號不一致的問題後正確率:{get_acc(ignore_error_text_output)}')
    
    # 保存結果到資料夾
    folder = './data/測試結果/'
    if not os.path.exists(folder):
        os.makedirs(folder)
    save_path = f'./predict_result/{tag}.xlsx'
    save_color_df(text_output,save_path)
    #text_output.to_excel(save_path)
    st.write(f'檔案已自動保存至{save_path}裡面')