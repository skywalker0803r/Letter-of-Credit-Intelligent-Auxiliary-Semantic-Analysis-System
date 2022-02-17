import pandas as pd
import streamlit as st
import re
import time

# 基於規則之匹配算法
def matching(sentence,database):
  candidate_list = []
  for word in database:
    if word in sentence: 
      candidate_list.append(word)
  return candidate_list

# 輸入sentence前處理
def preprocess_raw_sentence(x):
    x = str(x).upper() # 轉大寫字串
    x = re.sub('[\u4e00-\u9fa5]', '', x) # 去除中文
    x = re.sub(r'[^\w\s]','',x) # 去除標點符號
    x = x.replace('\n', '').replace('\r', '').replace('\t', '') # 去除換行符號
    str.strip(x) # 移除左右空白
    x = x.replace('   ', ' ')# 去除三重空白
    x = x.replace('  ', ' ')# 去除雙重空白
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

# 預測函數
def predict_keyword(title,test_df,Unrecognized,input_col,database,output_col):
    st.write(title)
    my_bar = st.progress(0)
    for percent_complete,i in enumerate(test_df.index):
        my_bar.progress(percent_complete/len(test_df.index))
        candidate_list = matching(
            sentence = test_df.loc[i,input_col],
            database = set(database) - set(Unrecognized)
            )
        output.loc[i,output_col] = candidate_list

# 讀取產品名資料庫
品名寶典 = pd.read_excel('./data/寶典/寶典人工處理後/寶典.v8.202111202.xlsx',engine='openpyxl')[['CODIV','DIVNM','ITEMNM']]
品名寶典 = 品名寶典.rename(columns={'ITEMNM':'品名','DIVNM':'公司事業部門','CODIV':'公司代號'})
品名寶典['品名'] = 品名寶典['品名'].apply(lambda x:product_name_postprocess(x))

# 讀取開狀人資料庫
開狀人寶典 = pd.read_csv('./data/寶典/開狀人寶典.csv')

# 讀取公司寶典
公司寶典 = pd.read_csv('./data/寶典/公司寶典加尾綴.csv')

# UI
st.title('公司與事業部輔助判斷模組')
st.image('./bert.png')
tag = st.text_input('請輸入預測結果保存檔案名稱')

# 上傳檔案
st.text('請上傳csv或xlsx格式的檔案')
test_df = st.file_uploader("upload file", type={"csv", "xlsx"})
if test_df is not None:
    try:
        test_df = pd.read_csv(test_df,index_col=0).reset_index(drop=True)
    except:
        test_df = pd.read_excel(test_df,index_col=0).reset_index(drop=True)

# 針對模型輸入做預處理
產品名輸入 = '45A' #產品名
開狀人輸入 = '50' #開狀人
受益人輸入 = '59' #受益人
開狀銀行輸入 = 'LTADDRESS.1' #銀行輸入
for i in [產品名輸入,開狀人輸入,受益人輸入]:
    test_df[i] = test_df[i].apply(lambda x:preprocess_raw_sentence(x))

# 渲染輸入資料
st.write('渲染輸入資料')
st.write(test_df)

# 預測程序
button = st.button('predict')
output = pd.DataFrame()
output[產品名輸入] = test_df[產品名輸入]
output['產品名'] = None
output[開狀人輸入] = test_df[開狀人輸入]
output['開狀人'] = None
output[受益人輸入] = test_df[受益人輸入]
output['受益人'] = None
output[開狀銀行輸入] = test_df[開狀銀行輸入]
output['開狀銀行'] = None

if button:
    start_time = time.time()
    
    # 1 預測產品
    predict_keyword(
        title = '正在預測產品',
        test_df = test_df,
        Unrecognized = ['PE','MA','EA','GRADE','INA','PACK','PP','PA','',' '],
        input_col = 產品名輸入,
        database = 品名寶典['品名'].values.tolist(),
        output_col = '產品名',
    )

    # 2.預測開狀人
    predict_keyword(
        title = '正在預測開狀人',
        test_df = test_df,
        Unrecognized = ['',' '],
        input_col = 開狀人輸入,
        database = 開狀人寶典['開狀人'].values.tolist(),
        output_col = '開狀人',
    )

    # 3.預測公司
    predict_keyword(
        title = '正在預測受益人',
        test_df = test_df,
        Unrecognized = ['',' '],
        input_col = 受益人輸入,
        database = 公司寶典['公司英文名稱'].values.tolist(),
        output_col = '受益人',
    )

    # 4.預測銀行
    st.write('正在預測開狀銀行')
    output['開狀銀行'] = output[開狀銀行輸入].apply(lambda x:str(x)[:8])

    # 計算消費時間
    cost_time = time.time() - start_time
    st.write(f'推論花費時間:{cost_time}')

    # 渲染輸出資料
    st.write('渲染輸出資料')
    st.write(output)
    
    # 保存結果到指定資料夾
    save_path = f'./predict_result/{tag}.xlsx'
    output.to_excel(save_path)
    st.write(f'檔案已自動保存至{save_path}裡面')