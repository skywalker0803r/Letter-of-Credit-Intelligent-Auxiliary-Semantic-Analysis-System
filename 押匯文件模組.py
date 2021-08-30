import streamlit as st
import pandas as pd
import numpy as np
import joblib


df = pd.read_csv('data/preprocess_for_SQUAD_押匯文件.csv')

寶典 = ['BENEFICIARYS DRAFT',
'BILL OF LADING','TRANSPORT DOCUMENTS','B/L','MARINE/OCEAN BILL OF LADING','MARINE BILL OF LADING','OCEAN BILL OF LADING',
'COMMERCIAL INVOICE','INVOICE','INV.',
'PACKING LIST','PACKING','P/L',
'INSURANCE POLICY/CERTIFICATE','I/P',
'MARINE INSURANCE POLICY' ,'MARINE CARGO POLICY','CARGO POLICY','INSURANCE AGENCY',
'CONSULAR DECLARATION','CONSULAR INVOICE','CONSULAGE','CONSULAR LEGALIZED INVOICE',
'CERTIFICATE OF ORIGIN','COO',
'TEST CERTIFICAT/INSPECTION','SURVEY REPORT','CERTIFICATE OF QUALITY','TEST CERTIFICAT','TEST INSPECTION'
'CERTIFICATE OF QUANTITY','COQ','WEIGHT MEMO',
'SHIPPING COMPANY CERTIFICATE','CERTIFICATE FROM SHIPPING COMPANY',
'FUMIGATION CERTIFICATE',
'BENEFICIARYS CERTIFICATE']

p2c = {'BENEFICIARYS DRAFT':'匯票',

'BILL OF LADING':'提單',
'TRANSPORT DOCUMENTS':'提單',
'B/L':'提單',
'MARINE/OCEAN BILL OF LADING':'提單',
'MARINE BILL OF LADING':'提單',
'OCEAN BILL OF LADING':'提單',

'COMMERCIAL INVOICE':'商業發票',
'INVOICE':'商業發票',
'INV.':'商業發票',

'PACKING LIST':'裝箱單',
'PACKING':'裝箱單',
'P/L':'裝箱單',

'INSURANCE POLICY/CERTIFICATE':'保險單據',
'INSURANCE POLICY':'保險單據',
'I/P':'保險單據',
'MARINE INSURANCE POLICY':'保險單據',
'MARINE CARGO POLICY':'保險單據',
'CARGO POLICY':'保險單據',
'INSURANCE AGENCY':'保險單據',

'CONSULAR DECLARATION':'領事簽證',
'CONSULAR INVOICE':'領事簽證',
'CONSULAGE':'領事簽證',
'CONSULAR LEGALIZED INVOICE':'領事簽證',

'CERTIFICATE OF ORIGIN':'產地(商會)證明',
'COO':'產地(商會)證明',

'TEST CERTIFICAT':'檢驗證明',
'TEST INSPECTION':'檢驗證明',
'SURVEY REPORT CERTIFICATE OF QUALITY':'檢驗證明',

'CERTIFICATE OF QUANTITY':'重量證明',
'COQ':'重量證明',
'WEIGHT MEMO':'重量證明',

'SHIPPING COMPANY CERTIFICATE':'船證',
'CERTIFICATE FROM SHIPPING COMPANY':'船證',

'FUMIGATION CERTIFICATE':'薰蒸證明書',

'BENEFICIARYS CERTIFICATE':'受益人證明',

}

def split_x(x,n=20):
    results = []
    
    # case1
    if '+1)' in x:
        for i in range(1,n):
            text = x[x.find(f'+{i})'):x.find(f'+{i+1})')]
            if len(text) > 10:
                results.append(str(text))
    
    # case2
    elif '1.' in x:
        for i in range(1,n):
            text = x[x.find(f'{i}.'):x.find(f'{i+1}.')]
            if len(text) > 10:
                results.append(str(text))
    
    return results

def Collection_method(x,寶典):
    labels = {}
    for i in x:
        products = []
        for p in 寶典:
            if p in i:
                products.append(p)
        try:
            labels[i] = max(products,key=len)
        except:
            labels[i] = 'not find'
    predict = pd.DataFrame(index=labels.keys(),columns=['predict'])
    predict['predict'] = labels.values()
    predict = predict.reset_index()
    predict.columns = ['string_X_train','predict']
    return predict

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

# UI
st.title('押匯文件模組')
init_df = df[df['LCNO'] == '6281MLC00000321']
init_value = ' '.join(init_df['string_X_train'].values.tolist())
text_input = st.text_area('輸入文字',value=init_value)
init_value_sp = split_x(text_input)

st.header('段落切分')
for text in init_value_sp:
    st.text(text)

if st.button('預測'):
    st.write('LCNO = {}'.format(init_df.LCNO.values[0]))
    x = init_value_sp
    answer = Collection_method(x,寶典)
    answer['class'] = answer['predict'].map(p2c)
    for i in range(len(answer)):
        text_input = answer.iloc[i]['string_X_train']
        text_output = answer.iloc[i]['predict']
        st.text(answer.iloc[i]['class'])
        color_output(text_input,text_output)
    table = answer[answer != 'not find'].dropna(axis=0)

    def g(x,keyword):
        try:
            return x[x.find(keyword)-2:x.find(keyword)]
        except:
            return 'not find'

    # 列表
    table['正本'] = table['string_X_train'].apply(lambda x:g(x,'ORIGINALS'))
    table['影本'] = table['string_X_train'].apply(lambda x:g(x,'COPIES'))
    table['正本2'] = table['string_X_train'].apply(lambda x:g(x,'ORIGINAL'))
    table['影本2'] = table['string_X_train'].apply(lambda x:g(x,'COPIE'))
    table['影本3'] = table['string_X_train'].apply(lambda x:g(x,'PHOTOCOPY'))

    for i in table.columns:
        table[i] = pd.to_numeric(table[i],errors='coerce')
    
    table['正本'].update(table['正本2'])
    table['影本'].update(table['影本2'])
    table['影本'].update(table['影本3'])
    
    table2 = answer[['class']]
    table2['正本'] = table['正本']
    table2['影本'] = table['影本']
    
    st.table(table2.dropna(axis=0).astype(str))

