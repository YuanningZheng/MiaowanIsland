import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression as MIR
from sklearn.preprocessing import MinMaxScaler
MM = MinMaxScaler()

ML_data = pd.read_excel('/Users/zheng/Documents/master/Miaowan/ML.xlsx',sheet_name = 'data')
mi_num = ML_data.iloc[:,0:12]
data_MM = pd.DataFrame(MM.fit_transform(mi_num))
length = len(data_MM.columns)
MI_all = []
y = (data_MM.iloc[:,0]).values.ravel()  
for i in range (3,length): 
    x = (data_MM.iloc[:,i]).values
    X = x.reshape(-1,1)
    print (MIR(X,y,n_neighbors=20,random_state=7))
