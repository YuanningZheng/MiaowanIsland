import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
MM = MinMaxScaler()

data = pd.read_excel('/Users/zheng/Documents/master/Miaowan/ML.xlsx',sheet_name = 'data')
X= np.array(pd.DataFrame(data.loc[: ,['temperature','salinity','depth','air pressure','relative humidity']]))
y = np.array(data.DO)
label=data.columns
#site1
site1=data.iloc[0:21000,:];site1.columns = label
site1_X= pd.DataFrame(site1.loc[: ,['temperature','salinity','depth','air pressure','relative humidity']])
#site2
site2=data.iloc[21000:,:];site2.columns = label
site2_X= pd.DataFrame(site2.loc[: ,['temperature','salinity','depth','air pressure','relative humidity']])

#RF model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= 7)
X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)
X_train = pd.DataFrame(MM.fit_transform(X_train))
X_test = pd.DataFrame(MM.transform(X_test))
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_estimators = 196,max_depth=19,random_state=7)
rfr.fit(X_train, y_train)
y_pred_rfr = rfr.predict(X_test)
y_train_rfr = rfr.predict(X_train)

#apply
#site1
site1_X_MM=pd.DataFrame(MM.transform(site1_X))
site1_y_rfr = rfr.predict(site1_X_MM)
site1_y = np.array(site1.DO)
y_all = np.c_[site1_y,site1_y_rfr]
results = pd.DataFrame(y_all)
writer = pd.ExcelWriter('/Users/zheng/Documents/master/Miaowan/ML_results.xlsx',mode='a', engine='openpyxl',if_sheet_exists='new')
results.to_excel(writer, sheet_name='site1_DO')
writer.close()
#site2
site2_X_MM=pd.DataFrame(MM.transform(site2_X))
site2_y_rfr = rfr.predict(site2_X_MM)
site2_y = np.array(site2.DO)
y_all = np.c_[site2_y,site2_y_rfr]
results = pd.DataFrame(y_all)
writer = pd.ExcelWriter('/Users/zheng/Documents/master/Miaowan/ML_results.xlsx',mode='a', engine='openpyxl',if_sheet_exists='new')
results.to_excel(writer, sheet_name='site2_DO')
writer.close()
#Yupai_Zhanjiang
valid_data = pd.read_excel('/Users/zheng/Documents/master/Miaowan/Zhanjiang_Yupai.xlsx',sheet_name = 'data_use')
X_valid = pd.DataFrame(valid_data.loc[: ,['temperature','salinity','depth','air pressure','relative humidity']])
X_valid = pd.DataFrame(MM.transform(X_valid))
y_valid = np.array(valid_data.DO)
y_valid_rfr = rfr.predict(X_valid)
y_all = np.c_[y_valid,y_valid_rfr]
results = pd.DataFrame(y_all)
writer = pd.ExcelWriter('/Users/zheng/Documents/master/Miaowan/Zhanjiang_Yupai.xlsx',mode='a', engine='openpyxl',if_sheet_exists='new')
results.to_excel(writer, sheet_name='DO')
writer.close()
