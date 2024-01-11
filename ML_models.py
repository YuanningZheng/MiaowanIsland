import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
MM = MinMaxScaler()

#import data
data = pd.read_excel('/Users/zheng/Documents/master/Miaowan/ML.xlsx',sheet_name = 'data')
X= np.array(pd.DataFrame(data.loc[: ,['temperature','salinity','depth','air pressure','relative humidity']]))
y = np.array(data.DO)

#train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= 7)
X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)
X_train = pd.DataFrame(MM.fit_transform(X_train))
X_test = pd.DataFrame(MM.transform(X_test))

#Model
from sklearn.neural_network import MLPRegressor          
mlp_reg=MLPRegressor(hidden_layer_sizes=(120,120,120),max_iter = 1000,random_state=7)
mlp_reg.fit(X_train,y_train)
y_pred_mlp = mlp_reg.predict(X_test)
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_estimators = 196,max_depth=19,random_state=7)
rfr.fit(X_train, y_train)
y_pred_rfr = rfr.predict(X_test)
from m5py import M5Prime, export_text_m5
regr = M5Prime(use_pruning=False,smoothing_constant=5,random_state=7)
regr.fit(X_train, y_train)
y_pred_regr = regr.predict(X_test)

#accuracy evaluation
print(mean_squared_error(y_test ,y_pred_mlp)**0.5,mean_absolute_error(y_test ,y_pred_mlp),r2_score(y_test ,y_pred_mlp))
print(mean_squared_error(y_test , y_pred_rfr)**0.5,mean_absolute_error(y_test , y_pred_rfr),r2_score(y_test , y_pred_rfr))
print(mean_squared_error(y_test,y_pred_regr)**0.5,mean_absolute_error(y_test,y_pred_regr),r2_score(y_test,y_pred_regr))

#save
y_all = np.c_[y_test,y_pred_mlp,y_pred_rfr,y_pred_regr]
test = pd.DataFrame(y_all)
writer = pd.ExcelWriter('/Users/zheng/Documents/master/Miaowan/ML_results.xlsx',mode='a', engine='openpyxl',if_sheet_exists='new')
test.to_excel(writer, sheet_name='DO')
writer.close()
