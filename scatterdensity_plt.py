import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from numpy import polyfit, poly1d
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

#import data
data = pd.read_excel('/Users/zheng/Documents/master/Miaowan/ML_results.xlsx',sheet_name = 'DO_test')
y=np.array(data.RF);x=np.array(data.y_test);

#fit line
coeff = polyfit(x,y, 1)
f = poly1d(coeff)

#text
title = 'RF'
str1 ='y=';str1 = str1.strip();str2 = str(f).strip()
func = str1+str2
rmse =round((mean_squared_error(x,y)**0.5),3)
rmse_t = str('RMSE='+str(rmse))
mae = round((mean_absolute_error(x,y)),3)
mae_t = str('MAE='+str(mae))
r = round(r2_score(x,y),3)
r_t = str('R\u00b2='+str(r))
str_all = '';str_all+=func+'\n';str_all+=rmse_t+'\n'
str_all+=mae_t+'\n';str_all+=r_t+'\n';

#plot
xy = np.vstack([x,y]);z = gaussian_kde(xy)(xy)
idx = z.argsort()
x, y, z = x[idx], y[idx], z[idx]
fig, ax = plt.subplots(figsize=(5,5),dpi=600)
plt.rc('font',family='Times New Roman')
plt.scatter(x, y,c=z, s=5,cmap='Spectral_r')
ax.plot((0, 1), (0, 1), transform=ax.transAxes, ls='--',c='k', label="1:1 line")
plt.plot(x, f(x),linewidth=1,c='r', label="fitted line")
plt.legend(fontsize=12,frameon=False,loc="center right")
plt.text(2.8,6.2,str_all,fontsize=20);plt.text(7,3.0,title,fontsize=20,fontweight='bold');
plt.xlabel('Measured DO (mg/L)',fontsize=20);plt.ylabel('Estimated DO (mg/L)',fontsize=20)
cbar=plt.colorbar();cbar.ax.tick_params(labelsize=20)
plt.tick_params(labelsize=20)  
plt.show()
