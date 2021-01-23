# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 01:31:00 2021

@author: Jerry
"""
import os
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import functools 
import scipy.optimize

os.chdir('C:\\Users\\Jerry\\Desktop\\Jerry\\projects\\covid19')
os.listdir()
file_names=os.listdir()
file_names=[x for x in file_names if '.csv' in x and ('map' in x or 'ts' in x or x=='COVID19_eng.csv')]

#%%
for n in file_names:
    print(n)
    names=n.replace('.csv','').replace('-','_')
    exec(names+'=pd.read_csv("'+n+'", encoding="latin1")')
    exec(names+'.name="'+names+'"')
#%%
data_list=[updated_ts_prov_active, updated_ts_prov_cases, updated_ts_prov_mortality, updated_ts_prov_testing, updated_ts_prov_recovered,
           updated_ts_prov_dvaccine, updated_ts_prov_avaccine]
#%%
for df in data_list:
    print(df.name)
    print(df.isnull().values.any())
    print(df.columns)
    
# updated_ts_prov_testing has null value!
updated_ts_prov_testing.iloc[np.where(updated_ts_prov_testing.isnull())] 
# looks like the testing_info is mostly nan. We disregard this since it doesn't impact our analysis directly. 

#%%
# now we gotta join each rame together by the province and date. The cumulative stats of each frame is already contained in the
# active dataset. we just need to join the day to day changes. We have to join by province and date

# renaming date columns or merging

for df in data_list:
    df.columns=['date' if 'date' in n else n for n in df.columns]
    
for df in [updated_ts_prov_cases, updated_ts_prov_mortality, updated_ts_prov_recovered]:
    df.drop(columns=[n for n in df.columns if 'cumulative' in n], inplace=True)

#%%
# joining
covid=functools.reduce(lambda x, y: pd.merge(x, y, on=['date', 'province'], how='outer'), data_list)
covid.date=pd.to_datetime(covid.date ,format='%d-%m-%Y')
covid=covid.fillna(0)
#%%
fig, axs=plt.subplots(4,4)
i=0
for prov in covid.province.unique():
    yloc=i%4
    xloc=i//4
    plt.rcParams.update({'font.size': 4})
    
    # cases
    temp=covid[covid.province==prov]
    x=matplotlib.dates.date2num(temp.date)
    y=temp.cumulative_cases
    plt.rc('xtick', labelsize=2.5)
    x1=axs[xloc, yloc].plot_date(x, y,  linestyle='-', linewidth=0.8, marker=None, c='black', alpha=1)[0]
    
    # active
    temp=covid[covid.province==prov]
    x=matplotlib.dates.date2num(temp.date)
    y=temp.active_cases
    axs[xloc, yloc].set_title(prov, fontsize=6)
    x2=axs[xloc, yloc].plot_date(x, y,  linestyle='-', linewidth=0.8, marker=None, c='r', alpha=0.5)[0]
    
    #recovered
    temp=covid[covid.province==prov]
    x=matplotlib.dates.date2num(temp.date)
    y=temp.recovered
    axs[xloc, yloc].set_title(prov, fontsize=6)
    x3=axs[xloc, yloc].plot_date(x, y,  linestyle='-', linewidth=0.8, marker=None, c='g', alpha=0.5)[0]
    
    #deaths
    temp=covid[covid.province==prov]
    x=matplotlib.dates.date2num(temp.date)
    y=temp.deaths
    axs[xloc, yloc].set_title(prov, fontsize=6)
    x4=axs[xloc, yloc].plot_date(x, y,  linestyle='-', linewidth=0.8, marker=None, c='b', alpha=0.5)[0]
    
    i=i+1

axs[3,2].axis('off')
axs[3,3].axis('off')
fig.tight_layout()
fig.legend([x1, x2, x3, x4], labels=['total cases','active cases', 'recovered', 'deaths'], loc='lower right')

plt.savefig('prov_cases_active.png', format='png', dpi=300)

#%%
# we model each 2-week time window with an exponential growth function and get its growthrate in the exponent
fig, axs=plt.subplots(4,4)
i=0

for prov in covid.province.unique():
    test=covid[covid.province==prov]
    res=[]
    coeffs=[]
    start=0
    window=10
    incubation=5
    for k in range(start, len(test)-window-incubation):
        x=np.arange(0, window)
        y=test.cumulative_cases.iloc[np.arange(k+incubation, k+window+incubation)]
        active=1 if test.active_cases.iloc[k]<1 else test.active_cases.iloc[k]
        def exp_fit(x,a,b):
            return a+active*np.exp(b*x)
        
        exp_coeff,_=scipy.optimize.curve_fit(exp_fit, x, y, maxfev=10000, p0=[1,0])
        coeffs.append(exp_coeff[1])
        residuals=y.array-exp_fit(x, exp_coeff[0], exp_coeff[1])
        residuals=sum((residuals/res_scale)**2)
        res.append(residuals)
        
    yloc=i%4
    xloc=i//4
    plt.rcParams.update({'font.size': 4})
    plt.tick_params(axis='x', which='major', labelsize=2)
    
    coeffs=[]
    x=matplotlib.dates.date2num(test.date)[start:len(test)-window-incubation]
    axs[xloc, yloc].plot_date(x, coeffs,  linestyle='-', linewidth=1, marker=None, c='b')
    axs[xloc, yloc].set_title(prov, fontsize=6)
    axs[xloc, yloc].grid(True)
    i=i+1

axs[3,2].axis('off')
axs[3,3].axis('off')
fig.tight_layout()
fig.subplots_adjust(top=0.88)
fig.suptitle('Transmission Rates as estimated with exponential growth by province', fontsize=10)
    
plt.savefig('prov_equivalent_exp.png', format='png', dpi=300)

#%% Susprciously, the number of active cases and the number of confirmed cases have 
# almost the same shapes, but with a constant factor difference and bit of a lag
test=covid[covid.province=='Alberta']
plt.plot_date(test.date, test.cases*13, linestyle='-', linewidth=1, marker=None)
plt.plot_date(test.date, test.active_cases, linestyle='-', linewidth=1, marker=None)