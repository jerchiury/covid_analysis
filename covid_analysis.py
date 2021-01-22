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

os.chdir('C:\\Users\\Jerry\\Desktop\\Jerry\\projects\\covid19')
os.listdir()
file_names=os.listdir()
file_names=[x for x in file_names if '.csv' in x and ('map' in x or 'ts' in x or x=='COVID19_eng.csv')]

#%%
for n in file_names:
    print(n)
    names=n.replace('.csv','').replace('-','_')
    exec(names+'=pd.read_csv("'+n+'", encoding="latin1")')

#%%
# we explore the data a bit
#--------------------updated_ts_prov_cases---------------------------
updated_ts_prov_cases.head()
updated_ts_prov_cases.isnull().values.any() # no null values

# we convert date from string to datetime
updated_ts_prov_cases.date_report=pd.to_datetime(updated_ts_prov_cases.date_report, format='%d-%m-%Y')
max(updated_ts_prov_cases.date_report)
min(updated_ts_prov_cases.date_report)
# looks like the dates are from 2020 Jan 25 to today
# we now plot it

i=0
for prov in updated_ts_prov_cases.province.unique():
    i=i+1
    plt.rcParams.update({'font.size': 4})
    temp=updated_ts_prov_cases[updated_ts_prov_cases.province==prov]
    x=matplotlib.dates.date2num(temp.date_report)
    y=temp.cumulative_cases
    plt.rc('xtick',labelsize=2.5)
    plt.tight_layout()
    plot=plt.subplot(4,4,i)
    plt.plot_date(x, y,  linestyle='-', marker=None)
    plot.set_title(prov, fontsize=6)
    
plt.savefig('prov_cases.png', format='png', dpi=500)
