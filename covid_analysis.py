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
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

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


#%% Identify the distribution of active_case resolution/death date
test=covid[covid.province=='Alberta']
plt.plot()


#%% fourier analysis
test=covid[covid.province=='Ontario']
decomp = seasonal_decompose(test.cases, model='additive', freq=2)
decomp.plot()

#%%
test=covid[covid.province=='Quebec']
def differencing(x,d):
    if d==0:
        return x
    else:
        temp=x
        for i in range(0,d):
            temp=temp.diff()
        temp=temp.dropna()
        return temp
    
def simple_plot(x):
    plt.plot(np.arange(0,len(x)),x)
 

def test_stationarity(timeseries, window = 7, cutoff = 0.01):

    #Determing rolling statistics
    rolmean = timeseries.rolling(window).mean()
    rolstd = timeseries.rolling(window).std()

    #Plot rolling statistics:
    fig = plt.figure(figsize=(12, 8))
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()
    
    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC', maxlag = 20 )
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    pvalue = dftest[1]
    if pvalue < cutoff:
        print('p-value = %.4f. The series is likely stationary.' % pvalue)
    else:
        print('p-value = %.4f. The series is likely non-stationary.' % pvalue)
    
    print(dfoutput)   



def train_test(x, train_size):
    length=int(np.floor(len(x)*train_size))
    return([x[:length], x[length:]])

def pq_search(x, maxp, mindif, maxdif, maxq, pval_cut):
    final=[]
    for d in range(mindif,maxdif+1):        
        pranges=['p'+str(n) for n in list(range(1, maxp+1))]
        qranges=['q'+str(n) for n in list(range(1, maxq+1))]
        model_data=pd.DataFrame(index=pranges, columns=qranges)
        for p in range(1, maxp+1):
            for q in range(1, maxq+1):
                print([p,d,q])
                try:
                    temp_arima=ARIMA(x, (p, d, q)).fit(disp=False)
                    if (temp_arima.pvalues[1:]<=pval_cut).all():
                        model_data.iloc[p-1,q-1]=temp_arima.aic
                except:
                    print('error')
        final.append(model_data)
    return(final)

####### cases
test=covid[covid.province=='Ontario']
test.cases=test.cases.replace(0,1)
test.cases=np.log(test.cases)
test_stationarity(differencing(test.cases, 1), window=7, cutoff=0.01) # cases order is 1
plot_pacf(differencing(test.cases,1)) ## looks like AR(2)
plot_acf(differencing(test.cases,1)) ## looks like MA(1)
cases_arima=ARIMA(test.cases, (2,1,2)).fit(disp=False) #412
print(cases_arima.summary())

res = pd.DataFrame(cases_arima.resid)
fig, ax = plt.subplots(1,2)
res.plot(title="Residuals", ax=ax[0])
res.plot(kind='kde', title='Density', ax=ax[1])
fig.tight_layout()
res.mean() #-0.118, good
res.std() #244, not too bad

cases_arima.plot_predict(dynamic=False) # nice, let's validate it with 8/2 train/test

train_data, test_data=train_test(test.cases, 0.8)
cases_arima=ARIMA(train_data, (2,1,1)).fit(disp=False)
print(cases_arima.summary()) # BAD Pvalues. Let's do a search for optimal pq

search=pq_search(train_data, 3, 1, 2, 3, 0.05) # looks like 1,1,2 or 3,2,1
plot_pacf(differencing(train_data,1)) ## looks like AR(2)
plot_acf(differencing(train_data,1)) ## looks like MA(1)
search[0]
#let's see their performances
cases_arima=ARIMA(train_data, (1,1,2)).fit(disp=False)
print(cases_arima.summary())
# Make as pandas series
fc, se, conf = cases_arima.forecast(len(test_data), alpha=0.05)  # 95% conf
fc_series = pd.Series(fc, index=test_data.index)
lower_series = pd.Series(conf[:, 0], index=test_data.index)
upper_series = pd.Series(conf[:, 1], index=test_data.index)

# Plot
plt.plot(train_data, label='training')
plt.plot(test_data, label='actual')
plt.plot(fc_series, label='forecast')
plt.fill_between(lower_series.index, lower_series, upper_series, 
                 color='k', alpha=.15)
plt.title('ARIMA Forecast vs Actuals (3,2,1)')
plt.legend(loc='upper left', fontsize=8)

plt.savefig('cases_forecast_321.png', format='png', dpi=300)

####### active_cases
test_stationarity(differencing(test.active_cases, 2), window=7, cutoff=0.01) # active_cases order is 2
plot_pacf(differencing(test.active_cases,2)) ## looks like AR(5)
plot_acf(differencing(test.active_cases,2)) ## looks like MA(2)
active_cases_arima=ARIMA(test.active_cases, (5,2,2)).fit(disp=False)
print(active_cases_arima.summary()) # looks like MA(1) is not too good
active_cases_arima=ARIMA(test.active_cases, (5,2,1)).fit(disp=False)
print(active_cases_arima.summary()) # looks like MA(1) is better with lower AIC

res = pd.DataFrame(active_cases_arima.resid)
fig, ax = plt.subplots(1,2)
res.plot(title="Residuals", ax=ax[0])
res.plot(kind='kde', title='Density', ax=ax[1])
fig.tight_layout()
res.mean() #-0.02, good
res.std() #300, not too bad
active_cases_arima.plot_predict(dynamic=False)

train_data, test_data=train_test(test.active_cases, 0.8)
search=pq_search(train_data, 5, 2, 3, 5, 0.05) # looks like either 4,2,1 is the ways to go
search[1]
#let's see their performances
active_cases_arima=ARIMA(train_data, (4,2,1)).fit(disp=False)

# Make as pandas series
fc, se, conf = active_cases_arima.forecast(len(test_data), alpha=0.05)  # 95% conf
fc_series = pd.Series(fc, index=test_data.index)
lower_series = pd.Series(conf[:, 0], index=test_data.index)
upper_series = pd.Series(conf[:, 1], index=test_data.index)

# Plot
plt.plot(train_data, label='training')
plt.plot(test_data, label='actual')
plt.plot(fc_series, label='forecast')
plt.fill_between(lower_series.index, lower_series, upper_series, 
                 color='k', alpha=.15)
plt.title('ARIMA Forecast vs Actuals (4,2,1)')
plt.legend(loc='upper left', fontsize=8)

plt.savefig('active_cases_forecast_421.png', format='png', dpi=300)
