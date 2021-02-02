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
from statsmodels.tsa.holtwinters import ExponentialSmoothing, Holt, SimpleExpSmoothing
import scipy.fftpack as fftp
import statsmodels.api as sm
import dtw
from fastdtw import fastdtw

os.chdir('C:\\Users\\Jerry\\Desktop\\Jerry\\projects\\covid19')
os.listdir()
file_names=os.listdir()
file_names=[x for x in file_names if '.csv' in x and ('map' in x or 'ts' in x or x=='COVID19_eng.csv')]

for n in file_names:
    print(n)
    names=n.replace('.csv','').replace('-','_')
    exec(names+'=pd.read_csv("'+n+'", encoding="latin1")')
    exec(names+'.name="'+names+'"')
data_list=[updated_ts_prov_active, updated_ts_prov_cases, updated_ts_prov_mortality, updated_ts_prov_testing, updated_ts_prov_recovered,
           updated_ts_prov_dvaccine, updated_ts_prov_avaccine]
# now we gotta join each rame together by the province and date. The cumulative stats of each frame is already contained in the
# active dataset. we just need to join the day to day changes. We have to join by province and date

# renaming date columns or merging

for df in data_list:
    df.columns=['date' if 'date' in n else n for n in df.columns]
    
for df in [updated_ts_prov_cases, updated_ts_prov_mortality, updated_ts_prov_recovered]:
    df.drop(columns=[n for n in df.columns if 'cumulative' in n], inplace=True)
    
# joining
covid=functools.reduce(lambda x, y: pd.merge(x, y, on=['date', 'province'], how='outer'), data_list)
covid.date=pd.to_datetime(covid.date ,format='%d-%m-%Y')
covid=covid.fillna(0)

# canada totals
data_list2=[updated_ts_canada_active, updated_ts_canada_cases, updated_ts_canada_mortality, updated_ts_canada_testing, updated_ts_canada_recovered,
           updated_ts_canada_dvaccine, updated_ts_canada_avaccine, updated_ts_canada_cvaccine]

for df in data_list2:
    df.columns=['date' if 'date' in n else n for n in df.columns]
    df.drop(columns='province', inplace=True)
    
for df in [updated_ts_canada_cases, updated_ts_canada_mortality, updated_ts_canada_recovered]:
    df.drop(columns=[n for n in df.columns if 'cumulative' in n], inplace=True)
    
# joining
canada_covid=functools.reduce(lambda x, y: pd.merge(x, y, on=['date'], how='outer'), data_list2)
canada_covid.date=pd.to_datetime(canada_covid.date ,format='%d-%m-%Y')
canada_covid=canada_covid.fillna(0)
#%%
for df in data_list:
    print(df.name)
    print(df.isnull().values.any())
    print(df.columns)
    
# updated_ts_prov_testing has null value!
updated_ts_prov_testing.iloc[np.where(updated_ts_prov_testing.isnull())] 
# looks like the testing_info is mostly nan. We disregard this since it doesn't impact our analysis directly. 


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
# we look at the inactivity time by shifting the recovered+deaths curve and fit it to the cumulative
inactive=canada_covid.cumulative_recovered.add(canada_covid.cumulative_deaths)[180:]
x=matplotlib.dates.date2num(canada_covid.date)[180:]
plt.plot_date(x, inactive, linestyle='-', linewidth=2, marker=None)
plt.plot_date(x, canada_covid.cumulative_cases[180:], linestyle='-', linewidth=1, marker=None)
# yikes, in July a change in the definition of 'recovered' has really bumped the numbers, we do the shift fit after
#let's shift it! we calculate the rmse for each shift and find the min rmse!
def shift_fit(shift_data, fit_data, shift_min, shift_max):
    rmse=pd.DataFrame(columns=['n','rmse'])
    for n in range(shift_min, shift_max+1):
        diff=fit_data.rsub(shift_data.shift(n))
        diff=diff.dropna()
        diff=np.sqrt((diff**2).mean())
        rmse=rmse.append({'n':n,'rmse':diff}, ignore_index=True)
    return rmse

test_shift=shift_fit(inactive, canada_covid.cumulative_cases[180:], -21, 0)
# rmse plot
plt.rcParams.update({'font.size': 10})
x1=plt.plot(test_shift.n, test_shift.rmse)
x2=plt.axvline(x=-12, linestyle='--', c='r')
plt.title('RMSE of shifting fit')
plt.xlabel('Days')
plt.ylabel('RMSE')
plt.legend([x1, x2], labels=['RMSE','minimum at x = -12'], loc='lower right')
plt.savefig('inactive_shift_fit_rmse.png', format='png', dpi=300)

x1=plt.plot_date(x, inactive.shift(-12), linestyle='-', linewidth=3, marker=None, alpha=0.7)
x2=plt.plot_date(x, canada_covid.cumulative_cases[180:], linestyle='-', linewidth=3, marker=None, alpha=0.7)
plt.title('shifted cumulative recovered and deaths vs cumulative cases for Canada', fontsize=9)
plt.legend([x1, x2], labels=['shifted cumulative recovered and deaths ','cumulative cases'], loc='upper left')
plt.savefig('inactive_shifted_cumulative.png', format='png', dpi=300)
## wow! it fits! so 12 days is the mean inactive period!
#%%
# we model each mean inactivity period (12 days) time window with an exponential growth function and get its growthrate in the exponent
fig, axs=plt.subplots(4,4)
plt.rc('xtick', labelsize=2.5)
i=0

for prov in covid.province.unique():
    test=covid[covid.province==prov]
    coeffs=[]
    start=0
    window=12
    incubation=5
    for k in range(start, len(test)-window-incubation):
        x=np.arange(0, window)
        y=test.cumulative_cases.iloc[np.arange(k+incubation, k+window+incubation)]
        active=1 if test.active_cases.iloc[k]<1 else test.active_cases.iloc[k]
        def exp_fit(x,a,b):
            return a+active*np.exp(b*x)
        
        exp_coeff,_=scipy.optimize.curve_fit(exp_fit, x, y, maxfev=10000, p0=[1,0])
        coeffs.append(exp_coeff[1])
        
    yloc=i%4
    xloc=i//4
    plt.rcParams.update({'font.size': 4})
    plt.tick_params(axis='x', which='major', labelsize=2)
    
    coeffs=[n*12 for n in coeffs]
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

#%% fourier analysis
test=covid[covid.province=='Ontario']
trans=fftp.fft(test.cases)
x=fftp.fftfreq(len(test), 1/len(test))
plt.rcParams.update({'font.size': 10})
plt.rc('xtick', labelsize=10)
plt.plot(x[:54], np.abs(trans)[:54])
plt.grid(True)

trans=pd.DataFrame({'n':x,'freq':trans})
trans.loc[np.abs(trans.freq).le(5000),'freq']=0
itrans=ifft(trans.freq)
x=np.arange(0,len(test))

plt.plot(x, itrans.real, linewidth=2)
plt.plot(x, test.cases, alpha=0.6)
plt.grid(True)

#%%
test=covid[covid.province=='Ontario']
test.loc[test.cases==0,'cases']=1
test.index=test.date
decomp = seasonal_decompose(test.cases, model='additive', freq=7)
decomp.plot()


#%%
test=covid[covid.province=='Ontario']
def differencing(x,d):
    if d==0:
        return x
    else:
        temp=x
        for i in range(0,d):
            temp=temp.diff()
        temp=temp.dropna()
        return temp
    
def simple_plot(x, grid=True):
    plt.plot(np.arange(0,len(x)),x)
    if grid:
        plt.grid(True)
 

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



def train_test(x, train_size, mode='r'):
    if mode=='r':
        length=int(np.floor(len(x)*train_size))
        return([x[:length], x[length:]])
    elif mode=='n':
        return([x[:len(x)-train_size], x[len(x)-train_size:]])

def pq_search(x, maxp, mindif, maxdif, maxq, pval_cut):
    final=[]
    for d in range(mindif,maxdif+1):        
        pranges=['p'+str(n) for n in list(range(0, maxp+1))]
        qranges=['q'+str(n) for n in list(range(0, maxq+1))]
        model_data=pd.DataFrame(index=pranges, columns=qranges)
        for p in range(0, maxp+1):
            for q in range(0, maxq+1):
                print([p,d,q])
                try:
                    temp_arima=ARIMA(x, (p, d, q)).fit(disp=False)
                    if (temp_arima.pvalues[1:]<=pval_cut).all():
                        model_data.iloc[p,q]=temp_arima.aic
                except:
                    print('error')
        final.append(model_data)
    return(final)

def fast_log(x):
    x=x.replace(0,1)
    return np.log(x)

def arima_res_plot(x):
    res = pd.DataFrame(x.resid)
    fig, ax = plt.subplots(1,2)
    res.plot(title="Residuals", ax=ax[0])
    res.plot(kind='kde', title='Density', ax=ax[1])
    fig.tight_layout()
    print('mean: '+str(res.mean()[0])+' std: '+str(res.std()[0]))

    
def arima_pred_plot(x, test_data, alpha, mode='n', conf_plot=True):
    fc, se, conf = x.forecast(len(test_data), alpha=alpha)  # 95% conf
    fc_series = pd.Series(fc, index=test_data.index)
    lower_series = pd.Series(conf[:, 0], index=test_data.index)
    upper_series = pd.Series(conf[:, 1], index=test_data.index)
    
    if mode=='exp':
        plt.plot(np.exp(test_data), label='actual')
        plt.plot(np.exp(fc_series), label='forecast')
        if conf_plot:
            plt.fill_between(lower_series.index, np.exp(lower_series), np.exp(upper_series), 
                 color='k', alpha=.15)
        plt.legend(loc='upper left', fontsize=8)
        print('SSE: '+str(((np.exp(fc_series)-np.exp(test_data))**2).sum()))
        return
    # Plot
    plt.plot(test_data, label='actual')
    plt.plot(fc_series, label='forecast')
    if conf_plot:
       plt.fill_between(lower_series.index, lower_series, upper_series, 
                     color='k', alpha=.15)
    plt.legend(loc='upper left', fontsize=8)
    print('SSE: '+str(((fc_series-test_data)**2).sum()))

def arima_pred_plot_all(x, train_data, test_data, alpha, mode='n', conf_plot=True):
    fc, se, conf = x.forecast(len(test_data), alpha=alpha)  # 95% conf
    fc_series = pd.Series(fc, index=test_data.index)
    lower_series = pd.Series(conf[:, 0], index=test_data.index)
    upper_series = pd.Series(conf[:, 1], index=test_data.index)
    
    if mode=='exp':
        plt.plot(np.exp(train_data), label='training')
        plt.plot(np.exp(test_data), label='actual')
        plt.plot(np.exp(fc_series), label='forecast')
        if conf_plot:
            plt.fill_between(lower_series.index, np.exp(lower_series), np.exp(upper_series), 
                 color='k', alpha=.15)
        plt.legend(loc='upper left', fontsize=8)
        print('SSE: '+str(((np.exp(fc_series)-np.exp(test_data))**2).sum()))
        return
    # Plot
    plt.plot(train_data, label='training')
    plt.plot(test_data, label='actual')
    plt.plot(fc_series, label='forecast')
    if conf_plot:
       plt.fill_between(lower_series.index, lower_series, upper_series, 
                     color='k', alpha=.15)
    plt.legend(loc='upper left', fontsize=8)
    print('SSE: '+str(((fc_series-test_data)**2).sum()))
    
def expsm_pred_plot(x, test_data):
    pred=x.forecast(len(test_data))
    plt.plot(test_data.index, test_data, label='actual')
    plt.plot(pred.index, pred, label='forecasted')
    plt.legend(loc='upper left', fontsize=8)
    print('SSE: '+str(((test_data-pred)**2).sum()))
    
def expsm_pred_plot_all(x, train_data, test_data):
    pred=x.forecast(len(test_data))
    plt.plot(train_data.index, train_data, label='training')
    plt.plot(test_data.index, test_data, label='actual')
    plt.plot(pred.index, pred, label='forecasted')
    plt.legend(loc='upper left', fontsize=8)
    print('SSE: '+str(((test_data-pred)**2).sum()))
    
############################# cases ###################################
test=covid[covid.province=='Ontario']
########## first try, no smooting
test.index=test.date
data=test.cumulative_cases
simple_plot(data) 
test_stationarity(differencing(data, 2), window=7, cutoff=0.01)# looks very bad, variance is wild
## trying some smoothing
data=data.rolling(7).mean().dropna()
test_stationarity(differencing(data, 3), window=7, cutoff=0.01)# looks very bad, variance is still wild
## trying log
data=test.cumulative_cases
data=fast_log(data)
test_stationarity(differencing(data, 2), window=7, cutoff=0.01)# looks very bad in the beginning, variance is wild
## removing the first 70
data=data[70:]
test_stationarity(differencing(data, 2), window=7, cutoff=0.01)# Not too too bad, let's try
train_data, test_data=train_test(data, 30, 'n') # we try to predict about a month's data
plot_pacf(differencing(train_data,2)) ## looks like AR(3)
plot_acf(differencing(test_data,2)) ## looks like MA(0)
search=pq_search(train_data, 5, 2, 2, 5, 0.05) #search for optimal pq
search[0] # looks like pdq=321 is the one with lowest aic
arima_model=ARIMA(train_data, (3,2,1)).fit(disp=False)
print(arima_model.summary())

plt.rcParams.update({'font.size': 10})
plt.rc('xtick', labelsize=5)
arima_res_plot(arima_model) #mean of residual almost 0, std is not bad
plt.savefig('cumulative_cases_arima_321_res.png', format='png', dpi=300)

plt.rc('xtick', labelsize=5)
arima_pred_plot(arima_model, test_data, 0.05, mode='exp') # the predictions are not looking too good, sse=4.1E10
plt.title('Predicted vs actual Cumulative Cases \n ARIMA(3,2,1) Residual: 4.1E10')
plt.savefig('cumulative_cases_pred_arima_321.png', format='png', dpi=300)

sm.stats.acorr_ljungbox(arima_model.resid, lags=3) # all residuals are above 0.05 so no relationships between the residuals

## trying other possible AIC's, like pdq=221
arima_model=ARIMA(train_data, (2,2,1)).fit(disp=False)
print(arima_model.summary())
plt.rc('xtick', labelsize=5)
arima_res_plot(arima_model) #mean of residual almost 0, std is not bad
plt.savefig('cumulative_cases_arima_221_res.png', format='png', dpi=300)

plt.rc('xtick', labelsize=5)
arima_pred_plot(arima_model, test_data, 0.05, mode='exp') # the predictions are looking good, sse=4.3E8
plt.title('Predicted vs actual Cumulative Cases \n ARIMA(2,2,1) Residual: 4.3E8')
plt.savefig('cumulative_cases_pred_arima_221.png', format='png', dpi=300)

arima_pred_plot_all(arima_model, train_data, test_data, 0.05, mode='exp')
plt.title('Cumulative Cases prediction Covid-19 \n ARIMA(2,2,1)')
plt.savefig('cumulative_cases_pred_all_arima_221.png', format='png', dpi=300)

sm.stats.acorr_ljungbox(arima_model.resid, lags=3) # all residuals are above 0.05 so no relationships between the residuals
## so in summary, Arima model with pdq=0,2,5 is pretty good with SSE of 3.4E8

########## exponential smoothing
# now we know the residual to beat is 4.3E8, let's see another type of model
test=covid[covid.province=='Ontario']
test.index=test.date
data=test.cumulative_cases
train_data, test_data=train_test(data, 30, 'n')

# exponential smoothing damped
exp_model = ExponentialSmoothing(train_data, trend='mul', seasonal=None, damped=True).fit()
arima_res_plot(exp_model)

expsm_pred_plot(exp_model, test_data) #sse=5.4E8
plt.title('Predicted vs actual Cumulative Cases \n damped exponential smoothing, Residual: 5.4E8')
plt.savefig('cumulative_cases_pred_expsm_damped.png', format='png', dpi=300)
expsm_pred_plot_all(exp_model, train_data, test_data)

#holt
holt_model=Holt(train_data, exponential=True).fit(optimized=True)
arima_res_plot(holt_model)
expsm_pred_plot(holt_model, test_data) #sse=1.5E9

### dynamic time warping brute forcing
test=covid[covid.province=='Ontario']
def dtw_pred(data, train_len, pred_len):
    data=data.values
    total_len=train_len+pred_len #total length of the train/pred series
    train_final=data[len(data)-total_len:len(data)-pred_len] # where the real training data is
    scale=train_final[-1]
    train_final=train_final/scale
    pred_final=np.array([0]*pred_len)
    denom=0
    for i in range(0, len(data)-pred_len-total_len+1):
        # normalized train
        train=data[i:i+train_len]
        last=train[-1]
        train=train/last
        #normalized pred
        pred=data[i+train_len:i+total_len]
        pred=pred/last
        #similarity
        sim=1/(1+fastdtw(train, train_final)[0])
        result=pred*sim
        #add result
        pred_final=pred_final+result
        denom=denom+sim
    result=pred_final/denom*scale
    return(result)
        
#finding the optimal number of training days for each prediction length, 

best=[]
for n in range(30, 110, 10):
    print(n)
    num=[]
    residuals=[]
    for i in range(50,60):
        a=dtw_pred(test.cumulative_cases, i, n)
        res=test.cumulative_cases[-n:]-a
        res=(res**2).sum()
        num.append(i)
        residuals.append(res)
    b=num[residuals.index(min(residuals))]
    best.append(b)
    
# [54, 55, 54, 53, 52, 51, 51, 52], looks bretty stable in the 52's, mean is 53
# we therefore use training length of 53 to train our model
dtw_forecast=dtw_pred(test.cumulative_cases, 53, 30)

simple_plot(test.cumulative_cases[-30:])
simple_plot(dtw_forecast)
res=test.cumulative_cases[-30:]-dtw_forecast
res=sum(res**2) # residual of 1.26E9, not bad! but not the best obviously
simple_plot(test.cumulative_cases)
