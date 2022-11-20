#%%
import yfinance as yf
import itertools
import prophet as p
import numpy as np
import investpy as ip
import pandas as pd
import investpy as inv
import yfinance as yf
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import gc

from dateutil.relativedelta import relativedelta
from datetime import date
from dateutil.rrule import rrule, DAILY
from datetime import datetime
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics

# %%
def get_log_return(ticker, column_name):
    np.random.seed(0)
    return_col = 'log_ret_' + ticker
    df = yf.download(ticker, start=(datetime.now() - relativedelta(years=5)).strftime("%Y-%m-%d"), end=datetime.now().strftime("%Y-%m-%d"))
    df['Date_NI'] = pd.to_datetime(df.index)
    df['Date_NI'] = df['Date_NI'].dt.date
    df['log_ret'] = np.log(df[column_name]) - np.log(df[column_name].shift(1))
    df['Color'] = np.where(df["log_ret"]<0, 'Loss', 'Gain')
    # df['Volatility'] =  df['log_ret'].rolling(window=252).std()*np.sqrt(252)*100
    df[return_col] = df['log_ret']
    return df

def get_model_columns(ticker,df):
    df = df[['log_ret_' + ticker,'Date_NI']]
    df.reset_index()
    return df

def get_complete_model_df(df_main, df_others):
    df_all = df_main
    for df in df_others:
        df_all = df_all.merge(df, on='Date_NI', how='left')
    return df_all

def get_tuned_param(df,start_date):
    param_grid = {  
        'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
        'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
    }
    cutoffs = [(start_date + relativedelta(months=12)),(start_date + relativedelta(months=18)),(start_date + relativedelta(months=24)),(start_date + relativedelta(months=30)),(start_date + relativedelta(months=36))]

    # Generate all combinations of parameters
    all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
    rmses = []  # Store the RMSEs for each params here
    # Use cross validation to evaluate all parameters
    for params in all_params:
        m = p.Prophet(**params).fit(df)  # Fit model with given params
        df_cv = cross_validation(m, cutoffs=cutoffs, horizon='30 days', parallel="processes")
        df_p = performance_metrics(df_cv, rolling_window=1)
        rmses.append(df_p['rmse'].values[0])

    # Find the best parameters
    tuning_results = pd.DataFrame(all_params)
    tuning_results['rmse'] = rmses
    return tuning_results

#%%
## Nifty 50 Data
df_NIFTY50 = get_log_return("^NSEI",'Close')
df_NIFTY50_mod = get_model_columns("^NSEI",df_NIFTY50)
gc.collect()

# %%
df_bist = get_log_return("XU100.IS",'Close')
df_bist_mod = get_model_columns("XU100.IS",df_bist)
gc.collect()

#%%
df_dji = get_log_return("^DJI",'Close')
df_dji_mod = get_model_columns("^DJI",df_dji)
gc.collect()

#%%
df_nya = get_log_return("^NYA",'Close')
df_nya_mod = get_model_columns("^NYA",df_nya)
gc.collect()

# %%
## FTSE
df_ftse = get_log_return("^FTSE",'Close')
df_ftse_mod = get_model_columns("^FTSE",df_ftse)
gc.collect()

## German DAX
df_dax = get_log_return("^GDAXI",'Close')
df_dax_mod = get_model_columns("^GDAXI",df_dax)
gc.collect()

## KOSPI
df_kospi = get_log_return("^KS11",'Close')
df_kospi_mod = get_model_columns("^KS11",df_kospi)
gc.collect()

## Dollar Index
df_usd = get_log_return("DX-Y.NYB",'Close')
df_usd_mod = get_model_columns("DX-Y.NYB",df_usd)
gc.collect()

## United States Brent Oil Fund
df_oil = get_log_return("BNO",'Close')
df_oil_mod = get_model_columns("BNO",df_oil)
gc.collect()

## Nikkei
df_nikkei = get_log_return("^N225",'Close')
df_nikkei_mod = get_model_columns("^N225",df_nikkei)
gc.collect()

## Gold ETF
df_au = get_log_return("AAAU",'Close')
df_au_mod = get_model_columns("AAAU",df_au)
gc.collect()

## SILVERBEES.NS
df_ag = get_log_return("SILVERBEES.NS",'Close')
df_ag_mod = get_model_columns("SILVERBEES.NS",df_ag)
gc.collect()

## Bursa Malaysia ^KLSE
df_my = get_log_return("^KLSE",'Close')
df_my_mod = get_model_columns("^KLSE",df_my)
gc.collect()

#%% create complete model by merging nifty with other index returns
df_overall_model = get_complete_model_df(df_main= df_NIFTY50_mod, df_others= [df_bist_mod,df_dji_mod, df_nya_mod, df_ftse_mod, df_dax_mod, df_kospi, df_usd_mod, df_oil_mod, df_nikkei_mod, df_au_mod, df_ag_mod, df_my_mod])

#%% rename the ds column (date sample)
df_overall_model.rename(columns = {'Date_NI':'DS'}, inplace = True)

# %% get the 5 year prior date
start_date = (datetime.now() - relativedelta(years=5)).date()
training_end_date = (datetime.now() - relativedelta(days=60)).date()
training_end_date_slice = (datetime.now() - relativedelta(days=120)).date()

mask_training_overall = (df_overall_model.DS <= training_end_date)
mask_prediction = (df_overall_model.DS > training_end_date)
df_training_overall = df_overall_model.loc[mask_training_overall,:]
df_prediction = df_overall_model.loc[mask_prediction,:]

holiday_dates = []

# %% get the list of weekends
for date in rrule(DAILY, dtstart=start_date, until=datetime.now()):
    if((date.strftime('%A')=="Saturday") or (date.strftime('%A')=="Sunday")):
        holiday_dates.append(date.date())

# %% attach holiday list to the model
df_holidays = pd.DataFrame()
df_holidays['ds'] = holiday_dates
df_holidays['holiday'] = "Weekend"
df_training_overall_tuning = df_training_overall[['DS','log_ret_^NSEI']]
df_training_overall_tuning.rename(columns = {'log_ret_^NSEI':'y','DS':'ds'}, inplace = True)
df_training_overall_tuning['ds'] = pd.to_datetime(df_training_overall_tuning['ds'])

#%%
# params = get_tuned_param(df_training_overall_tuning, (datetime.now() - relativedelta(years=5)))

# %% adding all the co-regressors
model = p.Prophet(holidays=df_holidays)
model.add_regressor('log_ret_XU100.IS')
model.add_regressor('log_ret_^DJI')
model.add_regressor('log_ret_^NYA')
model.add_regressor('log_ret_^FTSE')
model.add_regressor('log_ret_^GDAXI')
model.add_regressor('log_ret_^KS11')
model.add_regressor('log_ret_DX-Y.NYB')
model.add_regressor('log_ret_BNO')
model.add_regressor('log_ret_^N225')
model.add_regressor('log_ret_^KLSE')
df_training_overall.rename(columns = {'log_ret_^NSEI':'y','DS':'ds'}, inplace = True)

#%% fill in the missing value by using interpolate as imputation
df_training_overall['log_ret_XU100.IS'] = df_training_overall['log_ret_XU100.IS'].interpolate(method='spline', order=1, limit=10, limit_direction='both')
df_training_overall['log_ret_^DJI'] = df_training_overall['log_ret_^DJI'].interpolate(method='spline', order=1, limit=10, limit_direction='both')
df_training_overall['log_ret_^NYA'] = df_training_overall['log_ret_^NYA'].interpolate(method='spline', order=1, limit=10, limit_direction='both')
df_training_overall['log_ret_^FTSE'] = df_training_overall['log_ret_^FTSE'].interpolate(method='spline', order=1, limit=10, limit_direction='both')
df_training_overall['log_ret_^GDAXI'] = df_training_overall['log_ret_^GDAXI'].interpolate(method='spline', order=1, limit=10, limit_direction='both')
df_training_overall['log_ret_^KS11'] = df_training_overall['log_ret_^KS11'].interpolate(method='spline', order=1, limit=10, limit_direction='both')
df_training_overall['log_ret_DX-Y.NYB'] = df_training_overall['log_ret_DX-Y.NYB'].interpolate(method='spline', order=1, limit=10, limit_direction='both')
df_training_overall['log_ret_BNO'] = df_training_overall['log_ret_BNO'].interpolate(method='spline', order=1, limit=10, limit_direction='both')
df_training_overall['log_ret_^N225'] = df_training_overall['log_ret_^N225'].interpolate(method='spline', order=1, limit=10, limit_direction='both')
df_training_overall['log_ret_^KLSE'] = df_training_overall['log_ret_^KLSE'].interpolate(method='spline', order=1, limit=10, limit_direction='both')
model.fit(df_training_overall)

# %% interpolate missing data in prediction data set
df_prediction.rename(columns = {'DS':'ds'}, inplace = True)
df_prediction.drop(['log_ret_^NSEI'], axis=1)
df_prediction['log_ret_XU100.IS'] = df_prediction['log_ret_XU100.IS'].interpolate(method='spline', order=1, limit=10, limit_direction='both')
df_prediction['log_ret_^DJI'] = df_prediction['log_ret_^DJI'].interpolate(method='spline', order=1, limit=10, limit_direction='both')
df_prediction['log_ret_^NYA'] = df_prediction['log_ret_^NYA'].interpolate(method='spline', order=1, limit=10, limit_direction='both')
df_prediction['log_ret_^FTSE'] = df_prediction['log_ret_^FTSE'].interpolate(method='spline', order=1, limit=10, limit_direction='both')
df_prediction['log_ret_^GDAXI'] = df_prediction['log_ret_^GDAXI'].interpolate(method='spline', order=1, limit=10, limit_direction='both')
df_prediction['log_ret_^KS11'] = df_prediction['log_ret_^KS11'].interpolate(method='spline', order=1, limit=10, limit_direction='both')
df_prediction['log_ret_DX-Y.NYB'] = df_prediction['log_ret_DX-Y.NYB'].interpolate(method='spline', order=1, limit=10, limit_direction='both')
df_prediction['log_ret_BNO'] = df_prediction['log_ret_BNO'].interpolate(method='spline', order=1, limit=10, limit_direction='both')
df_prediction['log_ret_^N225'] = df_prediction['log_ret_^N225'].interpolate(method='spline', order=1, limit=10, limit_direction='both')
df_prediction['log_ret_^KLSE'] = df_prediction['log_ret_^KLSE'].interpolate(method='spline', order=1, limit=10, limit_direction='both')

df_forecast = model.predict(df_prediction)

# %% get the forecast and plot components
fig1 = model.plot(df_forecast)
fig2 = model.plot_components(df_forecast)

# %% create comparasion data set to check for predicted and actual value
df_prediction['ds'] = pd.to_datetime(df_prediction['ds'])
df_comparision = pd.merge(df_prediction,df_forecast, on='ds', how='inner')

# %% 
df_comparision = df_comparision[['ds','log_ret_^NSEI','trend','yhat_lower','yhat_upper','additive_terms','additive_terms_lower','additive_terms_upper','extra_regressors_additive','extra_regressors_additive_lower','extra_regressors_additive_upper','multiplicative_terms','multiplicative_terms_lower','multiplicative_terms_upper','yhat']]

# %%
mask_nifty50_close = (df_NIFTY50.Date_NI <= training_end_date)
df_NIFTY50_training = df_NIFTY50.loc[mask_nifty50_close,:]

#%%
df_NIFTY50_prediction = df_NIFTY50.loc[(df_NIFTY50.Date_NI > training_end_date),:]

# %%
df_NIFTY50_prediction['Date_NI'] = pd.to_datetime(df_NIFTY50_prediction['Date_NI'])

# %%
df_finall = pd.merge(df_NIFTY50_prediction,df_comparision, left_on='Date_NI', right_on='ds', how='inner')

# %%
starting_close_price = round(df_NIFTY50_training.tail(1)['Close'],2)[0]

# %%
df_final_utility_data = df_finall[['Close','Adj Close','Volume','ds','log_ret_^NSEI_y','trend','yhat_lower','yhat_upper','yhat']]

# %%
lst_predicted_close = []
lst_predicted_close_lower = []
lst_predicted_close_upper = []
calc_close_lower = starting_close_price
calc_close_upper = starting_close_price
calc_close = starting_close_price

for i in range(len(df_final_utility_data)):
    calc_close_lower = calc_close*(1+df_final_utility_data.loc[i,"yhat_lower"])
    calc_close_upper = calc_close*(1+df_final_utility_data.loc[i,"yhat_upper"])
    calc_close = calc_close*(1+df_final_utility_data.loc[i,"yhat"])
    lst_predicted_close.append(calc_close)
    lst_predicted_close_lower.append(calc_close_lower)
    lst_predicted_close_upper.append(calc_close_upper)

df_final_utility_data['pred_lower'] = lst_predicted_close_lower
df_final_utility_data['pred_upper'] = lst_predicted_close_upper
df_final_utility_data['pred_close'] = lst_predicted_close

# %%
df_final_utility_data['band'] = df_final_utility_data['pred_upper'] - df_final_utility_data['pred_lower']
df_final_utility_data['delta'] = df_final_utility_data['pred_close'] - df_final_utility_data['Close']

# %%
fig = px.line(df_final_utility_data, x=df_final_utility_data['ds'], y=df_final_utility_data['pred_close'])
fig.add_scatter(x=df_final_utility_data['ds'], y=df_final_utility_data['pred_upper'], mode='lines')
fig.add_scatter(x=df_final_utility_data['ds'], y=df_final_utility_data['pred_lower'], mode='lines')
fig.add_scatter(x=df_final_utility_data['ds'], y=df_final_utility_data['pred_lower'], mode='lines')
fig.show()
# %%
fig2 = px.line(df_final_utility_data, x=df_final_utility_data['ds'], y=df_final_utility_data['pred_close'])
fig2.add_scatter(x=df_final_utility_data['ds'], y=df_final_utility_data['Close'], mode='lines')
fig2.show()
# %%
df_final_utility_data.head(10)
# %%
delta_kurt = pd.Series(df_final_utility_data['delta']).kurtosis(skipna = True)
delta_mean = pd.Series(df_final_utility_data['delta']).mean(skipna = True)
delta_median = pd.Series(df_final_utility_data['delta']).median(skipna = True)
delta_std = df_final_utility_data['delta'].std()
df_key_metric = pd.DataFrame()

#%% 
lst_description = []
lst_value = []

#%%
lst_description.append("Kurstosis of prediction Delta")
lst_value.append(delta_kurt)
lst_description.append("Mean of prediction Delta")
lst_value.append(delta_mean)
lst_description.append("Median of prediction Delta")
lst_value.append(delta_median)
lst_description.append("Std. Deviation of prediction Delta")
lst_value.append(delta_std)

# %%
band_kurt = pd.Series(df_final_utility_data['band']).kurtosis(skipna = True)
band_mean = pd.Series(df_final_utility_data['band']).mean(skipna = True)
band_median = pd.Series(df_final_utility_data['band']).median(skipna = True)
band_std = df_final_utility_data['band'].std()

lst_description.append("Kurstosis of prediction Band")
lst_value.append(band_kurt)
lst_description.append("Mean of prediction Band")
lst_value.append(band_mean)
lst_description.append("Median of prediction Band")
lst_value.append(band_median)
lst_description.append("Std. Deviation of prediction Band")
lst_value.append(band_std)
#%%
bull_points = 0
bear_points = 0

# %% Continuous days of over prediction run
over_pred_run = 0
counter = 0
for index, row in df_final_utility_data.iterrows():
    if(row['delta']>0):
        counter+=1
    else:
        if(counter>over_pred_run):
            over_pred_run = counter
        counter = 0
lst_description.append("Continuous days for over prediction")
lst_value.append(over_pred_run)

#%% Continuous days of under prediction run
under_pred_run = 0
counter = 0
for index, row in df_final_utility_data.iterrows():
    if(row['delta']<0):
        counter+=1
    else:
        if(counter>under_pred_run):
            under_pred_run = counter
        counter = 0
lst_description.append("Continuous days for under prediction")
lst_value.append(under_pred_run)

# %% Average of positive returns
positive_average = round(df_final_utility_data[df_final_utility_data['log_ret_^NSEI_y']>0]['log_ret_^NSEI_y'].mean()*100,4)
lst_description.append("Average of positive returns")
lst_value.append(positive_average)

# %%
negative_average = round(df_final_utility_data[df_final_utility_data['log_ret_^NSEI_y']<0]['log_ret_^NSEI_y'].mean()*100,4)
lst_description.append("Average of Negative returns")
lst_value.append(negative_average)

# %% Continuous days of profits actual
profit_run = 0
counter = 0
for index, row in df_final_utility_data.iterrows():
    if(row['log_ret_^NSEI_y']>0):
        counter+=1
    else:
        if(counter>profit_run):
            profit_run = counter
        counter = 0
lst_description.append("Continuous days of profits actual")
lst_value.append(profit_run)

# %% continuous days of losses actual
losses_run = 0
counter = 0
for index, row in df_final_utility_data.iterrows():
    if(row['log_ret_^NSEI_y']<0):
        counter+=1
    else:
        if(counter>losses_run):
            losses_run = counter
        counter = 0
lst_description.append("Continuous days of losses actual")
lst_value.append(losses_run)

# %% continuous days of profit predicted
pred_profit_run = 0
counter = 0
for index, row in df_final_utility_data.iterrows():
    if(row['log_ret_^NSEI_y']>0):
        counter+=1
    else:
        if(counter>pred_profit_run):
            pred_profit_run = counter
        counter = 0
lst_description.append("Continuous days of profit predicted")
lst_value.append(pred_profit_run)

# %% continuous days of loss predicted
pred_loss_run = 0
counter = 0
for index, row in df_final_utility_data.iterrows():
    if(row['log_ret_^NSEI_y']<0):
        counter+=1
    else:
        if(counter>pred_loss_run):
            pred_loss_run = counter
        counter = 0
lst_description.append("Continuous days of losses predicted")
lst_value.append(pred_loss_run)