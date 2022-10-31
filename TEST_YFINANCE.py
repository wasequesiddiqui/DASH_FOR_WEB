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

#%%
df_overall_model = get_complete_model_df(df_main= df_NIFTY50_mod, df_others= [df_bist_mod,df_dji_mod, df_nya_mod])

# %%
df_overall_model.rename(columns = {'Date_NI':'DS'}, inplace = True)

# %%
start_date = (datetime.now() - relativedelta(years=5)).date()
training_end_date = (datetime.now() - relativedelta(days=60)).date()
training_end_date_slice = (datetime.now() - relativedelta(days=120)).date()

mask_training_overall = (df_overall_model.DS <= training_end_date)
mask_prediction = (df_overall_model.DS > training_end_date)
df_training_overall = df_overall_model.loc[mask_training_overall,:]
df_prediction = df_overall_model.loc[mask_prediction,:]

holiday_dates = []

# %%
for date in rrule(DAILY, dtstart=start_date, until=datetime.now()):
    if((date.strftime('%A')=="Saturday") or (date.strftime('%A')=="Sunday")):
        holiday_dates.append(date.date())

# %%
df_holidays = pd.DataFrame()
df_holidays['ds'] = holiday_dates
df_holidays['holiday'] = "Weekend"
df_training_overall_tuning = df_training_overall[['DS','log_ret_^NSEI']]
df_training_overall_tuning.rename(columns = {'log_ret_^NSEI':'y','DS':'ds'}, inplace = True)
df_training_overall_tuning['ds'] = pd.to_datetime(df_training_overall_tuning['ds'])

#%%
# params = get_tuned_param(df_training_overall_tuning, (datetime.now() - relativedelta(years=5)))

# %%
model = p.Prophet(holidays=df_holidays)
model.add_regressor('log_ret_XU100.IS')
model.add_regressor('log_ret_^DJI')
model.add_regressor('log_ret_^NYA')
df_training_overall.rename(columns = {'log_ret_^NSEI':'y','DS':'ds'}, inplace = True)

df_training_overall['log_ret_XU100.IS'] = df_training_overall['log_ret_XU100.IS'].interpolate(method='spline', order=1, limit=10, limit_direction='both')
df_training_overall['log_ret_^DJI'] = df_training_overall['log_ret_^DJI'].interpolate(method='spline', order=1, limit=10, limit_direction='both')
df_training_overall['log_ret_^NYA'] = df_training_overall['log_ret_^NYA'].interpolate(method='spline', order=1, limit=10, limit_direction='both')

model.fit(df_training_overall)
# %%
df_prediction.rename(columns = {'DS':'ds'}, inplace = True)
df_prediction.drop(['log_ret_^NSEI'], axis=1)
df_prediction['log_ret_XU100.IS'] = df_prediction['log_ret_XU100.IS'].interpolate(method='spline', order=1, limit=10, limit_direction='both')
df_prediction['log_ret_^DJI'] = df_prediction['log_ret_^DJI'].interpolate(method='spline', order=1, limit=10, limit_direction='both')
df_prediction['log_ret_^NYA'] = df_prediction['log_ret_^NYA'].interpolate(method='spline', order=1, limit=10, limit_direction='both')
df_forecast = model.predict(df_prediction)
# %%
df_forecast.tail()
# %%
fig1 = model.plot(df_forecast)
fig2 = model.plot_components(df_forecast)
# %%
