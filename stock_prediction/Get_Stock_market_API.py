"""
Sample API stock price predictions

Author: Quoc Thinh Vo
Last Edited: 04/ 05/ 2021

"""
from datetime import datetime, timedelta
import pandas as pd
from tqdm import tqdm
import time

import alpaca_trade_api as tradeapi

def break_period_in_dates_list(start_date, end_date, days_per_step):
    '''Break period between start_date and end_date in steps of days_per_step days.'''
    step_start_date = start_date
    delta = timedelta(days=days_per_step)
    dates_list = []
    while end_date > (step_start_date + delta):
        dates_list.append((step_start_date, step_start_date+delta))
        step_start_date += delta
    dates_list.append((step_start_date, end_date))
    return dates_list


def format_timestep_list(timestep_list):
    '''Format dates in ISO format plus timezone. Note that first day starts at 00:00 and last day ends at 23:00hs.'''
    for i, d in enumerate(timestep_list):
        timestep_list[i] = (d[0].isoformat() + '-04:00', (d[1].isoformat().split('T')[0] + 'T23:00:00-04:00'))
    return timestep_list


def get_df_from_barset(barset):
    '''Create a Pandas Dataframe from a barset.'''
    df_rows = []
    for symbol, bar in barset.items():
        rows = bar.__dict__.get('_raw')
        for i, row in enumerate(rows):
            row['symbol'] = symbol
        df_rows.extend(rows)

    return pd.DataFrame(df_rows)


def download_data(aps, symbols, start_date, end_date, filename='data.csv'):
    '''Download data from REST manager for list of symbols, from start_date at 00:00hs to end_date at 23:00hs,
    and save it to filename as a csv file.'''
    timesteps = format_timestep_list(break_period_in_dates_list(start_date, end_date, 10))
    df = pd.DataFrame()
    
    for timestep in tqdm(timesteps):
        barset = aps.get_barset(symbols, '5Min', limit=1000, start=timestep[0], end=timestep[1])
        df = df.append(get_df_from_barset(barset))
        time.sleep(0.1)
        
    df.to_csv(filename)
    
    
# Call method.
aps = tradeapi.REST(key_id = 'USER_KEY', 
                    secret_key = 'USER_SECRET_KEY',
                    base_url = 'https://paper-api.alpaca.markets')

download_data(aps        = aps, 
              symbols    = ['AAPL',], 
              start_date = datetime.strptime('01/01/20', '%d/%m/%y'), 
              end_date   = datetime.strptime('01/02/20', '%d/%m/%y'),
              filename   = 'test.csv')

#%%%

#Processing data now

import pickle
import pandas as pd
import numpy as np

with open('test.csv', 'rb') as f:
    df= pd.read_csv(f)    
DATA = df

X_data = DATA.drop(columns=['c','symbol']).to_numpy()

y_data = DATA['c'].copy().to_numpy()

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data,test_size=0.20,random_state=0)

from sklearn.preprocessing import StandardScaler
estimators = []
estimators.append(('standardize', StandardScaler()))
#estimators.append(('lda', LinearDiscriminantAnalysis()))
num_pipeline = Pipeline(estimators)
num_pipeline = num_pipeline.fit_transform(X_train)

from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor()
forest.fit(X_train, y_train)
predictions = forest.predict(X_test)

from sklearn.metrics import mean_squared_error
final_mse_RND = mean_squared_error(y_test, predictions)
final_rmse_RND = np.sqrt(final_mse_RND)
forest_score = forest.score(X_test, y_test)

from sklearn.linear_model import LinearRegression
linear = LinearRegression()
linear.fit(X_train, y_train)
predictions = linear.predict(X_test)

final_mse_linear = mean_squared_error(y_test, predictions)
final_rmse_linear = np.sqrt(final_mse_linear )
linear_score = linear.score(X_test, y_test)
