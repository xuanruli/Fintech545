import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA
portfolio_df = pd.read_csv('portfolio.csv')
daily_price_df = pd.read_csv('DailyPrices.csv', index_col='Date', parse_dates=True)


def calculate_return(data):
    return_list = data.pct_change().dropna()
    return list(return_list)


def EWMA_var(stock_list, value, lambda_):
    variance = np.var(stock_list)
    for returns in stock_list:
        variance = lambda_*variance + (1-lambda_)*(returns**2)
    z_score = 1.645
    std = np.sqrt(variance)
    VaR = value * z_score * std
    return VaR

def AR1_var(stock_list, value):
    model =ARIMA(stock_list, order=(1, 0, 0))
    model_fit = model.fit()
    std = np.std(model_fit.resid)
    z_score = 1.645
    VaR = value * z_score * std
    return VaR

portfolios = portfolio_df['Portfolio'].unique()
df_total = pd.DataFrame()
for p in portfolios:
    portfolio_A_df = portfolio_df[portfolio_df['Portfolio'] == p]
    df_p = pd.DataFrame()
    for _, row in portfolio_A_df.iterrows():
        name = row['Stock']
        hold = row['Holding']
        if name in daily_price_df.columns:
            df_p[name] = daily_price_df[name]*hold
    df_p_series = df_p.sum(axis=1)
    df_total[p] = df_p_series
    PV = float(df_p_series.iloc[-1])
    df_p_returns = calculate_return(df_p_series)
    Var = EWMA_var(df_p_returns, PV, 0.94)
    print(f'Var based on EWMA for portfolio {p} is {Var}')
df_total_series = df_total.sum(axis=1)
PV_total = float(df_total_series.iloc[-1])
df_total_returns = calculate_return(df_total_series)
Var_total = EWMA_var(df_total_returns, PV_total, 0.94)
print(f'Var based on EWMA for total portfolio is {Var_total}')


df_total = pd.DataFrame()
for p in portfolios:
    portfolio_A_df = portfolio_df[portfolio_df['Portfolio'] == p]
    df_p = pd.DataFrame()
    for _, row in portfolio_A_df.iterrows():
        name = row['Stock']
        hold = row['Holding']
        if name in daily_price_df.columns:
            df_p[name] = daily_price_df[name]*hold
    df_p_series = df_p.sum(axis=1)
    df_total[p] = df_p_series
    PV = float(df_p_series.iloc[-1])
    df_p_returns = calculate_return(df_p_series)
    Var = AR1_var(df_p_returns, PV)
    print(f'Var based on AR1 for portfolio {p} is {Var}')
df_total_series = df_total.sum(axis=1)
PV_total = float(df_total_series.iloc[-1])
df_total_returns = calculate_return(df_total_series)
Var_total = AR1_var(df_total_returns, PV_total)
print(f'Var based on AR1 for total portfolio is {Var_total}')


