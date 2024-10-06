import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA
portfolio_df = pd.read_csv('portfolio.csv')
daily_price_df = pd.read_csv('DailyPrices.csv', index_col='Date', parse_dates=True)
# 遍历每个组合
portfolios = portfolio_df['Portfolio'].unique()
portfolio_values = {}

for portfolio in portfolios:
    portfolio_data = portfolio_df[portfolio_df['Portfolio'] == portfolio]
    portfolio_value = pd.Series(index=daily_price_df.index, dtype=float)

    for _, row in portfolio_data.iterrows():
        stock = row['Stock']
        holding = row['Holding']
        if stock in daily_price_df.columns:
            portfolio_value += daily_price_df[stock] * holding
    portfolio_values[portfolio] = portfolio_value

print(portfolio_values)






