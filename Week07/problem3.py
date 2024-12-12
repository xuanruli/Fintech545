import pandas as pd
import numpy as np
from datetime import datetime
from scipy.optimize import minimize
from risk_management.returns import return_calculate

ff3 = pd.read_csv("F-F_Research_Data_Factors_daily.CSV")
mom = pd.read_csv("F-F_Momentum_Factor_daily.CSV")
prices = pd.read_csv("DailyPrices.csv")

rf = 0.05

prices['Date'] = pd.to_datetime(prices['Date'])
returns = return_calculate(prices, dateColumn="Date")
returns['Date'] = pd.to_datetime(returns['Date'])

ff3['Date'] = pd.to_datetime(ff3['Date'], format="%Y%m%d")
mom['Date'] = pd.to_datetime(mom['Date'], format="%Y%m%d")

ffData = pd.merge(ff3, mom, on="Date", how="inner")
ffData.columns = [col.strip() for col in ffData.columns]
print(ffData.head())
print(ffData.columns)
for col in ffData.columns[1:]:
    ffData[col] = ffData[col] / 100.0

stocks = ["AAPL", "META", "UNH", "MA", "MSFT", "NVDA", "HD", "PFE", "AMZN", "BRK-B",
          "PG", "XOM", "TSLA", "JPM", "V", "DIS", "GOOGL", "JNJ", "BAC", "CSCO"]

to_reg = pd.merge(returns[['Date'] + stocks], ffData, on='Date', how='inner')

xnames = ["Mkt-RF", "SMB", "HML", "Mom"]

X = np.column_stack([np.ones(to_reg.shape[0]), to_reg[xnames].values])
Y = to_reg[stocks].values - to_reg['RF'].values[:, None]

Betas = np.linalg.inv(X.T @ X) @ (X.T @ Y)
Betas = Betas.T

start_date = datetime(2014, 9, 30)
filtered = ffData[ffData['Date'] >= start_date]
means_factors = [filtered[x].mean() for x in xnames]
means = np.array([0.0] + means_factors)

stockMeans = np.log(1.0 + Betas @ means) * 255 + rf

ret_matrix = returns[stocks].values
log_returns = np.log(1.0 + ret_matrix)
covar = np.cov(log_returns, rowvar=False) * 255

def sr(w):
    w = np.array(w)
    m = w @ stockMeans - rf
    s = np.sqrt(w @ covar @ w)
    return m / s

def neg_sr(w):
    return -sr(w)

n = len(stocks)
constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
bounds = [(0.0, None) for _ in range(n)]

res = minimize(neg_sr, x0=np.ones(n)/n, bounds=bounds, constraints=constraints, method='SLSQP')
w_opt = res.x

icovar = np.linalg.inv(covar)
temp = icovar @ (stockMeans - rf)
w2 = temp / np.sum(temp)

OptWeights = pd.DataFrame({
    "Stock": stocks,
    "Weight": np.round(w_opt, 4),
    "Er": stockMeans,
    "UnconstWeight": w2
})

print(OptWeights)
print("Expected Return =", stockMeans @ w_opt)
print("Expected Vol =", np.sqrt(w_opt @ covar @ w_opt))
print("Expected SR =", sr(w_opt))
