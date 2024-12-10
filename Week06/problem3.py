from risk_management.returns import return_calculate
import pandas as pd
from datetime import datetime,date
import numpy as np
from risk_management.option import find_zero,gbsm
from statsmodels.tsa.arima.model import ARIMA
from risk_management.var import aggRisk
from types import SimpleNamespace

currentS = 170.15
current_dt = date(2023, 10, 30)
rf = 0.0525
dy = 0.0057

daily_prices = pd.read_csv("DailyPrices.csv")

daily_prices = return_calculate(daily_prices, method="LOG", dateColumn="Date")
returns = daily_prices["AAPL"].dropna().values

returns = returns - np.mean(returns)
sd = np.std(returns)

portfolio = pd.read_csv("problem3.csv")

def parse_date(x):
    return datetime.strptime(x, "%m/%d/%Y").date()

portfolio['ExpirationDate'] = portfolio.apply(
    lambda row: parse_date(row['ExpirationDate']) if row['Type'] == 'Option' else None, axis=1
)

implied_vols = []
for i, row in portfolio.iterrows():
    if row['Type'] == 'Option':
        call_type = (row['OptionType'] == "Call")
        strike = row['Strike']
        ttm = (row['ExpirationDate'] - current_dt).days / 365.0
        market_price = row['CurrentPrice']
        iv = find_zero(call_type, currentS, strike, ttm, rf, dy, market_price)
        implied_vols.append(iv)
    else:
        implied_vols.append(np.nan)

portfolio['ImpVol'] = implied_vols

model = ARIMA(returns, order=(1,0,0))
ar1_result = model.fit()
ar1_params = ar1_result.params

def ar1_simulation(y, coef, innovations, ahead=1):
    m = coef[0]
    a1 = coef[1]
    sigma2 = coef[2]
    s = np.sqrt(sigma2)

    l = len(y)
    n = len(innovations)//ahead

    out = np.zeros((ahead,n))

    y_last = y[-1] - m
    idx = 0
    for i in range(n):
        yl = y_last
        for j in range(ahead):
            eps = innovations[idx]
            next_val = a1*yl + s*eps
            yl = next_val
            out[j,i] = next_val
            idx += 1
    out = out + m
    return out

nSim = 10000
fwdT = 10
innovations = np.random.randn(fwdT*nSim)
arSim = ar1_simulation(returns, ar1_params, innovations, ahead=fwdT)

simReturns = np.sum(arSim, axis=0)
simPrices = currentS * np.exp(simReturns)

iteration = np.arange(1, nSim+1)
values = portfolio.loc[portfolio.index.repeat(nSim)].copy()
values['iteration'] = np.tile(iteration, len(portfolio))

def fwd_ttm(row):
    if row['Type'] == 'Option':
        ttm_days = (row['ExpirationDate'] - current_dt).days - fwdT
        return ttm_days / 365.0
    else:
        return np.nan

values['fwd_ttm'] = values.apply(fwd_ttm, axis=1)

simulatedValue = []
currentValue = []
pnl = []

for i, row in values.iterrows():
    simprice = simPrices[row['iteration'] - 1]

    cVal = row['Holding'] * row['CurrentPrice']
    currentValue.append(cVal)
    if row['Type'] == 'Option':
        val = row['Holding'] * gbsm(row['OptionType'] == "Call",
                                    simprice,
                                    row['Strike'],
                                    row['fwd_ttm'],
                                    rf, dy, row['ImpVol'])
        simulatedValue.append(val)
    elif row['Type'] == 'Stock':
        val = row['Holding'] * simprice
        simulatedValue.append(val)

    pnl.append(simulatedValue[-1] - cVal)

values['simulatedValue'] = simulatedValue
values['pnl'] = pnl
values['currentValue'] = currentValue

risk_df = aggRisk(values, ['AAPL_Options.csvPortfolio'])
print(risk_df)

import matplotlib.pyplot as plt
import numpy as np

portfolios = portfolio['AAPL_Options.csvPortfolio'].unique()

underlying_range = np.linspace(0.5 * currentS, 1.5 * currentS, 100)

for port_name in portfolios:
    sub_portfolio = portfolio[portfolio['AAPL_Options.csvPortfolio'] == port_name]

    portfolio_values = []
    for S in underlying_range:
        total_val = 0.0
        for i, row in sub_portfolio.iterrows():
            holding = row['Holding']
            if row['Type'] == 'Stock':
                total_val += holding * S
            elif row['Type'] == 'Option':
                strike = row['Strike']
                if row['OptionType'] == 'Call':
                    payoff = max(S - strike, 0.0)
                else:
                    payoff = max(strike - S, 0.0)
                total_val += holding * payoff
        portfolio_values.append(total_val)

    plt.figure()
    plt.plot(underlying_range, portfolio_values, label=port_name)
    plt.title(f'Payoff Diagram at Expiration for {port_name}')
    plt.xlabel('Underlying Price at Expiration')
    plt.ylabel('Portfolio Value')
    plt.grid(True)
    plt.legend()
    plt.show()
