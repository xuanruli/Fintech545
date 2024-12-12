import pandas as pd
import numpy as np
from datetime import datetime
from risk_management.returns import return_calculate
from risk_management.var import aggRisk
from option import bt_american, find_zero_bt


portfolio = pd.read_csv("problem2.csv")

currentDate = datetime(2023,3,3)
divDate = datetime(2023,3,15)
div = 1.00
currentS = 165
rf = 0.0425
mult = 5
daysDiv = (divDate - currentDate).days

prices = pd.read_csv("DailyPrices.csv", usecols=['Date','AAPL'])
prices['Date'] = pd.to_datetime(prices['Date']).dt.date

rets = return_calculate(prices, dateColumn="Date")
returns = rets['AAPL'].values
returns = returns - np.mean(returns)
sd = np.std(returns)

def parse_expiry(x):
    if isinstance(x, str):
        m,d,y = x.split('/')
        return datetime(int(y), int(m), int(d))
    return x

portfolio['ExpirationDate'] = portfolio.apply(lambda row: parse_expiry(row['ExpirationDate']) if row['Type']=='Option' else pd.NaT, axis=1)

portfolio['IV'] = np.nan
for i in range(len(portfolio)):
    if portfolio.at[i,'Type'] == 'Option':
        call_type = portfolio.at[i, 'OptionType'] == 'Call'
        T = (portfolio.at[i, 'ExpirationDate'] - currentDate).days / 365.0
        market_price =  portfolio.at[i, 'CurrentPrice']
        result = find_zero_bt(call_type, currentS, portfolio.at[i, 'Strike'], T, rf, [div],[daysDiv*mult], int(T*365*mult),market_price)
        portfolio.at[i,'IV'] = result
    else:
        portfolio.at[i,'IV'] = np.nan

def bt_delta(call, s, strike, ttm, rf, div, divPoint, iv, NPoints):
    h = 0.0001
    if call:
        am_call_up = bt_american(True, s + h, strike, ttm, rf, div, divPoint, iv, NPoints)
        am_call_dn = bt_american(True, s - h, strike, ttm, rf, div, divPoint, iv, NPoints)
        return (am_call_up - am_call_dn) / (2 * h)
    else:
        am_put_up = bt_american(False, s + h, strike, ttm, rf, div, divPoint, iv, NPoints)
        am_put_dn = bt_american(False, s - h, strike, ttm, rf, div, divPoint, iv, NPoints)
        return (am_put_up - am_put_dn) / (2 * h)

deltas = []
for i in range(len(portfolio)):
    if portfolio.at[i,'Type'] == 'Option':
        T = (portfolio.at[i,'ExpirationDate'] - currentDate).days / 365.0
        delta_val = bt_delta(portfolio.at[i,'OptionType'] =='Call',
                             currentS,
                             portfolio.at[i,'Strike'],
                             T,
                             rf,
                             [div], [daysDiv*mult],
                             portfolio.at[i,'IV'],
                             int(T*365*mult))*portfolio.at[i,'Holding']*currentS
        deltas.append(delta_val)
    else:
        deltas.append(portfolio.at[i,'Holding']*currentS)
portfolio['Delta'] = deltas

print(portfolio)


nSim = 100
fwdT = 10
_simReturns = np.random.normal(0, sd, nSim*fwdT)

simPrices = np.zeros(nSim)
for i in range(nSim):
    r = 1.0
    for j in range(fwdT):
        r *= (1+_simReturns[i*fwdT+j])
    simPrices[i] = currentS*r

iteration = np.arange(1,nSim+1)
values = portfolio.merge(pd.DataFrame({'iteration':iteration}), how='cross')
nVals = len(values)

def fwd_ttm(row):
    if row['Type'] == 'Option':
        return ((row['ExpirationDate'] - currentDate).days - fwdT)/365.0
    else:
        return np.nan

values['fwd_ttm'] = values.apply(fwd_ttm, axis=1)

simulatedValue = np.zeros(nVals)
currentValue = np.zeros(nVals)
pnl = np.zeros(nVals)

for i in range(nVals):
    simprice = simPrices[values.at[i,'iteration']-1]
    currentValue[i] = values.at[i,'Holding'] * values.at[i,'CurrentPrice']
    if values.at[i,'Type'] == 'Option':
        T = values.at[i,'fwd_ttm']
        if T < 0:
            T = 0
        simulatedValue[i] = values.at[i,'Holding'] * bt_american(values.at[i,'OptionType'] =='Call',
                                                                 simprice,
                                                                 values.at[i,'Strike'],
                                                                 T,
                                                                 rf,
                                                                 [div], [(daysDiv - fwdT)*mult],
                                                                 values.at[i,'IV'],
                                                                 int(T*365*mult) if T>0 else 0)
    else:
        simulatedValue[i] = values.at[i,'Holding']*simprice

    pnl[i] = simulatedValue[i] - currentValue[i]

values['simulatedValue'] = simulatedValue
values['pnl'] = pnl
values['currentValue'] = currentValue


risk = aggRisk(values, ['Portfolio'])

print(risk)
