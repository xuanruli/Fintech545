import pandas as pd
import numpy as np
from risk_management.regression import fit_general_t, fit_normal
from risk_management.returns import return_calculate
from scipy.stats import t, norm
from risk_management.var import ESS, VaR

def simulate_pca(a, nsim, nval=None):
    eigvals, eigvecs = np.linalg.eigh(a)
    flip = np.arange(len(eigvals)-1, -1, -1)
    vals = eigvals[flip]
    vecs = eigvecs[:, flip]
    tv = np.sum(vals)
    posv = np.where(vals >= 1e-8)[0]
    if nval is not None:
        if nval < len(posv):
            posv = posv[:nval]
    vals = vals[posv]
    vecs = vecs[:, posv]
    explained = np.sum(vals) / tv * 100
    B = vecs @ np.diag(np.sqrt(vals))

    m = len(vals)
    r = np.random.randn(m, nsim)
    sim_data = (B @ r).T
    return sim_data

def aggRisk(values, group_cols):
    alpha = 0.05
    results = []
    groups = values.groupby(group_cols)
    for gname, gdf in groups:
        portfolio_pnl = gdf['pnl'].values
        Var95 = VaR(portfolio_pnl, alpha=alpha)
        ES95 = ESS(portfolio_pnl, alpha=alpha)
        if isinstance(gname, tuple):
            gname = gname[0]
        results.append((gname, Var95, ES95))
    out = pd.DataFrame(results, columns=group_cols+['VaR95','ES95'])
    return out
prices = pd.read_csv("DailyPrices.csv")
returns = return_calculate(prices, dateColumn="Date")
returns = returns.drop(columns=["Date"])
rnames = returns.columns
currentPrice = prices.iloc[-1, :]
portfolio = pd.read_csv("portfolio.csv")
stocks = portfolio['Stock']
tStocks = portfolio.loc[portfolio['Portfolio'].isin(["A","B"]), 'Stock']
nStocks = portfolio.loc[portfolio['Portfolio'].isin(["C"]), 'Stock']
for nm in stocks:
    v = returns[nm]
    returns[nm] = v - v.mean()
fittedModels = {}
for s in tStocks:
    fittedModels[s] = fit_general_t(returns[s].values)
for s in nStocks:
    fittedModels[s] = fit_normal(returns[s].values)
U = pd.DataFrame(index=returns.index)
for nm in stocks:
    U[nm] = fittedModels[nm]["u"]
R = U.corr(method='spearman').values
evals = np.linalg.eigvals(R)
if np.min(evals) > -1e-8:
    print("Matrix is PSD")
else:
    print("Matrix is not PSD")

NSim = 5000
simZ = simulate_pca(R, NSim)
simU = pd.DataFrame(norm.cdf(simZ), columns=stocks)
simulatedReturns = pd.DataFrame(index=range(NSim), columns=stocks)
for stock in stocks:
    simulatedReturns[stock] = simU[stock].apply(fittedModels[stock]["eval"])

def calcPortfolioRisk(simulatedReturns, NSim):
    iteration = pd.DataFrame({'iteration': range(1, NSim+1)})
    portfolio['key'] = 1
    iteration['key'] = 1
    values = pd.merge(portfolio, iteration, on='key').drop('key', axis=1)
    currentValue = []
    simulatedValue = []
    pnl = []
    for i, row in values.iterrows():
        price = currentPrice[row['Stock']]
        cv = row['Holding'] * price
        sr = simulatedReturns.loc[row['iteration']-1, row['Stock']]
        sv = row['Holding'] * price * (1.0 + sr)
        currentValue.append(cv)
        simulatedValue.append(sv)
        pnl.append(sv - cv)

    values['currentValue'] = currentValue
    values['simulatedValue'] = simulatedValue
    values['pnl'] = pnl
    values['Portfolio'] = values['Portfolio'].astype(str)
    risk_df = aggRisk(values, ['Portfolio'])
    return risk_df

risk = calcPortfolioRisk(simulatedReturns, NSim)
print(risk)

def EWMA_var(stock_list, lambda_):
    variance = np.var(stock_list)
    for returns in stock_list:
        variance = lambda_*variance + (1-lambda_)*(returns**2)
    return variance
returns_matrix = returns.values
def ewCovar(returns, lambda_):
    n, m = returns.shape
    weights = np.array([(1 - lambda_) * (lambda_ ** i) for i in range(n)])
    weights = weights[::-1]
    weights /= weights.sum()

    mean_adjusted = returns - returns.mean(axis=0)
    ew_covar = mean_adjusted.T @ (weights[:, np.newaxis] * mean_adjusted)
    return ew_covar

covar = ewCovar(returns_matrix, 0.97)
simulatedReturns = pd.DataFrame(simulate_pca(covar, NSim), columns=rnames)
risk_n = calcPortfolioRisk(simulatedReturns, NSim)
print(risk_n)