import pandas as pd
from risk_management.regression import fit_general_t
from risk_management.var import ES,historic_var,ESS
import numpy as np
from scipy.stats import t, norm
def EWMA_var(stock_list, lambda_):
    variance = np.var(stock_list)
    for returns in stock_list:
        variance = lambda_*variance + (1-lambda_)*(returns**2)
    return variance

data = pd.read_csv("problem1.csv")["x"]
var = EWMA_var(data, 0.97)
mu = mean = data.mean()
dist = norm(mu, np.sqrt(var))
ES1 = ES(dist)
VaR1 = -dist.ppf(0.05)
print("normal distribution's ES and VaR:", ES1, VaR1)


fd = fit_general_t(data)
dist = fd["error_model"]
ES2 = ES(dist)
VaR2 = -dist.ppf(0.05)
print("t distribution's ES and VaR:", ES2, VaR2)

VaR3 = historic_var(data)
ES3 = ESS(data)
print("historical's ES and VaR:", ES3, VaR3)
