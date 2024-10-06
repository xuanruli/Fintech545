import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA

data = pd.read_csv('DailyPrices.csv')

def calculate_return(data, company_name):
    if company_name not in data.columns:
        raise ValueError

    price_list = data[company_name]
    return_list = price_list.pct_change().dropna()
    return list(return_list)

def remove_mean(stock_list):
    mean = sum(stock_list)/ len(stock_list)
    new_list = [x - mean for x in stock_list]
    return new_list

 # delta normal
def delta_normal_var(stock_list, value):
    z_score = 1.645
    std = float(np.std(stock_list, ddof=1))
    VaR = value * z_score * std
    return VaR

#EWMA
def EWMA_var(stock_list, value, lambda_):
    variance = np.var(stock_list)
    for returns in stock_list:
        variance = lambda_*variance + (1-lambda_)*(returns**2)
    z_score = 1.645
    std = np.sqrt(variance)
    VaR = value * z_score * std
    return VaR

def MLE_t_var(stock_list, value):
    stock_list = np.array(stock_list)
    def neg_log_likelihood(params, data):
        nu, mu, sigma = params
        return -np.sum(stats.t.logpdf(data, df=nu, loc=mu, scale=sigma))

    initial_params = [5, np.mean(stock_list), np.std(stock_list)]
    result = minimize(neg_log_likelihood, initial_params, args=(stock_list,),
                      bounds=((2.01, None), (None, None), (1e-6, None)))

    nu_mle, mu_mle, sigma_mle = result.x
    confidence_level = 0.95
    z_score_t_dist = stats.t.ppf(1 - confidence_level, df=nu_mle)
    VaR = value * -z_score_t_dist * sigma_mle
    return VaR

def AR1_var(stock_list, value):
    model =ARIMA(stock_list, order=(1, 0, 0))
    model_fit = model.fit()
    std = np.std(model_fit.resid)
    z_score = 1.645
    VaR = value * z_score * std
    return VaR

def historic_var(stock_list, value):
    sort_list = np.sort(stock_list)
    var_return = np.percentile(sort_list, 5)
    VaR = value * abs(var_return)
    return VaR

meta_list = calculate_return(data, 'META')
meta_list_remove = remove_mean(meta_list)
meta_price = data['META'].iloc[-1]
meta_delta_var = delta_normal_var(meta_list_remove, meta_price)
meta_EWMA_var = EWMA_var(meta_list_remove, meta_price, 0.93)
meta_MLE_var = MLE_t_var(meta_list_remove, meta_price)
meta_AR1_var = AR1_var(meta_list_remove, meta_price)
meta_hist_var = historic_var(meta_list, meta_price)
print(f"Delta-Normal VaR: ${meta_delta_var:.2f}")
print(f"EWMA VaR: ${meta_EWMA_var:.2f}")
print(f"MLE t-distribution VaR: ${meta_MLE_var:.2f}")
print(f"AR(1) VaR: ${meta_AR1_var:.2f}")
print(f"Historic Simulation VaR: ${meta_hist_var:.2f}")


