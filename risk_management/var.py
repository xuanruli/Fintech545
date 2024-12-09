from scipy.stats import norm
from scipy.stats import t as tdist
from scipy.optimize import minimize
import numpy as np

def ES(distribution, alpha=0.05):
    dist_name = distribution.dist.name
    if dist_name == 'norm':
        mu = distribution.mean()
        sigma = distribution.std()
        z_alpha = norm.ppf(alpha)
        pdf_z_alpha = norm.pdf(z_alpha)
        ES = mu - sigma * pdf_z_alpha / alpha
        return -ES

    elif dist_name == 't':
        mu = distribution.kwds["loc"]
        sigma = distribution.kwds["scale"]
        nu = distribution.kwds['df']
        t_alpha = tdist.ppf(alpha, df=nu)
        pdf_t_alpha = tdist.pdf(t_alpha, df=nu)
        ES_val = mu - sigma * ((nu + t_alpha ** 2) / (nu - 1)) * (pdf_t_alpha / alpha)
        return -ES_val
    else:
        raise ValueError("Unsupported distribution type")
def ESS(obj, alpha=0.05):
    xs = np.sort(obj)
    cutoff = int(np.ceil(alpha * len(xs)))
    ES_val = xs[:cutoff].mean()
    return -ES_val
def VaR(obj, alpha=0.05):
    xs = np.sort(obj)
    cutoff = int(np.ceil(alpha * len(xs)))
    Var = xs[cutoff]
    return -Var

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

def historic_var(stock_list, value = 1):
    sort_list = np.sort(stock_list)
    var_return = np.percentile(sort_list, 5)
    VaR = value * abs(var_return)
    return VaR