from scipy.stats import norm
from scipy.stats import t as tdist
from scipy.optimize import minimize
import numpy as np

def fit_normal(x):
    mu = np.mean(x)
    sigma = np.std(x)

    error_model = norm(loc=mu, scale=sigma)
    errors = x - mu
    u = error_model.cdf(x)
    def eval(u):
        return error_model.ppf(u)
    return {
        "mu": mu,
        "sigma": sigma,
        "error_model": error_model,
        "eval": eval,
        "errors": errors,
        "u": u
    }

def fit_general_t(x):
    start_m = np.mean(x)
    start_nu = 6.0 / ((np.mean(x ** 4) / (np.var(x) ** 2)) - 3) + 4
    start_s = np.sqrt(np.var(x) * (start_nu - 2) / start_nu)
    def general_t_ll(params):
        mu, s, nu = params
        if s < 1e-6 or nu < 2.0001:
            return np.inf
        pdf_values = tdist.pdf((x - mu) / s, df=nu) / s
        return -np.sum(np.log(pdf_values))
    initial_guess = [start_m, start_s, start_nu]
    bounds = [(None, None), (1e-6, None), (2.0001, None)]
    result = minimize(general_t_ll, initial_guess, bounds=bounds, method="L-BFGS-B")
    m, s, nu = result.x
    error_model = tdist(df=nu, loc=m, scale=s)
    cdf_func = lambda y: error_model.cdf(y)
    quantile_func = lambda u: error_model.ppf(u)
    errors = x - m
    u = cdf_func(x)
    def eval(u):
        return quantile_func(u)
    return {
        "m": m,
        "s": s,
        "nu": nu,
        "errors": errors,
        "u": u,
        "error_model": error_model,
        "cdf_func": cdf_func,
        "quantile_func": quantile_func,
        "eval": eval
    }


def fit_regression_t(y, x):
    X = np.hstack([np.ones((len(x), 1)), x])
    b_start = np.linalg.inv(X.T @ X) @ (X.T @ y)
    residuals = y - X @ b_start
    start_m = np.mean(residuals)
    start_nu = 6.0 / ((np.mean(residuals ** 4) / np.var(residuals) ** 2) - 3) + 4
    start_s = np.sqrt(np.var(residuals) * (start_nu - 2) / start_nu)
    def negative_log_likelihood(params):
        m, s, nu = params[:3]
        beta = params[3:]
        residuals = (y - X @ beta - m) / s
        if s <= 0 or nu <= 2.0001:
            return np.inf
        log_likelihood = np.sum(tdist.logpdf(residuals, df=nu)) - len(y) * np.log(s)
        return -log_likelihood

    initial_params = np.concatenate(([start_m, start_s, start_nu], b_start))
    bounds = [(None, None), (1e-6, None), (2.0001, None)] + [(None, None)] * X.shape[1]
    result = minimize(negative_log_likelihood, initial_params, bounds=bounds, method="L-BFGS-B")

    m, s, nu = result.x[:3]
    beta = result.x[3:]
    error_model = tdist(df=nu, loc=m, scale=s)
    def eval_model(new_x, u):
        new_X = np.hstack([np.ones((new_x.shape[0], 1)), new_x])
        quantile_correction = error_model.ppf(u)
        return new_X @ beta + quantile_correction
    errors = y - eval_model(x, np.full(len(y), 0.5))
    u = error_model.cdf(errors)

    return {
        "m": m,
        "s": s,
        "nu": nu,
        "beta": beta,
        "error_model": error_model,
        "eval_model": eval_model,
        "errors": errors,
        "u": u
    }

