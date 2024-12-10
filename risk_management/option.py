import numpy as np
from scipy.stats import norm
from scipy.optimize import root_scalar

def gbsm(call_type, current_price, strike, ttm, rf, cost_of_carry, iv):
    b = rf - cost_of_carry
    d1 = (np.log(current_price/strike)+(b+iv**2/2)*ttm)/(iv/np.sqrt(ttm))
    d2 = d1 - iv/np.sqrt(ttm)
    if call_type:
        value = current_price*norm.cdf(d1) - strike*norm.cdf(d2)*np.exp(-rf*ttm)
    else:
        value = strike*np.exp(-rf*ttm)*norm.cdf(-d2) - current_price*norm.cdf(-d1)
    return value

def find_zero(call_type, current_price, strike, ttm, rf, cost_of_carry,market_price):
    def objective(iv):
        root = gbsm(call_type, current_price,strike, ttm, rf, cost_of_carry, iv) - market_price
        return root
    result = root_scalar(objective, bracket=[-3, 3], method='brentq')
    return result.root