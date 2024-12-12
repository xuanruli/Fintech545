import numpy as np
from scipy.stats import norm
from scipy.optimize import root_scalar, minimize


def gbsm(call: bool, S, K, T, r, b, sigma, Greek=True):
    d1 = (np.log(S / K) + (b + (sigma ** 2) / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    Nd1 = norm.cdf(d1)
    Nd2 = norm.cdf(d2)
    nd1 = norm.pdf(d1)

    if call:
        value = S * np.exp((b - r) * T) * Nd1 - K * np.exp(-r * T) * Nd2
        delta = np.exp((b - r) * T) * Nd1
        cRho = T * S * np.exp((b - r) * T) * Nd1
    else:
        value = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp((b - r) * T) * norm.cdf(-d1)
        delta = np.exp((b - r) * T) * (Nd1 - 1)
        cRho = -T * S * np.exp((b - r) * T) * norm.cdf(-d1)

    if not Greek:
        return value

    gamma = (np.exp((b - r) * T) * nd1) / (S * sigma * np.sqrt(T))
    vega = S * np.exp((b - r) * T) * nd1 * np.sqrt(T)
    if call:
        theta = - (S * np.exp((b - r) * T) * nd1 * sigma) / (2 * np.sqrt(T)) - (b - r) * S * np.exp(
            (b - r) * T) * Nd1 - r * K * np.exp(-r * T) * Nd2
    else:
        Nnegd1 = norm.cdf(-d1)
        Nnegd2 = norm.cdf(-d2)
        theta = - (S * np.exp((b - r) * T) * nd1 * sigma) / (2 * np.sqrt(T)) + (b - r) * S * np.exp(
            (b - r) * T) * Nnegd1 + r * K * np.exp(-r * T) * Nnegd2

    return {
        'value': value,
        'delta': delta,
        'gamma': gamma,
        'vega': vega,
        'theta': theta,
        'cRho': cRho
    }

def bt_american(call, S, K, T, r, divAmts, divDays, sigma, N):
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = np.exp(-sigma * np.sqrt(dt))
    q = (np.exp(r * dt) - d) / (u - d)

    stock = np.zeros((N + 1, N + 1))
    stock[0, 0] = S

    for i in range(1, N + 1):
        stock[0, i] = stock[0, i - 1] * u
        for j in range(1, i + 1):
            stock[j, i] = stock[j - 1, i - 1] * d

        if i in divDays:
            idx = divDays.index(i)
            amt = divAmts[idx]
            for j in range(i + 1):
                stock[j, i] -= amt

    opt = np.zeros((N + 1, N + 1))
    if call:
        opt[:, N] = np.maximum(stock[:, N] - K, 0)
    else:
        opt[:, N] = np.maximum(K - stock[:, N], 0)

    discount = np.exp(-r * dt)
    for i in reversed(range(N)):
        for j in range(i + 1):
            exercise = (stock[j, i] - K if call else (K - stock[j, i]))
            hold = discount * (q * opt[j, i + 1] + (1 - q) * opt[j + 1, i + 1]) #value
            opt[j, i] = max(exercise, hold)

    return opt[0, 0]

def finite_diff_gradient(f, x, h=1e-6):
    grad = np.zeros_like(x)
    for i in range(len(x)):
        xph = x.copy()
        xph[i] += h
        xmh = x.copy()
        xmh[i] -= h
        grad[i] = (f(xph) - f(xmh)) / (2 * h)
    return grad

def find_zero(call_type, current_price, strike, ttm, rf, cost_of_carry,market_price):
    def objective(iv):
        root = (gbsm(call_type, current_price,strike, ttm, rf, cost_of_carry, iv) - market_price)**2
        return root
    result = minimize(objective, x0=0.2, bounds=[(0.01, 3.0)])
    return result.x[0] if result.success else np.nan

def find_zero_bt(call_type, S, K, T, rf, div, daysDiv, N, market_price):
    def objective(iv):
        price_model = (bt_american(call_type, S, K, T, rf, div, daysDiv, iv, N) - market_price)**2
        return price_model
    result = minimize(objective, x0=0.2, bounds=[(0.01, 3.0)])
    return result.x[0] if result.success else np.nan