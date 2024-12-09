import numpy as np

def expW(m, λ):
    w = np.zeros(m)
    for i in range(m):
        w[i] = (1 - λ) * (λ ** (m - i - 1))
    w = w / w.sum()
    return w

def ewCovar(x, λ):
    m, n = x.shape
    w = expW(m, λ)
    weighted_mean = w @ x
    centered = x - weighted_mean
    sqrt_w = np.sqrt(w).reshape(-1, 1)
    xm = sqrt_w * centered
    cov = xm.T @ xm
    return cov