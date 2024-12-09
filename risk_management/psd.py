import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

def cholesky_PSD(data):
    data = pd.DataFrame(data)
    n = len(data)  # number of rows
    matrix = pd.DataFrame(0, index=range(n), columns=range(n))

    for x in range(n):
        for y in range(x+1): # only lower upper half iterate
            if x == y:
                matrix.iloc[x, y] = np.sqrt(data.iloc[x, y] - np.sum(np.square(matrix.iloc[x, :y])))
            else:
                matrix.iloc[x, y] = (data.iloc[x, y] - np.dot(matrix.iloc[x, :y], matrix.iloc[y, :y])) / matrix.iloc[y, y]
    matrix = matrix.values
    return matrix

def higham_PSD(A, tol=1e-8, max_iter=100):
    Y = A.copy()
    delta_S = np.zeros_like(A)
    for k in range(max_iter):
        R = Y - delta_S
        X = proj_symmetric(R)
        delta_S = X - R
        Y = proj_psd(X)
        if np.linalg.norm(Y - X, ord='fro') < tol:
            break
    return Y

def proj_symmetric(matrix):
    return (matrix + matrix.T) / 2

def proj_psd(matrix):
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    eigenvalues[eigenvalues < 0] = 0
    return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

def near_PSD(a, epsilon=0.0):
    n = a.shape[0]
    out = np.copy(a)
    invSD = None

    if not np.allclose(np.diag(out), np.ones(n)):
        invSD = np.diag(1.0 / np.sqrt(np.diag(out)))
        out = invSD @ out @ invSD

    vals, vecs = np.linalg.eigh(out)
    vals = np.maximum(vals, epsilon)
    T = 1.0 / np.sqrt(np.dot(vecs ** 2, vals))
    T = np.diag(T)
    l = np.diag(np.sqrt(vals))
    B = T @ vecs @ l
    out = B @ B.T

    if invSD is not None:
        invSD = np.diag(1.0 / np.diag(invSD))
        out = invSD @ out @ invSD

    return out

def simulateNormal(N, cov, mean=None, seed=1234, fixmethod=None):
    np.random.seed(seed)
    n = cov.shape[0]
    if cov.shape[0] != cov.shape[1]:
        raise ValueError(f"Covariance matrix is not square ({n},{cov.shape[1]})")
    if mean is None:
        mean = np.zeros(n)
    elif len(mean) != n:
        raise ValueError(f"Mean ({len(mean)}) is not the size of cov ({n},{n})")
    if fixmethod:
        cov = fixmethod(cov)
    L = np.linalg.cholesky(cov)
    randn = np.random.randn(n, N)
    samples = L @ randn + mean[:, np.newaxis]
    return samples.T

def find_n_component(cumulative_variance,percent):
    n = 0
    while percent > cumulative_variance[n]:
        n +=1
    return n

def cov_PCA_simulation(samples,percent=1.0):
    pca = PCA()
    pca.fit(samples)
    eigenvector = pca.components_.T
    eigenvalues = pca.explained_variance_
    if percent == 1.0:
        eigenvalue = np.diag(eigenvalues)
    else:
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        n = find_n_component(cumulative_variance, percent)
        eigenvector = eigenvector[:, :n]
        eigenvalue = np.diag(eigenvalues[:n])

    cov = eigenvector @ eigenvalue @ eigenvector.T
    return cov