import numpy as np
import pandas as pd
from scipy.linalg import cholesky
import time


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
    n = A.shape[0]
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

n = 500
sigma = np.full((n, n), 0.9)
np.fill_diagonal(sigma, 1)
sigma[0, 1] = 0.7357
sigma[1, 0] = 0.7357


start_time = time.time()
matrix_near = near_PSD(sigma)
end_time = time.time()
runtime = end_time - start_time
eigenvalues = np.linalg.eigvals(matrix_near)
print(f'the matrix fixed by near_PSD is PSD matrix now: {np.all(eigenvalues >= 0)}')
print(f'the the runtime for near_PSD is: {runtime:.6f}')


start_time = time.time()
matrix_hig = higham_PSD(sigma)
end_time = time.time()
runtime = end_time - start_time
eigenvalues = np.linalg.eigvals(matrix_hig)
print(f'the matrix fixed by higham_PSD is PSD matrix now: {np.all(eigenvalues >= 0)}')
print(f'the the runtime for higham_PSD is: {runtime:.6f}')


distance = np.linalg.norm(matrix_hig-sigma,ord='fro')
distance1 = np.linalg.norm(matrix_near-sigma,ord='fro')
print(f'Fro norm for higham_psd is {distance:.6f},while Fro norm for near_psd is {distance1:.6f}')




