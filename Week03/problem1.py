
from itertools import accumulate

from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('DailyReturn.csv')


def calculate_cov(x,y,alpha):
    x_u = x[0]
    y_u = y[0]
    cov = 0
    for t in range(len(x)):
        cov = cov*alpha + (1 - alpha)*(x[t] - x_u)*(y[t] - y_u)
        x_u = x_u * alpha + (1 - alpha) * x[t]
        y_u = y_u * alpha + (1 - alpha) * y[t]

    return cov

def build_matrix(data, alpha):
    n = len(data.columns)
    matrix = np.zeros((n,n))
    for i,x in enumerate(data.columns):
        for j,y in enumerate(data.columns):
            cov = calculate_cov(data[x],data[y],alpha)
            matrix[i,j] = cov
    matrix = pd.DataFrame(matrix,columns=data.columns,index=data.columns)
    return matrix



lambda_list = [0.5, 0.8, 0.94]
plt.figure(figsize=(10, 6))
for lam in lambda_list:
    cov_matrix = build_matrix(df,lam)
    eigenva,eigenve = np.linalg.eigh(cov_matrix)
    explained_variance = eigenva[::-1][:10]
    total_variance = np.sum(eigenva)
    explained_variance_ratio = explained_variance / total_variance
    cumulative_explained_variance = np.cumsum(explained_variance_ratio)
    plt.plot(cumulative_explained_variance,label=f'lambda = {lam}')
plt.title('Explained Variance')
plt.ylabel('percentage of variance explained by components')
plt.xlabel('10 PCA components')
plt.grid(True)
plt.legend()
plt.show()
