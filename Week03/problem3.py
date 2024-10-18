import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import time



def EWMA_var(data,alpha):
    var_list = []
    for column in data.columns:
        var = 0
        u = data[column].iloc[0]
        for x in data[column]:
            var = var * alpha + (1-alpha) * (x-u)**2
            u = u * alpha + (1 - alpha) * x
        var = float(var)
        var_list.append(var)
    return var_list

def EWMA_corr(x,y,alpha):
    if len(x) != len(y):
        raise ValueError("x and y not equal length")
    cov = 0
    var_x = np.var(x)
    var_y = np.var(y)
    u_x = x.iloc[0]
    u_y = y.iloc[0]
    corr = 0
    for i in range(len(x)):
        var_x = var_x * alpha + (1-alpha) * (x[i]-u_x)**2
        var_y = var_y * alpha + (1-alpha) * (y[i]-u_y)**2
        cov = cov * alpha + (1-alpha) * (x[i]-u_x)*(y[i]-u_y)
        u_x = u_x * alpha + (1 - alpha) * x[i]
        u_y = u_y * alpha + (1 - alpha) * y[i]
    corr = cov /np.sqrt(var_x*var_y)
    return corr

def build_corr_matrix(data, alpha):
    n = len(data.columns)
    matrix = np.zeros((n,n))
    for i,x in enumerate(data.columns):
        for j,y in enumerate(data.columns):
            cov = EWMA_corr(data[x],data[y],alpha)
            matrix[i,j] = cov
    matrix = pd.DataFrame(matrix,columns=data.columns,index=data.columns)
    return matrix

def print_fnorm(cov_new,cov,simulation):
    fnorm = np.linalg.norm(cov_new - cov.values, ord='fro')
    print(f'fro norm for four cov matrix based on {simulation} is {fnorm}')

data = pd.read_csv('DailyReturn.csv')

#generate four cov matrix
corr = data.corr()
var = data.var()
ewma_var = EWMA_var(data,0.94)
ewma_corr = build_corr_matrix(data,0.94)
cov1 = corr * np.outer(var,var)
cov2 = corr * np.outer(ewma_var,ewma_var)
cov3 = ewma_corr * np.outer(var,var)
cov4 = ewma_corr * np.outer(ewma_var,ewma_var)

#generate simulate cov matrix
n = len(data.columns)
#direct simulation
mean = np.zeros(n)
samples_1 = np.random.multivariate_normal(mean, cov1, 25000)
samples_2 = np.random.multivariate_normal(mean, cov2, 25000)
samples_3 = np.random.multivariate_normal(mean, cov3, 25000)
samples_4 = np.random.multivariate_normal(mean, cov4, 25000)
cov_1_new = np.cov(samples_1,rowvar=False)
cov_2_new = np.cov(samples_2,rowvar=False)
cov_3_new = np.cov(samples_3,rowvar=False)
cov_4_new = np.cov(samples_4,rowvar=False)
print_fnorm(cov_1_new,cov1,'direct simulation')
print_fnorm(cov_2_new,cov2,'direct simulation')
print_fnorm(cov_3_new,cov3,'direct simulation')
print_fnorm(cov_4_new,cov4,'direct simulation')


#PCA
def find_n_component(cumulative_variance,percent):
    n = 0
    while percent > cumulative_variance[n]:
        n +=1
    return n
def cov_PCA_simulation(samples,percent=1.0):
    start_time = time.time()
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
    end_time = time.time()
    runtime = end_time - start_time
    print(f' the runtime is: {runtime:.6f}')
    return cov

#PCA 100%
cov_1_new = cov_PCA_simulation(samples_1)
cov_2_new = cov_PCA_simulation(samples_2)
cov_3_new = cov_PCA_simulation(samples_3)
cov_4_new = cov_PCA_simulation(samples_4)
print_fnorm(cov_1_new,cov1,'PCA100')
print_fnorm(cov_2_new,cov2,'PCA100')
print_fnorm(cov_3_new,cov3,'PCA100')
print_fnorm(cov_4_new,cov4,'PCA100')


# PCA 75%
cov_1_new = cov_PCA_simulation(samples_1,0.75)
cov_2_new = cov_PCA_simulation(samples_2,0.75)
cov_3_new = cov_PCA_simulation(samples_3,0.75)
cov_4_new = cov_PCA_simulation(samples_4,0.75)
print_fnorm(cov_1_new,cov1,'PCA75')
print_fnorm(cov_2_new,cov2,'PCA75')
print_fnorm(cov_3_new,cov3,'PCA75')
print_fnorm(cov_4_new,cov4,'PCA75')

# PCA 50%
cov_1_new = cov_PCA_simulation(samples_1,0.5)
cov_2_new = cov_PCA_simulation(samples_2,0.5)
cov_3_new = cov_PCA_simulation(samples_3,0.5)
cov_4_new = cov_PCA_simulation(samples_4,0.5)
print_fnorm(cov_1_new,cov1,'PCA50')
print_fnorm(cov_2_new,cov2,'PCA50')
print_fnorm(cov_3_new,cov3,'PCA50')
print_fnorm(cov_4_new,cov4,'PCA50')



