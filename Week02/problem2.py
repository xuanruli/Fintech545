# Using the standard approach in Python for simple linear regression with scikit-learn

from sklearn.linear_model import LinearRegression
import numpy as np
from scipy.optimize import minimize
import pandas as pd
from scipy.stats import t
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

data = pd.read_csv('problem2.csv')

#OLS 
model = LinearRegression()
model.fit(data[['x']], data['y'])

beta0_sklearn = model.intercept_
beta1_sklearn = model.coef_[0]
y_pred = model.predict(data[['x']])
residuals = data['y'] - y_pred
sigma_sklearn = np.std(residuals, ddof=0)  #get residual, which is error, std

print(beta0_sklearn, beta1_sklearn, sigma_sklearn)


#MLE normal distribution
def negative_log_likelihood_mle(params, x, y):
    beta0, beta1, sigma = params
    y_pred = beta0 + beta1 * x
    n = len(y)
    log_likelihood = -0.5 * n * np.log(2 * np.pi * sigma**2) - np.sum((y - y_pred)**2) / (2 * sigma**2)
    return -log_likelihood 

initial_params = [0, 0, 1]


result_mle = minimize(negative_log_likelihood_mle, initial_params, args=(data['x'], data['y']), method='L-BFGS-B', bounds=[(None, None), (None, None), (1e-5, None)])

beta0_mle, beta1_mle, sigma_mle = result_mle.x

print(beta0_mle, beta1_mle, sigma_mle)



# MLE t distribution
def negative_log_likelihood_t_dist(params, x, y, nu):
    beta0, beta1, sigma = params
    y_pred = beta0 + beta1 * x
    residuals = (y - y_pred) / sigma
    n = len(y)
    log_likelihood = np.sum(t.logpdf(residuals, df=nu)) - n * np.log(sigma)
    return -log_likelihood  

initial_params = [0, 0, 1]
nu = 5  
result_t_dist = minimize(negative_log_likelihood_t_dist, initial_params, args=(data['x'], data['y'], nu), method='L-BFGS-B', bounds=[(None, None), (None, None), (1e-5, None)])
beta0_t_mle, beta1_t_mle, sigma_t_mle = result_t_dist.x
print(beta0_t_mle, beta1_t_mle, sigma_t_mle)

def calculate_R2(b0,b1,data):
    y_pred_mle = b0 + b1 * data['x']
    residuals = data['y'] - y_pred_mle
    rss = np.sum(residuals ** 2)
    y_mean = data['y'].mean()
    tss = np.sum((data['y'] - y_mean) ** 2)
    r_squared = 1 - (rss / tss)
    return r_squared

r_squared_t_mle = calculate_R2(beta0_t_mle,beta1_t_mle,data)
r_squared_mle = calculate_R2(beta0_mle,beta1_mle,data)
print(r_squared_mle,r_squared_t_mle)

data_x = pd.read_csv('problem2_x.csv')
mean_vector = data_x.mean().values  
cov_matrix = np.cov(data_x.T)  

sigma_11 = cov_matrix[0, 0]  # Var(X1)
sigma_22 = cov_matrix[1, 1]  # Var(X2)
sigma_12 = cov_matrix[0, 1]  # Cov(X1, X2)
x1_values = data_x['x1'].values
x2_values = data_x['x2'].values

conditional_mean_x2_given_x1 = mean_vector[1] + sigma_12 / sigma_11 * (x1_values - mean_vector[0])
conditional_var_x2_given_x1 = sigma_22 - sigma_12**2 / sigma_11


z = 1.96  # for 95% confidence
lower_bound = conditional_mean_x2_given_x1 - z * np.sqrt(conditional_var_x2_given_x1)
upper_bound = conditional_mean_x2_given_x1 + z * np.sqrt(conditional_var_x2_given_x1)


plt.figure(figsize=(10, 6))
plt.plot(x1_values, x2_values, 'o', label='X2')
plt.plot(x1_values, conditional_mean_x2_given_x1, 'r-', label='Expected X2 given X1')
plt.fill_between(x1_values, lower_bound, upper_bound, alpha=0.3, label='95% CI')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('problem2_x')
plt.legend()
plt.show()