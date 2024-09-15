# Import necessary libraries for ARMA model fitting
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_csv('problem3.csv')


# Fit AR(1) to AR(3) and MA(1) to MA(3) models and calculate AIC for comparison
aic_values = {}

# Fit AR(1) to AR(3)
for p in range(1, 4):
    model = ARIMA(data['x'], order=(p, 0, 0))  # AR(p)
    model_fit = model.fit()
    aic_values[f'AR({p})'] = model_fit.aic

# Fit MA(1) to MA(3)
for q in range(1, 4):
    model = ARIMA(data['x'], order=(0, 0, q))  # MA(q)
    model_fit = model.fit()
    aic_values[f'MA({q})'] = model_fit.aic


print(aic_values)

def calculate_aicc(aic, k, n):
    # Applying the correction for AICc
    aicc = aic + (2 * k**2 + 2 * k) / (n - k - 1)
    return aicc

# Number of data points (n) and number of parameters (k) for each model
n = len(data)  # Total number of observations

# Calculate AICc for each AR(1) to AR(3) and MA(1) to MA(3)
aicc_values = {}
for model, aic in aic_values.items():
    # Determine the number of parameters (k) for each model
    if "AR" in model:
        p = int(model[3])  # Extract the order of AR
        k = p + 1  # p coefficients + 1 constant
    elif "MA" in model:
        q = int(model[3])  # Extract the order of MA
        k = q + 1  # q coefficients + 1 constant
    
    # Calculate AICc
    aicc_values[model] = calculate_aicc(aic, k, n)

print(aicc_values)