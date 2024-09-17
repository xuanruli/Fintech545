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

