import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import t as tdist
from risk_management.missing import missing_cov
from risk_management.cov import ewCovar
from risk_management.psd import higham_PSD, near_PSD, cholesky_PSD, simulateNormal, cov_PCA_simulation
from risk_management.returns import calculate_return, calculate_log_return
from risk_management.regression import fit_regression_t, fit_normal, fit_general_t
from risk_management.var import ES, ESS, VaR
# from yourlib.ewcov import ewCovar
# from yourlib.psd_fix import near_psd, higham_nearestPSD
# from yourlib.simulation import simulateNormal, simulate_pca
# from yourlib.returns import return_calculate
# from yourlib.model_fit import fit_normal, fit_general_t, fit_regression_t
# from yourlib.risk_metrics import VaR, ES
# from yourlib.utils import corspearman, aggRisk


x = pd.read_csv("data/test1.csv")
# Test 1
# 1.1 Skip Missing rows - Covariance
cout = missing_cov(x.values, skipMiss=True) # TODO: implement missing_cov
reference_output = pd.read_csv("data/testout_1.1.csv").values
print(cout)
if np.allclose(cout, reference_output, equal_nan=True):
    print("1.1 Test passed!")
else:
    print("1.1 Test failed!")

# 1.2 Skip Missing rows - Correlation
cout = missing_cov(x.values, skipMiss=True, fun=np.corrcoef)
reference_output = pd.read_csv("data/testout_1.2.csv").values
print(cout)
if np.allclose(cout, reference_output, equal_nan=True):
    print("1.2 Test passed!")
else:
    print("1.2 Test failed!")

# 1.3 Pairwise - Covariance
cout = missing_cov(x.values, skipMiss=False)
reference_output = pd.read_csv("data/testout_1.3.csv").values
print(cout)
if np.allclose(cout, reference_output, equal_nan=True):
    print("1.3 Test passed!")
else:
    print("1.3 Test failed!")

# 1.4 Pairwise - Correlation
cout = missing_cov(x.values, skipMiss=False, fun=np.corrcoef)
reference_output = pd.read_csv("data/testout_1.4.csv").values
print(cout)
if np.allclose(cout, reference_output, equal_nan=True):
    print("1.4 Test passed!")
else:
    print("1.4 Test failed!")

# Test 2 - EW Covariance
x = pd.read_csv("data/test2.csv")
# 2.1 EW Covariance λ=0.97
cout = ewCovar(pd.DataFrame(x,columns=["x1","x2","x3","x4","x5"]),0.97)
reference_output = pd.read_csv("data/testout_2.1.csv")
if np.allclose(cout.round(0), reference_output.round(0), equal_nan=True):
    print("2.1 Test passed!")
else:
    print("2.1 Test failed!")

# 2.2 EW Correlation λ=0.94
ewma_cov = ewCovar(x,0.97)
ewma_std = np.sqrt(np.diag(ewma_cov))
cout = np.diag(1./ewma_std) @ ewma_cov @ np.diag(1./ewma_std)
reference_output = pd.read_csv("data/testout_2.1.csv")
if np.allclose(cout.round(0), reference_output.round(0), equal_nan=True):
    print("2.2 Test passed!")
else:
    print("2.2 Test failed!")


# Test 3 - non-psd
cin = pd.read_csv("data/testout_1.3.csv").values
cout = near_PSD(cin)
reference_output = pd.read_csv("data/testout_3.1.csv")
print(cout)
if np.allclose(cout.round(3), reference_output.round(3), equal_nan=True):
    print("3.1 Test passed!")
else:
    print("3.1 Test failed!")

cin = pd.read_csv("data/testout_1.4.csv").values
cout = near_PSD(cin)
reference_output = pd.read_csv("data/testout_3.2.csv")
print(cout)
if np.allclose(cout.round(3), reference_output.round(3), equal_nan=True):
    print("3.2 Test passed!")
else:
    print("3.2 Test failed!")

cin = pd.read_csv("data/testout_1.3.csv").values
cout = higham_PSD(cin)
reference_output = pd.read_csv("data/testout_3.3.csv")
print(cout)
if np.allclose(cout.round(0), reference_output.round(0), equal_nan=True):
    print("3.3 Test passed!")
else:
    print("3.3 Test failed!")

cin = pd.read_csv("data/testout_1.4.csv").values
cout = higham_PSD(cin)
reference_output = pd.read_csv("data/testout_3.4.csv")
print(cout)
if np.allclose(cout, reference_output, equal_nan=True):
    print("3.4 Test passed!")
else:
    print("3.4 Test failed!")

# 4 cholesky factorization
cin = pd.read_csv("data/testout_3.1.csv").values
cout = cholesky_PSD(cin)
reference_output = pd.read_csv("data/testout_4.1.csv")
print(cout)
if np.allclose(cout, reference_output, equal_nan=True):
    print("3.4 Test passed!")
else:
    print("3.4 Test failed!")

# 5 Normal Simulation
cin = pd.read_csv("data/test5_1.csv")
n = len(cin.columns)
mean = np.zeros(n)
samples = simulateNormal(100000, cin, mean)
cout = np.cov(samples, rowvar=False)
print(cout)
reference_output = pd.read_csv("data/testout_5.1.csv")
if np.allclose(cout.round(2), reference_output.round(2), equal_nan=True):
    print("5.1 Test passed!")
else:
    print("5.1 Test failed!")

# 5.2 PSD Input
cin = pd.read_csv("data/test5_2.csv")
n = len(cin.columns)
mean = np.zeros(n)
samples = simulateNormal(100000, cin, mean)
cout = np.cov(samples, rowvar=False)
print(cout)
reference_output = pd.read_csv("data/testout_5.2.csv")
if np.allclose(cout.round(2), reference_output.round(2), equal_nan=True):
    print("5.2 Test passed!")
else:
    print("5.2 Test failed!")

# 5.3 nonPSD Input, near_psd fix
cin = pd.read_csv("data/test5_3.csv").values
n = len(cin[0])
mean = np.zeros(n)
samples = simulateNormal(100000, cin, fixmethod=near_PSD)
cout = np.cov(samples, rowvar=False)
print(cout)
reference_output = pd.read_csv("data/testout_5.3.csv")
if np.allclose(cout.round(2), reference_output.round(2), equal_nan=True):
    print("5.3 Test passed!")
else:
    print("5.3 Test failed!")


# 5.4 nonPSD Input Higham Fix
cin = pd.read_csv("data/test5_3.csv").values
n = len(cin[0])
mean = np.zeros(n)
samples = simulateNormal(100000, cin, fixmethod=higham_PSD)
cout = np.cov(samples, rowvar=False)
print(cout)
reference_output = pd.read_csv("data/testout_5.4.csv")
if np.allclose(cout.round(1), reference_output.round(1), equal_nan=True):
    print("5.4 Test passed!")
else:
    print("5.4 Test failed!")

# 5.5 PSD Input - PCA Simulation

cin = pd.read_csv("data/test5_2.csv")
sample = np.random.multivariate_normal(np.zeros(len(cin.columns)), cin, 100000)
cout = cov_PCA_simulation(sample, percent=.99)
print(cout)
reference_output = pd.read_csv("data/testout_5.5.csv")
if np.allclose(cout.round(1), reference_output.round(1), equal_nan=True):
    print("5.5 Test passed!")
else:
    print("5.5 Test failed!")

# Test 6
prices = pd.read_csv("data/test6.csv")
cout = calculate_return(prices, prices.columns[1:])
reference_output = pd.read_csv("data/testout6_1.csv")
print(cout)

# Test 6.1
prices = pd.read_csv("data/test6.csv")
cout = calculate_log_return(prices, prices.columns[1:])
reference_output = pd.read_csv("data/testout6_2.csv")
print(cout)

# Test 7
# 7.1 Fit Normal Distribution
cin = pd.read_csv("data/test7_1.csv").values
fd = fit_normal(cin[:,0])
cout = pd.DataFrame({"mu":[fd["error_model"].mean()],"sigma":[fd["error_model"].std()]})
print(cout)
reference_output = pd.read_csv("data/testout7_1.csv")
if np.allclose(cout.round(3), reference_output.round(3), equal_nan=True):
    print("7.1 Test passed!")
else:
    print("7.1 Test failed!")

# 7.2
cin = pd.read_csv("data/test7_2.csv").values
fd = fit_general_t(cin[:,0])
cout = pd.DataFrame({"mu":[fd["m"]],"sigma":[fd["s"]], "nu":[fd["nu"]]})
print(cout)
reference_output = pd.read_csv("data/testout7_2.csv")
if np.allclose(cout.round(3), reference_output.round(3), equal_nan=True):
    print("7.2 Test passed!")
else:
    print("7.2 Test failed!")

# 7.3 Fit T Regression
cin = pd.read_csv("data/test7_3.csv")
fd = fit_regression_t(cin["y"].values, cin[["x1","x2","x3"]].values)
cout = pd.DataFrame({"mu":[fd["m"]],"sigma":[fd["s"]], "nu":[fd["nu"]], "Alpha":[fd["beta"][0]],  "B1":[fd["beta"][1]],  "B2":[fd["beta"][2]],  "B3":[fd["beta"][3]]})
print(cout)
reference_output = pd.read_csv("data/testout7_3.csv")
if np.allclose(cout.round(2), reference_output.round(2), equal_nan=True):
    print("7.3 Test passed!")
else:
    print("7.3 Test failed!")

# Test 8
#Test8.1
cin = pd.read_csv("data/test7_1.csv").values
fd = fit_normal(cin[:,0]) # TODO
print(fd["error_model"].ppf(0.05))
cout = pd.DataFrame({"VaR Absolute":[-fd["error_model"].ppf(0.05)],
              "VaR Diff from Mean":[-norm.ppf(0.05)*fd["error_model"].std()]})
print(cout)
reference_output = pd.read_csv("data/testout8_1.csv")
if np.allclose(cout.round(3), reference_output.round(3), equal_nan=True):
    print("8.1 Test passed!")
else:
    print("8.1 Test failed!")

#Test 8.2
cin = pd.read_csv("data/test7_2.csv").values
fd = fit_general_t(cin[:,0]) # TODO
cout = pd.DataFrame({"VaR Absolute":[-fd["error_model"].ppf(0.05)],
              "VaR Diff from Mean":[-fd["error_model"].ppf(0.05)+fd["m"]]})
print(cout)
reference_output = pd.read_csv("data/testout8_2.csv")
if np.allclose(cout.round(3), reference_output.round(3), equal_nan=True):
    print("8.2 Test passed!")
else:
    print("8.2 Test failed!")


# 8.3 VaR Simulation
cin = pd.read_csv("data/test7_2.csv").values
fd = fit_general_t(cin[:,0])
sim = fd["eval"](np.random.rand(10000))
cout = pd.DataFrame({"VaR Absolute":[VaR(sim)],
              "VaR Diff from Mean":[VaR(sim - sim.mean())]})
reference_output = pd.read_csv("data/testout8_3.csv")
print(cout)
if np.allclose(cout.round(1), reference_output.round(1), equal_nan=True):
    print("8.3 Test passed!")
else:
    print("8.3 Test failed!")


# 8.4 ES Normal
cin = pd.read_csv("data/test7_1.csv").values
fd = fit_normal(cin[:,0])
cout = pd.DataFrame({"ES Absolute":[ES(fd["error_model"])],
              "ES Diff from Mean":[ES(fd["error_model"])+cin[:,0].mean()]})
reference_output = pd.read_csv("data/testout8_4.csv")
print(cout)
if np.allclose(cout.round(2), reference_output.round(2), equal_nan=True):
    print("8.4 Test passed!")
else:
    print("8.4 Test failed!")

# 8.5 ES t dist
cin = pd.read_csv("data/test7_2.csv").values
fd = fit_general_t(cin[:,0])
cout = pd.DataFrame({"ES Absolute":[ES(fd["error_model"])],
              "ES Diff from Mean":[ES(tdist(df=fd["nu"], loc=0, scale=fd["s"]))]})
reference_output = pd.read_csv("data/testout8_5.csv")
print(cout)
if np.allclose(cout.round(2), reference_output.round(2), equal_nan=True):
    print("8.5 Test passed!")
else:
    print("8.5 Test failed!")


# 8.6 VaR Simulation
cin = pd.read_csv("data/test7_2.csv").values
fd = fit_general_t(cin[:,0])
sim = fd["eval"](np.random.rand(10000))
print(-np.percentile(sim, 5), ESS(sim))
cout = pd.DataFrame({"ES Absolute":[ESS(sim)],
              "ES Diff from Mean":[ESS(sim - sim.mean())]})
reference_output = pd.read_csv("data/testout8_6.csv")
print(cout)
if np.allclose(cout.round(1), reference_output.round(1), equal_nan=True):
    print("8.6 Test passed!")
else:
    print("8.6 Test failed!")
