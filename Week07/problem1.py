
import numpy as np
from datetime import datetime
import pandas as pd

from option import gbsm, bt_american, finite_diff_gradient

s = 151.03
strike = 165
ttm = (datetime(2022,4,15)-datetime(2022,3,13)).days/365
rf = 0.0425
q = 0.0053

call_dict = gbsm(True, s, strike, ttm, rf, rf-q, 0.2, Greek=True)
put_dict = gbsm(False, s, strike, ttm, rf, rf-q, 0.2, Greek=True)

gbsmTable = pd.DataFrame({
    "Type":["Call","Put"],
    "Delta":[call_dict["delta"], put_dict["delta"]],
    "Gamma":[call_dict["gamma"], put_dict["gamma"]],
    "Vega":[call_dict["vega"], put_dict["vega"]],
    "Theta":[call_dict["theta"], put_dict["theta"]],
    "cRho":[call_dict["cRho"], put_dict["cRho"]],
})

print(gbsmTable)



def function_call(xx):
    return gbsm(True, xx[0], xx[1], xx[2], xx[3], xx[4], xx[5], Greek=True)['value']
def function_put(xx):
    return gbsm(False, xx[0], xx[1], xx[2], xx[3], xx[4], xx[5], Greek=True)['value']

X = np.array([s, strike, ttm, rf, rf - q, 0.2])
call_grad = finite_diff_gradient(function_call, X)
put_grad = finite_diff_gradient(function_put, X)

def function_call_delta(xx):
    return gbsm(True, xx[0], xx[1], xx[2], xx[3], xx[4], xx[5])['delta']
call_gamma = finite_diff_gradient(function_call_delta, X)

def function_put_delta(xx):
    return gbsm(False, xx[0], xx[1], xx[2], xx[3], xx[4], xx[5])['delta']
put_gamma = finite_diff_gradient(function_put_delta, X)

numericTable = pd.DataFrame({
    "Type":["Call","Put"],
    "Delta":[call_grad[0], put_grad[0]],
    "Gamma":[call_gamma[0], put_gamma[0]],
    "Vega":[call_grad[5], put_grad[5]],
    "Theta":[-call_grad[2], -put_grad[2]],
    "Rho":[call_grad[3], put_grad[3]],
    "CarryRho":[call_grad[4], put_grad[4]]
})

print(numericTable)


div = 0.88
days_til_expire = (datetime(2022,4,15)-datetime(2022,3,13)).days
divDays = (datetime(2022, 4, 11) - datetime(2022,3,13)).days
NPoints = days_til_expire * 3
divPoint = divDays * 3

am_call = bt_american(True, s, strike, ttm, rf, [div], [divPoint], 0.2, NPoints)
am_put = bt_american(False, s, strike, ttm, rf, [div], [divPoint], 0.2, NPoints)


def bt_function_call(xx):
    return bt_american(True, xx[0], xx[1], xx[2], xx[3], [div], [divPoint], xx[4], NPoints)
def bt_function_put(xx):
    return bt_american(False, xx[0], xx[1], xx[2], xx[3], [div], [divPoint], xx[4], NPoints)


Y = np.array([s, strike, ttm, rf, 0.2])
call_grad_bt = finite_diff_gradient(bt_function_call, Y)
put_grad_bt = finite_diff_gradient(bt_function_put, Y)

h = 1
am_call_up = bt_american(True, s + h, strike, ttm, rf, [div], [divPoint], 0.2, NPoints)
am_call_dn = bt_american(True, s - h, strike, ttm, rf, [div], [divPoint], 0.2, NPoints)
call_gamma_bt = (am_call_up + am_call_dn - 2 * am_call) / (h ** 2)
am_put_up = bt_american(False, s + h, strike, ttm, rf, [div], [divPoint], 0.2, NPoints)
am_put_dn = bt_american(False, s - h, strike, ttm, rf, [div], [divPoint], 0.2, NPoints)
put_gamma_bt = (am_put_up + am_put_dn - 2 * am_put) / (h ** 2)


h_div = 1
am_call_div_up = bt_american(True, s, strike, ttm, rf, [div + h_div], [divPoint], 0.2, NPoints)
call_div_sens = (am_call_div_up - am_call) / (h_div)
am_put_div_up = bt_american(False, s, strike, ttm, rf, [div + h_div], [divPoint], 0.2, NPoints)
put_div_sens = (am_put_div_up - am_put) / (h_div)

print(f"call bt value: {am_call}")
print(f"Put bt value: {am_put}")

btTable = pd.DataFrame({
    "Type":["Call","Put"],
    "Delta":[call_grad_bt[0], put_grad_bt[0]],
    "Gamma":[call_gamma_bt, put_gamma_bt],
    "Vega":[call_grad_bt[4], put_grad_bt[4]],
    "Theta":[-call_grad_bt[2], -put_grad_bt[2]],
    "Rho":[call_grad_bt[3], put_grad_bt[3]],
    "CarryRho":[call_div_sens,put_div_sens]
})

print(btTable)
