import numpy as np
import pandas as pd
def calculate_return(data, company_list):
    results = {}
    for company in company_list:
        price_list = data[company]
        return_list = price_list.pct_change().dropna()
        results[company] =  list(return_list)
    result = pd.DataFrame(results, index=data.index[1:])
    if data.columns[0] in data:
        result.insert(0, data.columns[0], data.iloc[1:, 0].values)
    return result


def calculate_log_return(data, company_list):
    results = {}
    for company in company_list:
        price_list = data[company]
        log_return_list = np.log(price_list / price_list.shift(1)).dropna()
        results[company] = list(log_return_list)
    result = pd.DataFrame(results, index=data.index[1:])
    if data.columns[0] in data:
        result.insert(0, data.columns[0], data.iloc[1:, 0].values)

    return result

def return_calculate(prices: pd.DataFrame, method="DISCRETE", dateColumn="date"):
    if dateColumn not in prices.columns:
        raise ValueError(f"dateColumn: {dateColumn} not in DataFrame columns: {prices.columns.tolist()}")
    vars = [c for c in prices.columns if c != dateColumn]
    if len(vars) == 0:
        raise ValueError("No price columns found.")
    p = prices[vars].values
    n, m = p.shape
    p2 = p[1:] / p[:-1]
    method_upper = method.upper()
    if method_upper == "DISCRETE":
        p2 = p2 - 1.0
    elif method_upper == "LOG":
        p2 = np.log(p2)
    else:
        raise ValueError(f"method: {method} must be in (\"LOG\",\"DISCRETE\")")
    dates = prices[dateColumn].values[1:]
    result = pd.DataFrame(p2, columns=vars)
    result[dateColumn] = dates
    result = result[[dateColumn] + vars]
    return result
