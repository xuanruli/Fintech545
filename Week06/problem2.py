
from risk_management.option import gbsm, find_zero
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime


df = pd.read_csv("AAPL_Options.csv")
expirations = (pd.to_datetime(df["Expiration"]) - datetime(2023,10,30)).dt.days/365
type_list = df["Type"]
market_prices = df["Last Price"]
strikes = df["Strike"]
ivC = []
ivP = []
for i in range(len(df)):
    call_type = type_list[i]
    ttm = expirations[i]
    market_price = market_prices[i]
    strike = strikes[i]
    if call_type == "Call":
        ivC.append(find_zero(True,170.15, strike, ttm, 0.0525,0.0525-0.0057, market_price))
    else:
        ivP.append(find_zero(False, 170.15, strike, ttm, 0.0525, 0.0525-0.0057, market_price))
print("IV for those call in csv: ", ivC)
print("IV for those put in csv: ", ivP)


strikeC = df["Strike"][df["Type"] == "Call"]
strikeP = df["Strike"][df["Type"] == "Put"]
plt.plot(strikeC, ivC, label = "Call", color = "r")
plt.plot(strikeP, ivP, label = "Put", color = "b")
plt.xlabel("Strike Price")
plt.ylabel("IV")
plt.legend()
plt.show()