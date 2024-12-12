import numpy as np
from scipy.stats import norm
from datetime import datetime
from risk_management.option import gbsm

import matplotlib.pyplot as plt
days_to_maturity = (datetime(2023, 3, 17) - datetime(2023, 3, 3)).days/365
ivs = np.linspace(0.1,0.8,10)
valueC = []
valueP = []
for iv in ivs:
    valueC.append(gbsm(True, 165, 165,days_to_maturity, 0.0525, 0.0053, iv))
    valueP.append(gbsm(False, 165, 165,days_to_maturity, 0.0525, 0.0053, iv))
plt.plot(ivs,valueC,label = "Call", color = "r")
plt.plot(ivs,valueP,label = "Put", color = "b")
plt.xlabel("IV")
plt.ylabel("Option price")
plt.legend(["Call","Put"])
plt.show()
