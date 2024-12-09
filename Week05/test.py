import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import t as tdist
from scipy.optimize import minimize

# from yourlib.missing_cov import missing_cov
# from yourlib.ewcov import ewCovar
# from yourlib.psd_fix import near_psd, higham_nearestPSD
# from yourlib.simulation import simulateNormal, simulate_pca
# from yourlib.returns import return_calculate
# from yourlib.model_fit import fit_normal, fit_general_t, fit_regression_t
# from yourlib.risk_metrics import VaR, ES
# from yourlib.utils import corspearman, aggRisk
def aggRisk(values, group_cols):
    alpha = 0.05
    results = []
    groups = values.groupby(group_cols)
    for gname, gdf in groups:
        portfolio_pnl = gdf['pnl'].values
        Var95 = VaR(portfolio_pnl, alpha=alpha)
        ES95 = ESS(portfolio_pnl, alpha=alpha)
        if isinstance(gname, tuple):
            gname = gname[0]
        results.append((gname, Var95, ES95))
    out = pd.DataFrame(results, columns=group_cols+['VaR95','ES95'])
    return out

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


