import pandas as pd
import numpy as np
data = pd.read_csv('DailyPrices.csv')
variance = 0.01
mean = 0
n_simulation = 2000

for x in range (5):
    r_list = np.random.normal(mean, np.sqrt(variance), n_simulation)
    P_t_1 = data.iloc[x,:]['SPY']
    P_t_simple = P_t_1 + r_list
    P_t_arith = P_t_1 * (1+r_list)
    P_t_log = P_t_1*np.exp(r_list)
    std_tuple = (float(P_t_simple.std()), float(P_t_arith.std()), float(P_t_log.std()))
    expect_tuple = (float(P_t_simple.mean()), float(P_t_arith.mean()), float(P_t_log.mean()))
    print(f'for day{x+1} SPY price, expected price is {expect_tuple}, std is {std_tuple}')



