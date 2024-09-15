import pandas as pd
import numpy as np

data = pd.read_csv('problem1.csv')

def unbiased_skew(data):
    n = len(data)
    mean_x = data.mean()
    std_x = data.std(ddof=0) 
    numerator = np.sum((data - mean_x)**3)
    correction_factor = n / ((n - 1) * (n - 2))
    unbiased_skew = correction_factor * numerator
    normalized_skew = unbiased_skew / (std_x**3)
    return normalized_skew

def unbiased_kurt(data):
    n = len(data)

    # 样本均值
    mean_x = data.mean()
    
    # 样本总体方差 (ddof=0) 和四次中心矩
    variance_x = data.var(ddof=0)
    fourth_moment = np.sum((data - mean_x) ** 4) / n

    # 有偏峰度估计 K_x(x)
    biased_kurtosis = fourth_moment / (variance_x ** 2)

    # 有偏方差的四次方 σ_x^4(x)
    biased_variance_fourth = variance_x ** 2

    # 修正项，分母和括号部分
    numerator = n ** 2
    denominator = (n - 1) ** 3 * (n ** 2 - 3 * n + 3)
    
    # 计算无偏峰度
    unbiased_kurtosis = (numerator / denominator) * (
        (n * (n - 1) ** 2 + (6 * n - 9)) * biased_kurtosis - n * (6 * n - 9) * biased_variance_fourth
    )

    return unbiased_kurtosis

mean = data.mean()
var = data.var()
skew = data.skew()
kur = data.kurt()
print(mean)
print(var)
print(skew)
unbiased_skew = unbiased_skew(data)
print(unbiased_skew)
print(kur)
unbiased_kurt = unbiased_kurt(data)
print(unbiased_kurt)