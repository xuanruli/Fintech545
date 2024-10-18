from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

# 加载数据集
data = load_iris()
X, y = data.data, data.target

# 创建 K 折交叉验证器，5 折
k_fold = KFold(n_splits=5, shuffle=True, random_state=42)

# 创建逻辑回归模型
model = LogisticRegression()

# 进行交叉验证，得到每次验证集上的得分
scores = cross_val_score(model, X, y, cv=shuffle_split)

