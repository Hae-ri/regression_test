# 회귀분석

import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

body_data = pd.read_csv('data/weight-height.csv')

regression_model = sm.OLS.from_formula('Weight~Height', body_data).fit()
print(regression_model.summary())

# y = -158.8027 + 1.3776x
# R-squared이 1에 가까울수록 상관관계?가 높음, 0에 가까우면 쓸 수 없는 데이터라고 보면 됨

height = body_data['Height'].tolist()
weight = body_data['Weight'].tolist()

body = pd.DataFrame({'몸무게':weight, '키':height})
plt.scatter(body['키'], body['몸무게'])

plt.xlabel('HEIGHT')
plt.ylabel('WEIGHT')

plt.show()