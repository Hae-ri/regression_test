import pandas as pd
import statsmodels.api as sm

height = [170,168,177,181,172,171,169,175,174,178,170,167,177,182,173,171,170,179,175,177,186,166,183,168]
weight = [70,66,73,77,74,73,69,79,77,80,74,68,71,76,78,72,68,79,77,81,84,73,78,69]
# height -> 독립변수, weight-> 종속변수

body = pd.DataFrame({'키':height, '몸무게':weight})

regression = sm.OLS.from_formula('weight~height', body).fit() # 종속변수, 독립변수 순서로

print(regression.summary())

# print(body)