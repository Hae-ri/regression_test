# 다중회귀분석
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

mpg_df = pd.read_csv('data/auto-mpg.csv')
# print(mpg_df)

mpg_df1 = mpg_df.drop(['horsepower','origin','car_name'], axis=1) # 필요없는 변수 제거
# print(mpg_df1.info())

x = mpg_df1.drop(['mpg'],axis=1) # mpg를 제외한 독립변수들
y = mpg_df1['mpg'] # 종속변수(연비)

# 과적합을 방지하기 위해 테스트를 진행 데이터를 보통 7:3 비율로 나눠서 7로 만들고 3으로 테스트 함
# 독립변수7,독립변수3,종속변수7,종속변수3 - 훈련용 데이터와 평가용 데이터 분할
X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size=0.3, random_state=0) # 0.7만큼 랜덤으로 빼서 트레이닝데이터 4개 추출

lr = LinearRegression() # 회귀모델생성
lr.fit(X_train, Y_train) # 모델 훈련
Y_predict = lr.predict((X_test)) # 평가데이터에 대한 예측 수행
mse = mean_squared_error(Y_test, Y_predict) # 평균제곱오차 - 잔차의 제곱한 값을 모두 더해 평균
rmse = np.sqrt(mse) # 루트를 씌운 것
r2_s = r2_score(Y_test,Y_predict)
print('mse',mse)
print('rmse', rmse) # 오차

print('결정계수:',r2_s) # R-squared : 5개의 독립변수가 종속변수와 약 81% 관계가 있다.

mpg_intercept = lr.intercept_
mpg_coef=lr.coef_

print('절편 : ', mpg_intercept)
print('회귀계수(기울기) :', mpg_coef)
# y = -0.13707609x + 0.00748253x1 -0.00688522x2 +  0.19807649x4 +  0.7577852x5
coef = pd.Series(data=mpg_coef, index=x.columns) # column별로 회귀계수 출력
print(coef)

import matplotlib.pyplot as plt
import seaborn as sns

# fig, axs = plt.subplots(figsize=(15,15), ncols=3, nrows=2)
# x_fea = ['cylinders','displacement','weight','acceleration','model_year']
#
# sns.regplot(x_fea[0], y='mpg', data=mpg_df1, ax=axs[0][0])
# sns.regplot(x_fea[1], y='mpg', data=mpg_df1, ax=axs[0][1])
# sns.regplot(x_fea[2], y='mpg', data=mpg_df1, ax=axs[0][2])
# sns.regplot(x_fea[3], y='mpg', data=mpg_df1, ax=axs[1][0])
# sns.regplot(x_fea[4], y='mpg', data=mpg_df1, ax=axs[1][1])
#
# plt.show()


# 연비를 예측
print('연비를 예측하고 싶은 자동차의 정보를 입력하세요.')
cylinders_ = int(input('실린더 수 : '))
displacement_ = int(input('배기량 : '))
weight_ = int(input('차의 무게 : '))
acceleration_ = int(input('가속력 : '))
model_year_ = int(input('차의 연식 : '))

mpg_predict = lr.predict([[cylinders_,displacement_,weight_,acceleration_,model_year_]])
print('입력하신 자동차의 연비(mpg)는 %f 입니다,' %mpg_predict)