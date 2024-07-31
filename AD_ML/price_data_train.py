from datetime import datetime
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pymysql
from skimage.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from clustering import camera_clustering as cc

# Database Properties
db = pymysql.connect(
    host='ad-database.cjo66a2aoqcq.ap-southeast-1.rds.amazonaws.com',
    port=3306,
    user='admin',
    passwd='lushuwen',
    db='CamC',
    charset='utf8'
)

cursor = db.cursor()

hp_c0_list = []
hp_c1_list = []
hp_c2_list = []
hp_c3_list = []

for i in range(1, 36):
    query = '''select * from price where platform = 'Amazon' and camera_id = %s'''
    cursor.execute(query, [i])
    results = cursor.fetchall()
    columns = cursor.description
    cols = [col[0] for col in columns]
    historical_price = pd.DataFrame(results, columns=cols)
    historical_price = historical_price.drop(['id', 'platform'], axis=1)

    # Days and Scaled Price
    query = '''select release_time, initial_price from camera where id = %s'''
    cursor.execute(query, [i])
    results = cursor.fetchone()
    release_time = results[0]
    initial_price = results[1]

    date_format = "%Y-%m-%d"
    release_time = pd.to_datetime(release_time, format=date_format)

    for j in range(historical_price.shape[0]):
        historical_date = pd.to_datetime(historical_price.at[j, 'date'], format=date_format)
        historical_price.at[j, 'date'] = (historical_date - release_time).days

    historical_price['price'] = historical_price['price'] / initial_price

    # 根据分类结果，将数据添加到相应的列表中
    cluster = cc(i)
    if cluster == 0:
        hp_c0_list.append(historical_price)
    elif cluster == 1:
        hp_c1_list.append(historical_price)
    elif cluster == 2:
        hp_c2_list.append(historical_price)
    elif cluster == 3:
        hp_c3_list.append(historical_price)

cursor.close()
db.close()

hp_c0 = pd.concat(hp_c0_list, ignore_index=True)
hp_c0 = hp_c0.sort_values(['date'], ascending=True)
hp_c0 = hp_c0.groupby('date', as_index=False).agg({'price': 'mean'})
hp_c0.to_csv('datas/historical_price/0.csv', index=True)

hp_c1 = pd.concat(hp_c1_list, ignore_index=True)
hp_c1 = hp_c1.sort_values(['date'], ascending=True)
hp_c1 = hp_c1.groupby('date', as_index=False).agg({'price': 'mean'})
hp_c1.to_csv('datas/historical_price/1.csv', index=True)

hp_c2 = pd.concat(hp_c2_list, ignore_index=True)
hp_c2 = hp_c2.sort_values(['date'], ascending=True)
hp_c2 = hp_c2.groupby('date', as_index=False).agg({'price': 'mean'})
hp_c2.to_csv('datas/historical_price/2.csv', index=True)

hp_c3 = pd.concat(hp_c3_list, ignore_index=True)
hp_c3 = hp_c3.sort_values(['date'], ascending=True)
hp_c3 = hp_c3.groupby('date', as_index=False).agg({'price': 'mean'})
hp_c3.to_csv('datas/historical_price/3.csv', index=True)

'''
Multiple Linear Regression

hp_c0['price'] = hp_c0['price'].rolling(window=30, center=True).mean()
hp_c1['price'] = hp_c1['price'].rolling(window=30, center=True).mean()
hp_c2['price'] = hp_c2['price'].rolling(window=30, center=True).mean()
hp_c3['price'] = hp_c3['price'].rolling(window=30, center=True).mean()

hp_c0 = hp_c0.iloc[15:-15]
hp_c1 = hp_c1.iloc[15:-15]
hp_c2 = hp_c2.iloc[15:-15]
hp_c3 = hp_c3.iloc[15:-15]

x_0 = hp_c0[['date']].values
x_1 = hp_c1[['date']].values
x_2 = hp_c2[['date']].values
x_3 = hp_c3[['date']].values

y_0 = hp_c0['price'].values
y_1 = hp_c1['price'].values
y_2 = hp_c2['price'].values
y_3 = hp_c3['price'].values

poly = PolynomialFeatures(degree=5)
x_poly_0 = poly.fit_transform(x_0)
x_poly_1 = poly.fit_transform(x_1)
x_poly_2 = poly.fit_transform(x_2)
x_poly_3 = poly.fit_transform(x_3)

model_0 = LinearRegression()
model_1 = LinearRegression()
model_2 = LinearRegression()
model_3 = LinearRegression()

model_0.fit(x_poly_0, y_0)
model_1.fit(x_poly_1, y_1)
model_2.fit(x_poly_2, y_2)
model_3.fit(x_poly_3, y_3)


X_fit_0 = np.linspace(x_0.min(), x_0.max(), 500).reshape(-1, 1)
X_fit_1 = np.linspace(x_1.min(), x_1.max(), 500).reshape(-1, 1)
X_fit_2 = np.linspace(x_2.min(), x_2.max(), 500).reshape(-1, 1)
X_fit_3 = np.linspace(x_3.min(), x_3.max(), 500).reshape(-1, 1)

X_fit_poly_0 = poly.transform(X_fit_0)
X_fit_poly_1 = poly.transform(X_fit_1)
X_fit_poly_2 = poly.transform(X_fit_2)
X_fit_poly_3 = poly.transform(X_fit_3)

y_fit_0 = model_0.predict(X_fit_poly_0)
y_fit_1 = model_1.predict(X_fit_poly_1)
y_fit_2 = model_2.predict(X_fit_poly_2)
y_fit_3 = model_3.predict(X_fit_poly_3)

# 计算模型评估指标
y_pred_0 = model_0.predict(x_poly_0)
y_pred_1 = model_1.predict(x_poly_1)
y_pred_2 = model_2.predict(x_poly_2)
y_pred_3 = model_3.predict(x_poly_3)

mse_0 = mean_squared_error(y_0, y_pred_0)
mse_1 = mean_squared_error(y_1, y_pred_1)
mse_2 = mean_squared_error(y_2, y_pred_2)
mse_3 = mean_squared_error(y_3, y_pred_3)

r2_0 = r2_score(y_0, y_pred_0)
r2_1 = r2_score(y_1, y_pred_1)
r2_2 = r2_score(y_2, y_pred_2)
r2_3 = r2_score(y_3, y_pred_3)

# 输出评估指标
print(f"MSE_0: {mse_0:.8f}      MSE_1: {mse_1:.8f}")
print(f"R²_0: {r2_0:.8f}        R²_1: {r2_1:.8f}")

print(f"MSE_2: {mse_2:.8f}      MSE_3: {mse_3:.8f}")
print(f"R²_2: {r2_2:.8f}        R²_3: {r2_3:.8f}")

fig, axs = plt.subplots(2, 2, figsize=(10, 10))

ax = axs[0, 0]
ax.scatter(x_0, y_0, color='blue', label='Initial Data')
ax.plot(X_fit_0, y_fit_0, color='red', label=f'5 Degree Multiple Linear Regression')
ax.set_xlabel('Date')
ax.set_ylabel('Price')
ax.set_title('Group_0')

ax = axs[0, 1]
ax.scatter(x_1, y_1, color='blue', label='Initial Data')
ax.plot(X_fit_1, y_fit_1, color='red', label=f'5 Degree Multiple Linear Regression')
ax.set_xlabel('Date')
ax.set_ylabel('Price')
ax.set_title('Group_1')

ax = axs[1, 0]
ax.scatter(x_2, y_2, color='blue', label='Initial Data')
ax.plot(X_fit_2, y_fit_2, color='red', label=f'5 Degree Multiple Linear Regression')
ax.set_xlabel('Date')
ax.set_ylabel('Price')
ax.set_title('Group_2')

ax = axs[1, 1]
ax.scatter(x_3, y_3, color='blue', label='Initial Data')
ax.plot(X_fit_3, y_fit_3, color='red', label=f'5 Degree Multiple Linear Regression')
ax.set_xlabel('Date')
ax.set_ylabel('Price')
ax.set_title('Group_3')

plt.show()
'''
