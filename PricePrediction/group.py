import pandas as pd
import pymysql

import category_predict
from statsmodels.tsa.api import VAR

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
RED = '\033[91m'
RESET = '\033[0m'

'''
Retrieve historical price
'''
df_cluster = pd.read_csv('clusters.csv')
df_cluster.drop(columns=['Unnamed: 0'], inplace=True)
print(df_cluster)

db = pymysql.connect(
        host='ad-database.cjo66a2aoqcq.ap-southeast-1.rds.amazonaws.com', port=3306,
        user='admin', passwd='lushuwen',
        db='CamC', charset='utf8')
cursor = db.cursor()

sql = '''select * from price where platform = 'Amazon' '''
cursor.execute(sql)
prices = cursor.fetchall()
columns = cursor.description
cols = [col[0] for col in columns]
df_prices = pd.DataFrame(prices, columns=cols)
df_prices.drop(columns=['id'], inplace=True)
df_prices.drop(columns=['platform'], inplace=True)

cursor.close()
db.close()

df_cluster.index.name = 'camera_id'
df_prices['date'] = pd.to_datetime(df_prices.date)
df_cluster['release_time'] = pd.to_datetime(df_cluster.release_time)
df_prices['price'] = df_prices.apply(lambda row: row['price'] / df_cluster.loc[row['camera_id'] - 1, 'initial_price'], axis=1)
df_prices['date'] = df_prices.apply(lambda row: (row['date'] - df_cluster.loc[row['camera_id'] - 1, 'release_time']).days, axis=1)

df_prices.to_csv('price/prices.csv', index=False)

group1_price = pd.DataFrame(columns=['date'])
group2_price = pd.DataFrame(columns=['date'])
group3_price = pd.DataFrame(columns=['date'])
group4_price = pd.DataFrame(columns=['date'])

for camera_id, row in df_cluster.iterrows():
        camera_id += 1
        if row['cluster'] == 0:
                price_now = df_prices[df_prices['camera_id'] == camera_id].copy()
                if price_now.empty:
                        continue
                price_now.drop(columns=['camera_id'], inplace=True)
                price_now = price_now.rename(columns={'price': camera_id})
                group1_price = pd.merge(group1_price, price_now, on='date', how='outer')
        elif row['cluster'] == 1:
                price_now = df_prices[df_prices['camera_id'] == camera_id].copy()
                if price_now.empty:
                        continue
                price_now.drop(columns=['camera_id'], inplace=True)
                price_now = price_now.rename(columns={'price': camera_id})
                group2_price = pd.merge(group2_price, price_now, on='date', how='outer')
        elif row['cluster'] == 2:
                price_now = df_prices[df_prices['camera_id'] == camera_id].copy()
                if price_now.empty:
                        continue
                price_now.drop(columns=['camera_id'], inplace=True)
                price_now = price_now.rename(columns={'price': camera_id})
                group3_price = pd.merge(group3_price, price_now, on='date', how='outer')
        elif row['cluster'] == 3:
                price_now = df_prices[df_prices['camera_id'] == camera_id].copy()
                if price_now.empty:
                        continue
                price_now.drop(columns=['camera_id'], inplace=True)
                price_now = price_now.rename(columns={'price': camera_id})
                group4_price = pd.merge(group4_price, price_now, on='date', how='outer')

group1_price.set_index('date', inplace=True)
group2_price.set_index('date', inplace=True)
group3_price.set_index('date', inplace=True)
group4_price.set_index('date', inplace=True)
group1_price = group1_price.sort_index()
group2_price = group2_price.sort_index()
group3_price = group3_price.sort_index()
group4_price = group4_price.sort_index()

group1_price = group1_price.interpolate(method='linear')
group2_price = group2_price.interpolate(method='linear')
group3_price = group3_price.interpolate(method='linear')
group4_price = group4_price.interpolate(method='linear')

group1_price = group1_price.ffill().bfill()
group2_price = group2_price.ffill().bfill()
group3_price = group3_price.ffill().bfill()
group4_price = group4_price.ffill().bfill()

group1_price.to_csv('price/group1_price.csv', index=True)
group2_price.to_csv('price/group2_price.csv', index=True)
group3_price.to_csv('price/group3_price.csv', index=True)
group4_price.to_csv('price/group4_price.csv', index=True)

#删除常数列
group1_price_clean = group1_price.drop(columns=[11])

var_model = VAR(group1_price_clean)

results = var_model.fit(maxlags=4)
print(results.summary())
