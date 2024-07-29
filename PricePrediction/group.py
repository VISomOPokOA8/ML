import pandas as pd
import pymysql

import category_predict

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
df_prices['price_ratio'] = df_prices.apply(lambda row: row['price'] / df_cluster.loc[row['camera_id'] - 1, 'initial_price'], axis=1)
df_prices['days_after'] = df_prices.apply(lambda row: (row['date'] - df_cluster.loc[row['camera_id'] - 1, 'release_time']).days, axis=1)

print(df_prices)

group1_price = pd.DataFrame()
group2_price = pd.DataFrame()
group3_price = pd.DataFrame()
group4_price = pd.DataFrame()

