from datetime import datetime
import pandas as pd
import pymysql
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
