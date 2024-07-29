import pandas as pd
import pymysql

import category_predict

'''
Retrieve historical price
'''
db = pymysql.connect(
        host='ad-database.cjo66a2aoqcq.ap-southeast-1.rds.amazonaws.com', port=3306,
        user='admin', passwd='lushuwen',
        db='CamC', charset='utf8')
cursor = db.cursor()

sql = '''select * from camera'''
cursor.execute(sql)
cameras = cursor.fetchall()
columns = cursor.description
cols = [col[0] for col in columns]
df_cameras = pd.DataFrame(cameras, columns=cols)

sql = '''select * from price where platform = 'Amazon' '''
cursor.execute(sql)
prices = cursor.fetchall()
columns = cursor.description
cols = [col[0] for col in columns]
df_prices = pd.DataFrame(prices, columns=cols)

cursor.close()
db.close()

df_0 = pd.DataFrame()
df_1 = pd.DataFrame()
df_2 = pd.DataFrame()
df_3 = pd.DataFrame()

