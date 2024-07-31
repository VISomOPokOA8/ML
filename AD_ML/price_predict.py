from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pickle
import pymysql

import clustering

def price_predict(id):
    db = pymysql.connect(
        host='ad-database.cjo66a2aoqcq.ap-southeast-1.rds.amazonaws.com',
        port=3306,
        user='admin',
        passwd='lushuwen',
        db='CamC',
        charset='utf8')

    cursor = db.cursor()
    query = '''select release_time, initial_price from camera where id = %s'''
    cursor.execute(query, [id])
    results = cursor.fetchone()
    release_time = results[0]
    initial_price = results[1]
    print(release_time)
    print(initial_price)

    date_format = "%Y-%m-%d"
    release_date = pd.to_datetime(release_time, format=date_format)
    current_date = pd.to_datetime(datetime.today().strftime("%Y-%m-%d"))
    days = (current_date - release_date).days
    dates = []
    for i in range(26):
        dates.append(days + 7 * i)
    dates_arr = np.array(dates).reshape(-1, 1)

    group = clustering.camera_clustering(id)[0]
    with open(f'models/predict/{group}.pickle', 'rb') as f:
        model = pickle.load(f)

    prices = model.predict(dates_arr)

    dates_list = []
    for i in range(len(dates)):
        dates_list.append(pd.to_datetime(release_date + timedelta(dates[i])).strftime("%Y-%m-%d"))

    prices_list = []
    for i in range(len(prices)):
        prices_list.append(initial_price * prices[i])

    date_prices = {dates_list: prices_list for dates_list, prices_list in zip(dates_list, prices_list)}

    print(date_prices)

    return date_prices
