import numpy as np
import pymysql
import pickle

import pandas as pd
from scipy.spatial.distance import cdist

import camera_preprocessing

def classification(id):
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)

    db = pymysql.connect(
        host='ad-database.cjo66a2aoqcq.ap-southeast-1.rds.amazonaws.com', port=3306,
        user='admin', passwd='lushuwen',
        db='CamC', charset='utf8')
    cursor = db.cursor()

    sql = '''select * from camera where id = %s'''

    cursor.execute(sql, (id, ))
    result = cursor.fetchall()
    columns = cursor.description
    cursor.close()
    db.close()

    cols = [col[0] for col in columns]
    df = pd.DataFrame(result, columns=cols)
    df.drop(columns='id', inplace=True)
    df.drop(columns='description', inplace=True)

    df = camera_preprocessing.camera_preprocessing(df)

    '''
    Onehot
    '''
    with open('models/encoder_brand.pickle', 'rb') as f:
        encoder_brand = pickle.load(f)

    with open('models/encoder_category.pickle', 'rb') as f:
        encoder_category = pickle.load(f)

    with open('models/encoder_iso.pickle', 'rb') as f:
        encoder_iso = pickle.load(f)

    with open('models/encoder_video_resolution.pickle', 'rb') as f:
        encoder_resolution = pickle.load(f)

    with open('models/encoder_video_rate.pickle', 'rb') as f:
        encoder_rate = pickle.load(f)

    brands_onehot = encoder_brand.transform(df[['brand']])
    df_brands_onehot = pd.DataFrame(brands_onehot, columns=encoder_brand.get_feature_names_out(['brand']))

    categories_onehot = encoder_category.transform(df[['category']])
    df_categories_onehot = pd.DataFrame(categories_onehot, columns=encoder_category.get_feature_names_out(['category']))

    isoes_onehot = encoder_iso.transform(df[['iso']])
    df_isoes_onehot = pd.DataFrame(isoes_onehot, columns=encoder_iso.get_feature_names_out(['iso']))

    resolutions_onehot = encoder_resolution.transform(df[['video_resolution']])
    df_resolutions_onehot = pd.DataFrame(resolutions_onehot, columns=encoder_resolution.get_feature_names_out(['video_resolution']))

    rates_onehot = encoder_rate.transform(df[['video_rate']])
    df_rates_onehot = pd.DataFrame(rates_onehot, columns=encoder_rate.get_feature_names_out(['video_rate']))

    df_onehot = pd.concat([df, df_brands_onehot, df_categories_onehot, df_isoes_onehot, df_resolutions_onehot, df_rates_onehot], axis=1)
    df_onehot = df_onehot.drop(['brand', 'category', 'iso', 'video_resolution', 'video_rate'], axis=1)

    '''
    PCA
    '''
    with open('models/scaler.pickle', 'rb') as f:
        scaler = pickle.load(f)
    df_scaled = df_onehot.copy()
    desired_order = ['initial_price', 'effective_pixel', 'focus_point', 'continuous_shot',
                     'brand_Canon', 'brand_Nikon', 'brand_Sony',
                     'category_DC', 'category_MIC', 'category_SLR',
                     'iso_0', 'iso_1', 'iso_2',
                     'video_resolution_0', 'video_resolution_1',
                     'video_rate_0', 'video_rate_1', 'video_rate_2']
    df_scaled = df_scaled[desired_order]
    columns = ['initial_price', 'effective_pixel', 'focus_point', 'continuous_shot']
    df_scaled[columns] = scaler.transform(df_onehot[columns])

    with open('models/pca.pickle', 'rb') as f:
        pca = pickle.load(f)
    df_pca = pca.transform(df_scaled)

    with open('models/centers.pickle', 'rb') as f:
        centers = pickle.load(f)
    distances = cdist(df_pca, centers, metric='euclidean')
    closest = np.argmin(distances, axis=1)

    return closest
