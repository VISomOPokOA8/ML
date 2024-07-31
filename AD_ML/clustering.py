import numpy as np
import pandas as pd
import pickle
import pymysql
from scipy.spatial.distance import cdist

import preprocessing

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

def camera_clustering(id):
    # Database Properties
    db = pymysql.connect(
        host='ad-database.cjo66a2aoqcq.ap-southeast-1.rds.amazonaws.com',
        port=3306,
        user='admin',
        passwd='lushuwen',
        db='CamC',
        charset='utf8')

    cursor = db.cursor()
    query = '''select * from camera where id = %s'''
    cursor.execute(query, [id])
    results = cursor.fetchall()
    columns = cursor.description
    cols = [col[0] for col in columns]
    camera = pd.DataFrame(results, columns=cols)
    # print(camera)
    camera_preprocessed = preprocessing.camera_preprocessing(camera)
    # print(camera_preprocessed)

    cursor.close()
    db.close()

    # Scaling
    with open('models/scaler.pickle', 'rb') as f:
        scaler = pickle.load(f)
    camera_scaled = camera_preprocessed.copy()
    columns = ['initial_price', 'effective_pixel', 'focus_point', 'continuous_shot']
    camera_scaled[columns] = scaler.transform(camera_scaled[columns])
    # print(camera_scaled)

    # PCA
    with open('models/pca.pickle', 'rb') as f:
        pca = pickle.load(f)
    camera_scaled = camera_scaled.drop('id', axis=1)
    camera_pca = pca.transform(camera_scaled)
    # print(camera_pca)

    # Hierarchical Clustering
    with open('models/centers.pickle', 'rb') as f:
        centers = pickle.load(f)
    distances = cdist(camera_pca, centers, metric='euclidean')
    camera_group = np.argmin(distances, axis=1)

    return camera_group
