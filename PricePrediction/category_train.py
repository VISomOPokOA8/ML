import pandas as pd
import numpy as np
from matplotlib.colors import ListedColormap

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

from matplotlib import pyplot as plt

import pickle

import pymysql

import camera_preprocessing

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
RED = '\033[91m'
RESET = '\033[0m'

"""
Data
"""
db = pymysql.connect(
        host='ad-database.cjo66a2aoqcq.ap-southeast-1.rds.amazonaws.com', port=3306,
        user='admin', passwd='lushuwen',
        db='CamC', charset='utf8')
cursor = db.cursor()

sql = '''select * from camera'''
cursor.execute(sql)
result = cursor.fetchall()
columns = cursor.description
cursor.close()
db.close()
cols = [col[0] for col in columns]
df = pd.DataFrame(result, columns=cols)
df.drop(columns=['id'], inplace=True)
df.drop(columns=['description'], inplace=True)

df_origin = df.copy()

df = camera_preprocessing.camera_preprocessing(df)

"""
OneHot
"""
# Initialize separate OneHotEncoder instances for each feature
encoder_brand = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoder_category = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoder_iso = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoder_video_resolution = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoder_video_rate = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

brands_onehot = encoder_brand.fit_transform(df[['brand']])
df_brands_onehot = pd.DataFrame(brands_onehot, columns=encoder_brand.get_feature_names_out(['brand']))

categories_onehot = encoder_category.fit_transform(df[['category']])
df_categories_onehot = pd.DataFrame(categories_onehot, columns=encoder_category.get_feature_names_out(['category']))

isoes_onehot = encoder_iso.fit_transform(df[['iso']])
df_isoes_onehot = pd.DataFrame(isoes_onehot, columns=encoder_iso.get_feature_names_out(['iso']))

resolutions_onehot = encoder_video_resolution.fit_transform(df[['video_resolution']])
df_resolutions_onehot = pd.DataFrame(resolutions_onehot, columns=encoder_video_resolution.get_feature_names_out(['video_resolution']))

rates_onehot = encoder_video_rate.fit_transform(df[['video_rate']])
df_rates_onehot = pd.DataFrame(rates_onehot, columns=encoder_video_rate.get_feature_names_out(['video_rate']))

df_onehot = pd.concat([df, df_brands_onehot, df_categories_onehot, df_isoes_onehot, df_resolutions_onehot, df_rates_onehot], axis=1)
df_onehot = df_onehot.drop(['brand', 'category', 'iso', 'video_resolution', 'video_rate'], axis=1)

with open('models/encoder_brand.pickle', 'wb') as f:
    pickle.dump(encoder_brand, f)

with open('models/encoder_category.pickle', 'wb') as f:
    pickle.dump(encoder_category, f)

with open('models/encoder_iso.pickle', 'wb') as f:
    pickle.dump(encoder_iso, f)

with open('models/encoder_video_resolution.pickle', 'wb') as f:
    pickle.dump(encoder_video_resolution, f)

with open('models/encoder_video_rate.pickle', 'wb') as f:
    pickle.dump(encoder_video_rate, f)

"""
PCA
"""
# print(df_onehot)
# print('\n')

standard = MinMaxScaler()
df_scaled = df_onehot.copy()
columns = ['initial_price', 'effective_pixel', 'focus_point', 'continuous_shot']
df_scaled[columns] = standard.fit_transform(df_onehot[columns])
# print(f"{RED}The scaled data: {RESET}")
# print(df_scaled)
# print('\n')

pca = PCA(n_components=6)
df_pca = pca.fit_transform(df_scaled)

with open('models/scaler.pickle', 'wb') as f:
    pickle.dump(standard, f)

with open('models/pca.pickle', 'wb') as f:
    pickle.dump(pca, f)

# explained_variance_ratio = pca.explained_variance_ratio_
# print("Cumulative Explained Variance Ratio: ", explained_variance_ratio)
# print(explained_variance_ratio.cumsum())
# print('\n')

# Loading
# loading = pca.components_.T * np.sqrt(pca.explained_variance_)
# loading_matrix = pd.DataFrame(loading, index=df_onehot.columns)
# print(f"\n{RED}The loading matrix: {RESET}")
# print(loading_matrix)

"""
Hierarchical Clustering
"""
hc_model = AgglomerativeClustering(linkage="ward", n_clusters=4)
labels = hc_model.fit_predict(df_pca)
df_origin['cluster'] = labels
df_origin.to_csv('clusters.csv', index=True)

df_labels = pd.DataFrame(df_pca)
df_labels['cluster'] = labels

centers = df_labels.groupby('cluster').mean().values

with open('models/centers.pickle', 'wb') as f:
    pickle.dump(centers, f)

# colors = ['red', 'yellow', 'green', 'blue', 'magenta']
# cmap = ListedColormap(colors)
# plt.figure(figsize=(12, 8))
# plt.scatter(df_pca[:, 0], df_pca[:, 1], c=labels, cmap=cmap)
# plt.title("Hierarchical Clustering")
# plt.show()
