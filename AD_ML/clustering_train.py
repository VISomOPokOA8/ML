import pandas as pd
import pickle
import pymysql
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

import preprocessing_train

# Print Settings
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
RED = '\033[91m'
RESET = '\033[0m'

# Database Properties
db = pymysql.connect(
    host='ad-database.cjo66a2aoqcq.ap-southeast-1.rds.amazonaws.com',
    port=3306,
    user='admin',
    passwd='lushuwen',
    db='CamC',
    charset='utf8')

cursor = db.cursor()
sql = '''select * from camera'''
cursor.execute(sql)
results = cursor.fetchall()
columns = cursor.description
cols = [col[0] for col in columns]
cameras = pd.DataFrame(results, columns=cols)
print(f"{RED}The original data: {RESET}")
print(cameras)
print("\n")

cursor.close()
db.close()

cameras_preprocessed = preprocessing_train.camera_preprocessing(cameras)
print(f"{RED}The preprocessed data: {RESET}")
print(cameras_preprocessed)
print("\n")

'''
Scaling
'''
scaler = MinMaxScaler()
cameras_scaled = cameras_preprocessed.copy()
columns = ['initial_price', 'effective_pixel', 'focus_point', 'continuous_shot']
cameras_scaled[columns] = scaler.fit_transform(cameras_scaled[columns])
with open('models/scaler.pickle', 'wb') as f:
    pickle.dump(scaler, f)
print(f"{RED}The scaled data: {RESET}")
print(cameras_scaled)
print("\n")

'''
PCA
'''
pca = PCA(n_components=6)
cameras_pca = pca.fit_transform(cameras_scaled)
with open('models/pca.pickle', 'wb') as f:
    pickle.dump(pca, f)
print(f"{RED}PCA: {RESET}")
print(cameras_pca)
print("\n")

'''
Hierarchical Clustering
'''
hc_model = AgglomerativeClustering(linkage="ward", n_clusters=4)
labels = hc_model.fit_predict(cameras_pca)
cameras_clustered = cameras.copy()
cameras_clustered['cluster'] = labels
cameras_clustered.to_csv('datas/clustering.csv', index=True)

cameras_labels = pd.DataFrame(cameras_pca)
cameras_labels['cluster'] = labels
centers = cameras_labels.groupby('cluster').mean().values
with open('models/centers.pickle', 'wb') as f:
    pickle.dump(centers, f)

print(f"{RED}The clustered data: {RESET}")
print(cameras_clustered)
print("\n")
print(f"{RED}The clustering center points: {RESET}")
print(centers)
print("\n")
