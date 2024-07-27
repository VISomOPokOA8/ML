import pandas as pd
import numpy as np
from matplotlib.colors import ListedColormap

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

from matplotlib import pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
RED = '\033[91m'
RESET = '\033[0m'

"""
Data
"""
df = pd.read_excel("Models.xlsx")
df_origin = df.copy()

"""
Preprocessing
"""
df = df.fillna(0)
df = df.drop(['Model', 'Release_Time'], axis=1)

# ISO
iso_bins = [0, 12700, 51200, 1000000]
iso_labels = [0, 1, 2]
df['ISO'] = pd.cut(df['ISO'], bins=iso_bins, labels=iso_labels, right=True)

# Video Resolution
resolution_bins = [0, 2, 8]
resolution_labels = [0, 1]
df['Video_Resolution'] = pd.cut(df['Video_Resolution'], bins=resolution_bins, labels=resolution_labels, right=True)

# Video Rate
rate_bins = [0, 30, 60, 120]
rate_labels = [0, 1, 2]
df['Video_Rate'] = pd.cut(df['Video_Rate'], bins=rate_bins, labels=rate_labels, right=True)

print(f"{RED}The preprocessed data: {RESET}")
print(df)
print('\n')

"""
OneHot
"""
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

brands = df[['Brand']]
brands_onehot = encoder.fit_transform(brands)
df_brands_onehot = pd.DataFrame(brands_onehot, columns=encoder.get_feature_names_out(['Brand']))

types = df[['Type']]
types_onehot = encoder.fit_transform(types)
df_types_onehot = pd.DataFrame(types_onehot, columns=encoder.get_feature_names_out(['Type']))

isoes = df[['ISO']]
isoes_onehot = encoder.fit_transform(isoes)
df_isoes_onehot = pd.DataFrame(isoes_onehot, columns=encoder.get_feature_names_out(['ISO']))

resolutions = df[['Video_Resolution']]
resolutions_onehot = encoder.fit_transform(resolutions)
df_resolutions_onehot = pd.DataFrame(resolutions_onehot, columns=encoder.get_feature_names_out(['Video_Resolution']))

rates = df[['Video_Rate']]
rates_onehot = encoder.fit_transform(rates)
df_rates_onehot = pd.DataFrame(rates_onehot, columns=encoder.get_feature_names_out(['Video_Rate']))

df_onehot = pd.concat([df, df_brands_onehot, df_types_onehot, df_isoes_onehot, df_resolutions_onehot, df_rates_onehot], axis=1)
df_onehot = df_onehot.drop(['Brand', 'Type', 'ISO', 'Video_Resolution', 'Video_Rate'], axis=1)

"""
PCA
"""
standard = MinMaxScaler()
df_scaled = df_onehot.copy()
columns = df_onehot.columns[df_onehot.columns.get_loc('Initial_Price'):df_onehot.columns.get_loc('Brand_Canon')]
df_scaled[columns] = standard.fit_transform(df_onehot[columns])
print(f"{RED}The scaled data: {RESET}")
print(df_scaled)
print('\n')

pca = PCA(n_components=6)
df_pca = pca.fit_transform(df_scaled)
# df_pca = pca.fit_transform(df_onehot)

explained_variance_ratio = pca.explained_variance_ratio_
print("Cumulative Explained Variance Ratio: ", explained_variance_ratio)
print(explained_variance_ratio.cumsum())
print('\n')

# Loading
loading = pca.components_.T * np.sqrt(pca.explained_variance_)
loading_matrix = pd.DataFrame(loading, index=df_onehot.columns)
print(f"\n{RED}The loading matrix: {RESET}")
print(loading_matrix)

"""
Hierarchical Clustering
"""
hc_model = AgglomerativeClustering(linkage="ward", n_clusters=4)
row_cluster_map = hc_model.fit_predict(df_pca)
df_origin['Cluster'] = row_cluster_map
df_origin.to_csv('Clusters.csv', index=True)

colors = ['red', 'yellow', 'green', 'blue', 'magenta']
cmap = ListedColormap(colors)
plt.figure(figsize=(12, 8))
plt.scatter(df_pca[:, 0], df_pca[:, 1], c=row_cluster_map, cmap=cmap)
plt.title("Hierarchical Clustering")
plt.show()



# plt.figure(figsize=(12, 8))
# plt.title("Hierarchical Clustering")
# plt.xlabel("Species")
# plt.ylabel("distance")
# dendrogram(linkage(df_pca, method='ward'), leaf_font_size=8)
# plt.axhline(y=8)
# plt.show()

