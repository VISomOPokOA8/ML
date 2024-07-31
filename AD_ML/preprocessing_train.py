import pandas as pd
import pickle
from sklearn.preprocessing import OneHotEncoder

def camera_preprocessing(df):
    df = df.fillna(0)
    df = df.drop(['id', 'description', 'model', 'release_time'], axis=1)

    '''
    Classification
    '''
    # ISO
    iso_bins = [0, 12700, 51200, 1000000]
    iso_labels = [0, 1, 2]
    df['iso'] = pd.cut(df['iso'], bins=iso_bins, labels=iso_labels, right=True)

    # Video Resolution
    resolution_bins = [0, 2, 8]
    resolution_labels = [0, 1]
    df['video_resolution'] = pd.cut(df['video_resolution'], bins=resolution_bins, labels=resolution_labels, right=True)

    # Video Rate
    rate_bins = [0, 30, 60, 120]
    rate_labels = [0, 1, 2]
    df['video_rate'] = pd.cut(df['video_rate'], bins=rate_bins, labels=rate_labels, right=True)

    '''
    Onehot
    '''
    encoder_brand = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder_category = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder_iso = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder_video_resolution = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder_video_rate = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    # Brand
    brands_onehot = encoder_brand.fit_transform(df[['brand']])
    df_brands_onehot = pd.DataFrame(brands_onehot, columns=encoder_brand.get_feature_names_out(['brand']))
    with open('models/onehot/brand.pickle', 'wb') as f:
        pickle.dump(encoder_brand, f)

    # Category
    categories_onehot = encoder_category.fit_transform(df[['category']])
    df_categories_onehot = pd.DataFrame(categories_onehot, columns=encoder_category.get_feature_names_out(['category']))
    with open('models/onehot/category.pickle', 'wb') as f:
        pickle.dump(encoder_category, f)

    # ISO
    isoes_onehot = encoder_iso.fit_transform(df[['iso']])
    df_isoes_onehot = pd.DataFrame(isoes_onehot, columns=encoder_iso.get_feature_names_out(['iso']))
    with open('models/onehot/iso.pickle', 'wb') as f:
        pickle.dump(encoder_iso, f)

    # Video Resolution
    resolutions_onehot = encoder_video_resolution.fit_transform(df[['video_resolution']])
    df_resolutions_onehot = pd.DataFrame(resolutions_onehot, columns=encoder_video_resolution.get_feature_names_out(['video_resolution']))
    with open('models/onehot/resolution.pickle', 'wb') as f:
        pickle.dump(encoder_video_resolution, f)

    # Video Rate
    rates_onehot = encoder_video_rate.fit_transform(df[['video_rate']])
    df_rates_onehot = pd.DataFrame(rates_onehot, columns=encoder_video_rate.get_feature_names_out(['video_rate']))
    with open('models/onehot/rate.pickle', 'wb') as f:
        pickle.dump(encoder_video_rate, f)

    df_onehot = pd.concat([df, df_brands_onehot, df_categories_onehot, df_isoes_onehot, df_resolutions_onehot, df_rates_onehot], axis=1)
    df_onehot = df_onehot.drop(['brand', 'category', 'iso', 'video_resolution', 'video_rate'], axis=1)

    return df_onehot
