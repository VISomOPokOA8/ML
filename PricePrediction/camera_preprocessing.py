import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def camera_preprocessing(df):
    """
    Preprocessing
    """
    df = df.fillna(0).infer_objects()
    df = df.drop(['model', 'release_time'], axis=1)

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

    return df