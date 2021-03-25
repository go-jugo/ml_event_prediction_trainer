import pandas as pd
from sklearn.preprocessing import StandardScaler
from ..monitoring.time_it import timing

@timing
def standardize_features(df, errorcode_col, scaler = StandardScaler()):
    feature_columns = [col for col in df.columns if col not in ['global_timestamp', errorcode_col, 'samples_used', 'window_start', 'window_end', 'window_length']]
    df[feature_columns] = df[feature_columns].to_numpy()
    df[feature_columns] = scaler.fit_transform(df[feature_columns])
    return df