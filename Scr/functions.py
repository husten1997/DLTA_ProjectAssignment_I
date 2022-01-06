import os
import numpy as np
import pandas as pd

def import_data(dir):

    dtypes = {
        'timestamp': np.int64,
        'Asset_ID': np.int8,
        'Count': np.int32,
        'Open': np.float64,
        'High': np.float64,
        'Low': np.float64,
        'Close': np.float64,
        'Volume': np.float64,
        'VWAP': np.float64,
        'Target': np.float64,
    }

    file_path = os.path.join(dir, 'train.csv')
    all_data = pd.read_csv(file_path, dtype=dtypes, usecols=list(dtypes.keys()))
    all_data['Time'] = pd.to_datetime(all_data['timestamp'], unit='s')

    file_path = os.path.join(dir, 'asset_details.csv')
    all_data_details = pd.read_csv(file_path)

    all_data = pd.merge(all_data,
                         all_data_details,
                         on="Asset_ID",
                         how='left')

    return all_data, all_data_details