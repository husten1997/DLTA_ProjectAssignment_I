import os
import numpy as np
import pandas as pd

directory = "Data/"
file_path = os.path.join(directory, 'train.csv')
dtypes={
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
data = pd.read_csv(file_path, dtype=dtypes, usecols=list(dtypes.keys()))
data ['Time']=pd.to_datetime(data['timestamp'], unit='s')

file_path = os.path.join(directory, 'asset_details.csv')
details = pd.read_csv(file_path)

data = pd.merge(data,
                details,
                on ="Asset_ID",
                how = 'left')

print(data.head())

#%% Data Preperation

data_eval = data[data.timestamp >= 1622505660]
data = data[data.timestamp < 1622505660]

# feature generation

import ta
btc = data[data.Asset_ID == 1]
btc.set_index('timestamp', inplace = True)
btc = btc.reindex(range(btc.index[0], btc.index[-1] + 60, 60), method = 'pad')
btc.sort_index(inplace = True)

training_fraction = 0.70
training_size = int(np.floor(len(btc) * training_fraction))

train_data, test_data = btc[:training_size], btc[training_size:]

ROC = ta.momentum.ROCIndicator(close = train_data['Close'],window = 5,fillna=False)
train_data['ROC'] = ROC.roc()

ROC = ta.momentum.ROCIndicator(close = test_data['Close'],window = 5,fillna=False)
test_data['ROC'] = ROC.roc()

CMF =ta.volume.ChaikinMoneyFlowIndicator(close = train_data['Close'],high = train_data['High'], low = train_data['Low'], volume = train_data['Volume'], window = 5,fillna=False)
train_data['CMF'] = CMF.chaikin_money_flow()

CMF =ta.volume.ChaikinMoneyFlowIndicator(close = test_data['Close'],high = test_data['High'], low = test_data['Low'], volume = test_data['Volume'], window = 5,fillna=False)
test_data['CMF'] = CMF.chaikin_money_flow()

AVR =ta.volatility.AverageTrueRange(close = train_data['Close'],high = train_data['High'], low = train_data['Low'], window = 5,fillna=False)
train_data['AVR'] = AVR.average_true_range()

AVR =ta.volatility.AverageTrueRange(close = test_data['Close'],high = test_data['High'], low = test_data['Low'], window = 5,fillna=False)
test_data['AVR'] = AVR.average_true_range()

# drop NAs
train_data.dropna(inplace = True)
test_data.dropna(inplace = True)

train_data_features = train_data.drop(['Asset_ID','Time','Weight','Asset_Name'], axis = 1)
test_data_features = test_data.drop(['Asset_ID','Time','Weight','Asset_Name'], axis = 1)


# Scaling
from sklearn.preprocessing import MinMaxScaler

X_scaler = MinMaxScaler(feature_range = (0, 1))
X_train = train_data_features.drop(['Target'], axis = 1)
X_test = test_data_features.drop(['Target'], axis = 1)

y_train = train_data_features['Target'].values
y_test = test_data_features['Target'].values

X_train_ = X_scaler.fit_transform(X_train)
X_test_ = X_scaler.transform(X_test)