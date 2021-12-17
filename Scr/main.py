#%% Data Import

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


#%% Simple NN as performance reference

# Construct NN
import tensorflow as tf
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib')

model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape = (X_train_.shape[1])),
    tf.keras.layers.Dense(64, activation = 'relu'),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(32, activation = 'selu'),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(1)
])

model.compile(loss = 'mean_absolute_error', optimizer = 'adam')
history = model.fit(X_train_, y_train, epochs = 15, batch_size = 100000, validation_data = (X_test_, y_test))

#%%
plt.plot(history.history['loss'], label = 'training')
plt.plot(history.history['val_loss'], label = 'test')
plt.show()

#%% Show results
fig, axs = plt.subplots(1, 2, figsize = (12,8))

axs[0].scatter(model.predict(X_train_).flatten(), y_train)
axs[1].scatter(model.predict(X_test_).flatten(), y_test)
plt.show()
print(np.corrcoef(model.predict(X_train_).flatten(), y_train)[0, 1])
print(np.corrcoef(model.predict(X_test_).flatten(), y_test)[0, 1])

# Test Model performance
btc_eval = data_eval[data_eval.Asset_ID == 1]
btc_eval.set_index('timestamp', inplace = True)

ROC = ta.momentum.ROCIndicator(close = btc_eval['Close'],window = 5,fillna=False)
btc_eval['ROC'] = ROC.roc()

CMF =ta.volume.ChaikinMoneyFlowIndicator(close = btc_eval['Close'],high = btc_eval['High'], low = btc_eval['Low'], volume = btc_eval['Volume'], window = 5,fillna=False)
btc_eval['CMF'] = CMF.chaikin_money_flow()

AVR =ta.volatility.AverageTrueRange(close = btc_eval['Close'],high = btc_eval['High'], low = btc_eval['Low'], window = 5,fillna=False)
btc_eval['AVR'] = AVR.average_true_range()

btc_eval.dropna(inplace = True)

X_eval = btc_eval.drop(['Asset_ID','Time','Weight','Asset_Name','Target'], axis = 1)
X_eval_ = X_scaler.transform(X_eval)
y_eval = btc_eval['Target'].values

plt.scatter(model.predict(X_eval_).flatten(), y_eval)
plt.show()
print(np.corrcoef(model.predict(X_eval_).flatten(), y_eval)[0, 1])
