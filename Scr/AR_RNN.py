import os
import numpy as np
import pandas as pd
# TODO: Add import function

def import_data(dir):
    file_path = os.path.join(dir, 'train.csv')
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
    data = pd.read_csv(file_path, dtype=dtypes, usecols=list(dtypes.keys()))
    data['Time'] = pd.to_datetime(data['timestamp'], unit='s')

    file_path = os.path.join(directory, 'asset_details.csv')
    details = pd.read_csv(file_path)

    data = pd.merge(data,
                    details,
                    on="Asset_ID",
                    how='left')

    return data

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
data['Time'] = pd.to_datetime(data['timestamp'], unit='s')

file_path = os.path.join(directory, 'asset_details.csv')
details = pd.read_csv(file_path)

data = pd.merge(data,
                details,
                on ="Asset_ID",
                how = 'left')

print(data.head())

#%% Data Preperation
# doggicoin ID: 4
# etherium ID: 6

# TODO: Question: Etherium or Etherium Classic?

# TODO: Add Data selection function and seperation into test/trainings data
coin_ID = 4

data_eval = data[data.timestamp >= 1622505660]
data = data[(data.timestamp < 1622505660) & (data.timestamp >= 1609459200)]
#1.1.2021: 1609459200
#1.1.2020: 1577836800

# feature generation

btc = data[data.Asset_ID == coin_ID]
btc.set_index('timestamp', inplace = True)
btc = btc.reindex(range(btc.index[0], btc.index[-1] + 60, 60), method = 'pad')
btc.sort_index(inplace = True)

btc_eval = data_eval[data_eval.Asset_ID == coin_ID]
btc_eval.set_index('timestamp', inplace = True)
btc_eval = btc_eval.reindex(range(btc_eval.index[0], btc_eval.index[-1] + 60, 60), method = 'pad')
btc_eval.sort_index(inplace = True)

training_fraction = 0.70
training_size = int(np.floor(len(btc) * training_fraction))

train_data, test_data = btc[:training_size], btc[training_size:]

# Unload btc dataset to reduce memory usage
btc = None

# drop NAs
train_data.dropna(inplace = True)
test_data.dropna(inplace = True)

train_data_features = train_data.drop(['Asset_ID','Time','Weight','Asset_Name'], axis = 1)
test_data_features = test_data.drop(['Asset_ID','Time','Weight','Asset_Name'], axis = 1)

#price_train = train_data_features['Close'].values
#price_test = test_data_features['Close'].values

price_train = train_data_features['Target'].values
price_test = test_data_features['Target'].values

test_data_features = None
train_data_features = None


#%% Scaling

# TODO: Add Scaling Function

from sklearn.preprocessing import MinMaxScaler

X_scaler = MinMaxScaler()
# X_train = train_data_features.drop(['Target'], axis = 1)
# X_test = test_data_features.drop(['Target'], axis = 1)


price_train_ = X_scaler.fit_transform(price_train.reshape(-1, 1))
price_test_ = X_scaler.transform(price_test.reshape(-1, 1))


#%% Generate Feature Matrix

# TODO: Add function for feature matrix generation

#ar_order = 60 * 60 # Use last hour of data
ar_order = 60 * 60
forecast_steps = 15 * 60 # Predict the next 15 min

target_var = ['Close']
predictor_var = ['Close']

index_range = range(ar_order, len(price_train) - forecast_steps - 1)
X_data_train = []
Y_data_train = []
for i in index_range:
    X_data_train.append(price_train[(i - ar_order):i])
    #Y_data_train.append(price_train[(i+1):(i+forecast_steps)])
    #Y_data_train.append(price_train[[(i + 1), (i + forecast_steps + 1)]])
    Y_data_train.append(price_train[[(i + 1)]])


index_range = range(ar_order, len(price_test) - forecast_steps - 1)
X_data_test = []
Y_data_test = []
for i in index_range:
    X_data_test.append(price_test[(i - ar_order):i])
    #Y_data_test.append(price_test[(i+1):(i+forecast_steps)])
    #Y_data_test.append(price_test[[(i + 1), (i + forecast_steps + 1]])
    Y_data_test.append(price_test[[(i + 1)]])

X_data_train = np.array(X_data_train)
X_data_test = np.array(X_data_test)
Y_data_train = np.array(Y_data_train)
Y_data_test = np.array(Y_data_test)


#%% Plot Prices
import matplotlib.pylab as plt

fig, axs = plt.subplots(1, 1, figsize = (12, 6))
fig.autofmt_xdate()

axs.plot(train_data['Close'], label = 'data')
axs.plot(test_data['Close'], label = 'prediction', alpha = 0.5)
axs.legend()

plt.show()


fig, axs = plt.subplots(2, 1, figsize = (12, 6))
fig.autofmt_xdate()

axs[0].plot(train_data['Close'], label = 'data')
axs[0].plot(test_data['Close'], label = 'prediction', alpha = 0.5)
axs[0].plot(btc_eval['Close'], label = "Evaluation", alpha = 0.3 )
axs[0].legend()

axs[1].plot(train_data['High'] - train_data['Low'], label = 'data')
axs[1].plot(test_data['High'] - test_data['Low'], label = 'prediction', alpha = 0.5)
axs[1].plot(btc_eval['High'] - btc_eval['Low'], label = "Evaluation", alpha = 0.3 )
axs[1].legend()

plt.show()


#%% Setup Autoencoder
import tensorflow as tf

reduced_dimension = int(ar_order / 60)


encoder = tf.keras.Sequential([
    tf.keras.layers.Dense(reduced_dimension, activation = 'linear', input_shape = [X_data_train.shape[1]], use_bias = False)
])

decoder = tf.keras.Sequential([
    tf.keras.layers.Dense(X_data_train.shape[1], activation = 'linear', input_shape = [reduced_dimension], use_bias = False)
])

autoencoder = tf.keras.Sequential([encoder, decoder])

autoencoder.compile(loss = 'mse', optimizer = 'adam')

auto_history = autoencoder.fit(X_data_train, X_data_train, epochs = 20, verbose = 1, validation_data = (X_data_test, X_data_test))

#%% History Plot
plt.plot(auto_history.history['loss'], label = 'loss')
plt.plot(auto_history.history['val_loss'], label = 'validation_loss')
plt.legend()
plt.show()

#%% Autoencoder prediction perfomance
X_data_train_encoded = autoencoder.predict(X_data_train)

# Calculating the columnwise correlation
X = (X_data_train_encoded - X_data_train_encoded.mean(axis=0)) / X_data_train_encoded.std(axis=0)
Y = (X_data_train - X_data_train.mean(axis=0)) / X_data_train.std(axis=0)
pearson_r = np.dot(X.T, Y) / X.shape[0]

X = None
Y = None

# Five-Number-Statiscs of the Corr-Vector
print(np.min(pearson_r))
print(np.max(pearson_r))
print(np.std(pearson_r))
print(np.mean(pearson_r))
print(np.median(pearson_r))

# Plot
import random as rand
rand_cols = rand.sample(range(X_data_train.shape[1]), 5)

fig, axs = plt.subplots(5, 1, figsize = (24, 30))
fig.autofmt_xdate()

for i in range(len(rand_cols)):
    axs[i].plot(X_data_train[:, rand_cols[i]], label='data')
    axs[i].plot(X_data_train_encoded[:, rand_cols[i]], alpha=0.5, label='prediction')
    axs[i].title.set_text('Plot of col: ' + str(rand_cols[i]))
    axs[i].legend()

plt.show()

# Unload X_data_train_encoded
X_data_train_encoded = None


#%% Generate Reduced feature set

# TODO: Add function for selection of featureset reduction method and feature set reduction

def generate_redFeatureSet(data, encoder = None, file = None, method = "average"):
    if method == "autoencoder":
        if encoder is None and file is None:
            print("No Weights File path or autoencoder given!")
        elif encoder is None and file is not None:
            weights = np.matrix(pd.read_csv(file))
        elif encoder is not None:
            weights = np.matrix(encoder.get_weights()[0])
    elif method == "average":
        weights = np.zeros((3600, 60))

        for j in range(60):
            for i in range(60):
                weights[(j * 60) + i, j] = 1 / 60

    else:
        print("Unknown method")

    data = np.matmul(data.reshape((-1, 3600)), weights)
    data = np.array(data)

    return data


weights = np.matrix(encoder.get_weights()[0])

if ('encoder' not in globals()): weights = np.matrix(pd.read_csv("weights.csv"))

pd.DataFrame(weights).to_csv("weights.csv", header=None, index=None)

X_data_train_red = np.matmul(X_data_train, weights)
X_data_test_red = np.matmul(X_data_test, weights)

X_data_train_red = np.array(X_data_train_red)
X_data_test_red = np.array(X_data_test_red)

#%% Generate reduced feature set (60 sec average)

weights = np.zeros((3600, 60))

for j in range(60):
    for i in range(60):
        weights[(j * 60) + i, j] = 1/60

X_data_train_red = np.matmul(X_data_train.reshape((-1, 3600)), weights)
X_data_test_red = np.matmul(X_data_test.reshape((-1, 3600)), weights)

X_data_train_red = np.array(X_data_train_red)
X_data_test_red = np.array(X_data_test_red)

#%% Setup NN
import tensorflow as tf
def build_ARRNN(hp):
    input_shape = X_data_train_red.shape[1]

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape = (input_shape, 1)))

    if hp.Boolean("LSTM"):
        model.add(tf.keras.layers.LSTM(hp.Choice('LSTMunits', [60, 120, 240]), return_sequences = False, activation=hp.Choice("activation", ["relu", "tanh"])))
    else:
        model.add(tf.keras.layers.GRU(hp.Choice('GRUunits', [60, 120, 240])))

    if hp.Boolean("dropout"):
        model.add(tf.keras.layers.Dropout(rate=0.25))

    model.add(tf.keras.layers.Dense(1))

    learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")

    model.compile(loss = 'mean_squared_error', optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate))
    return model

input_shape = X_data_train_red.shape[1]

# define a recurrent network with Gated Recurrent Units
model = tf.keras.Sequential([
    #tf.keras.layers.InputLayer(input_shape = (ar_order, 1)),
    tf.keras.layers.InputLayer(input_shape = (input_shape, 1)),
    #tf.keras.layers.Dense(60, activation = 'linear', input_shape = [ar_order], use_bias = False),
    #tf.keras.layers.GRU(5),
    tf.keras.layers.LSTM(60, return_sequences = False)
    #tf.keras.layers.GRU(60)
    ,tf.keras.layers.Dense(1)
])

model.compile(loss = 'mean_squared_error', optimizer = 'adam')
model.summary()

#%% Reshape Data

X_data_train_red = np.reshape(X_data_train_red, X_data_train_red.shape + (1,))
X_data_test_red = np.reshape(X_data_test_red, X_data_test_red.shape + (1,))
Y_data_train = np.reshape(Y_data_train, Y_data_train.shape + (1,))
Y_data_test = np.reshape(Y_data_test, Y_data_test.shape + (1,))

#%% Fit RNN
ARRNN_history = model.fit(X_data_train_red, Y_data_train, epochs = 20, validation_data = (X_data_test_red, Y_data_test), batch_size=1024)

#%% History Plot
import matplotlib.pylab as plt

plt.plot(ARRNN_history.history['loss'], label = 'loss')
plt.plot(ARRNN_history.history['val_loss'], label = 'validation_loss')
plt.legend()
plt.show()


#%% Model performance
Y_train_hat = model.predict(X_data_train_red)
Y_test_hat = model.predict(X_data_test_red)

#%% Test
from scipy.stats.stats import pearsonr

var = lambda x: (1/(len(x)-1)) * (np.sum(x * x) - (1/len(x)) * (np.sum(x)**2))
cov = lambda x, y: (1/(len(x)-1)) * (np.sum(x * y) - (1/len(x)) * np.sum(x) * np.sum(y))
corr = lambda x, y: (cov(x, y))/np.sqrt(var(x) * var(y))

#%%
print("In-sample corr: " + str(corr(Y_train_hat.reshape((-1)), Y_data_train.reshape((-1)))))
print("Out-of-sample corr: " + str(corr(Y_test_hat.reshape((-1)), Y_data_test.reshape((-1)))))


#%% Plot
fig, axs = plt.subplots(1, 1, figsize = (12, 6))
fig.autofmt_xdate()

axs.plot(train_data['Close'].values, label = 'data')
axs.plot(Y_train_hat[:, 1], alpha = 0.5, label = 'prediction')
axs.legend()

plt.show()


fig, axs = plt.subplots(1, 1, figsize = (12, 6))
fig.autofmt_xdate()

axs.plot(test_data['Close'].values, label = 'data')
axs.plot(Y_test_hat[:, 1], alpha = 0.5, label = 'prediction')
axs.legend()

plt.show()

#%% Tune the Model
import keras_tuner as kt

tuner = kt.RandomSearch(build_ARRNN, objective='val_loss', max_trials=5)

tuner.search(X_data_train_red, Y_data_train, epochs=10, validation_data=(X_data_test_red, Y_data_test), batch_size = 512)
#%%
best_model = tuner.get_best_models()[0]





