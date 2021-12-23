#(1) Step: Descriptive Analysis

#Data Import
import os
import numpy as np
import pandas as pd

directory = "C:/Users/Albert Nietz/PyCharm_Projects/DLTA, First Project Assignment/DLTA_ProjectAssignment_I/Data"
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
print(details.head())


#Filter train dataset by DOGE and ETH & update asset_ids
ethdoge = data[(data.Asset_ID == 4) | (data.Asset_ID == 6)]
print(ethdoge.head())

#Filter asset_details dataset by ETH and DOGE & update asset_ids
detailsnew = details[(data.Asset_ID == 4) | (data.Asset_ID == 6)]
print(details.head())

#Change index values from 5 and 13 to 0 and 1
detailsnew.index = [0, 1]
print(detailsnew.head())

#Create subplots: return over time
import matplotlib.pyplot as plt

cols = 1
rows = len(detailsnew.Asset_ID)

position = range(1,rows + 1)

fig = plt.figure(1)
fig.set_figheight(20)
fig.set_figwidth(20)

#Add every single subplot to the figure with a for loop
for k in range(rows):

    tmp_df = ethdoge[ethdoge.Asset_ID == detailsnew.Asset_ID[k]]
    ax = fig.add_subplot(rows, cols, position[k])
    ax.plot(tmp_df.Time, tmp_df.Target)
    ax.set_title(detailsnew.Asset_Name[k])

plt.show()
del tmp_df

#Combine subplots in one plot
#TODO: Programming dynamically instead of statically
eth = ethdoge[ethdoge.Asset_ID == detailsnew.Asset_ID[0]]
doge = ethdoge[ethdoge.Asset_ID == detailsnew.Asset_ID[1]]
plt.figure(figsize=(12,4))
plt.plot(eth.Time, eth.Target)
plt.plot(doge.Time, doge.Target)
plt.title('Returns over time of ' + detailsnew.Asset_Name[0] + ' (blue) and ' + detailsnew.Asset_Name[1] + ' (orange)')
plt.show()
del eth, doge

#Create subplots: histogram / distribution
import matplotlib.pyplot as plt

cols = 1
rows = len(detailsnew.Asset_ID)

position = range(1,rows + 1)

fig = plt.figure(1)
fig.set_figheight(20)
fig.set_figwidth(20)

#Add every single subplot to the figure with a for loop
for k in range(rows):

    tmp_df = ethdoge[ethdoge.Asset_ID == detailsnew.Asset_ID[k]]
    ax = fig.add_subplot(rows, cols, position[k])
    ax.hist(tmp_df.Target, bins = 50)
    ax.set_xlim(-0.1, 0.1)
    ax.set_title(detailsnew.Asset_Name[k])

plt.show()
del tmp_df

#Create new dataframes by adding BTC to old dataframes
ethdogebtc = ethdoge.append(data[data.Asset_ID == 1])

tmp_detailsnew = detailsnew.append(details[details.Asset_ID == 1])
print(tmp_detailsnew.head())

#Create new dataframe for correlation over time & heat map
all_timestamps = np.sort(ethdogebtc['timestamp'].unique())
targets = pd.DataFrame(index=all_timestamps)

for i, id_ in enumerate(tmp_detailsnew.Asset_ID):
    asset = ethdogebtc[ethdogebtc.Asset_ID == id_].set_index(keys='timestamp')
    price = pd.Series(index=all_timestamps, data=asset['Close'])
    targets[tmp_detailsnew.Asset_Name[i]] = (
                                             price.shift(periods=-16) /
                                             price.shift(periods=-1)
                                     ) - 1

print(targets.head())

#Create subplots: 7-day-correlation over time
cols = 1
rows = len(tmp_detailsnew.Asset_ID)

position = range(1,rows + 1)

fig = plt.figure(1)
fig.set_figheight(20)
fig.set_figwidth(20)

#Add every single subplot to the figure with a for loop
cols = 1
rows = len(tmp_detailsnew.Asset_ID)

position = range(1,rows + 1)

fig = plt.figure(1)
fig.set_figheight(20)
fig.set_figwidth(20)

#TODO: Programming dynamically instead of statically
corr_time = targets.groupby(targets.index//(10000*60)).corr().loc[:,tmp_detailsnew.Asset_Name[0]].loc[:,
            tmp_detailsnew.Asset_Name[1]]
ax = fig.add_subplot(rows, cols, position[0])
ax.plot(corr_time)
ax.set_title('7-Days-Corr. between ' + tmp_detailsnew.Asset_Name[0] + ' and ' + tmp_detailsnew.Asset_Name[1])
plt.xticks([])
plt.xlabel("Time")
plt.ylabel("Correlation")

corr_time = targets.groupby(targets.index//(10000*60)).corr().loc[:,tmp_detailsnew.Asset_Name[0]].loc[:,
            tmp_detailsnew.Asset_Name[2]]
ax = fig.add_subplot(rows, cols, position[1])
ax.plot(corr_time)
ax.set_title('7-Days-Corr. between ' + tmp_detailsnew.Asset_Name[0] + ' and ' + tmp_detailsnew.Asset_Name[2])
plt.xticks([])
plt.xlabel("Time")
plt.ylabel("Correlation")

corr_time = targets.groupby(targets.index//(10000*60)).corr().loc[:,tmp_detailsnew.Asset_Name[1]].loc[:,
            tmp_detailsnew.Asset_Name[2]]
ax = fig.add_subplot(rows, cols, position[2])
ax.plot(corr_time)
ax.set_title('7-Days-Corr. between ' + tmp_detailsnew.Asset_Name[1] + ' and ' + tmp_detailsnew.Asset_Name[2])
plt.xticks([])
plt.xlabel("Time")
plt.ylabel("Correlation")

plt.show()

#Create heat map
import seaborn as sns
sns.heatmap(targets.corr())
plt.show()

#(2) Step: Feature Engineering

#Create subplots: closeprice development
cols = 1
rows = len(detailsnew.Asset_ID)

position = range(1,rows + 1)

fig = plt.figure(1)
fig.set_figheight(20)
fig.set_figwidth(20)

#Add every single subplot to the figure with a for loop
for k in range(rows):

    tmp_df = ethdoge[ethdoge.Asset_ID == detailsnew.Asset_ID[k]]
    ax = fig.add_subplot(rows, cols, position[k])
    ax.plot(tmp_df.Time, tmp_df.Close)
    ax.set_title(detailsnew.Asset_Name[k])

plt.show()
del tmp_df

#Create subplots for each coin: split up closeprice development to determine a suitable training and test period
from datetime import datetime
from datetime import date
from dateutil.relativedelta import relativedelta

for k in range(len(detailsnew.Asset_ID)):

    #Set and sort the index
    coin_tmp = ethdoge[ethdoge.Asset_ID == detailsnew.Asset_ID[k]]
    coin_tmp.set_index('timestamp', inplace = True)
    coin_tmp = coin_tmp.reindex(range(coin_tmp.index[0], coin_tmp.index[-1] + 60, 60), method='pad')
    coin_tmp.sort_index(inplace=True)

    #Calculate number of month in the dataset
    starting_date = coin_tmp.Time.iloc[0]
    ending_date = date(2021, 9, 21)
    numb_month = (ending_date.year - starting_date.year) * 12 + (ending_date.month - starting_date.month)

    #Calculate six month timesplits
    timesplits = [starting_date + i * relativedelta(months = 6) for i in range(numb_month // 6)] + [coin_tmp.Time.iloc[-1]]

    Tot = len(timesplits) - 1
    Cols = 2

    Rows = Tot // Cols
    Rows += Tot % Cols

    Position = range(1, Tot + 1)

    fig = plt.figure(1)
    fig.set_figheight(30)
    fig.set_figwidth(20)
    fig.suptitle(detailsnew.Asset_Name[k])

    #Add every single subplot to the figure with a for loop
    for j in range(Tot):
        coin_tmp2 = coin_tmp.loc[datetime.timestamp(timesplits[j]):datetime.timestamp(timesplits[j + 1])]
        ax = fig.add_subplot(Rows, Cols, Position[j])
        ax.plot(coin_tmp2.Time, coin_tmp2.Close)

    plt.show()


""""
#Create seven days correlation over time between ETH and DOGE
corr_time = targets.groupby(targets.index//(10000*60)).corr().loc[:,"Ethereum"].loc[:,"Dogecoin"]
corr_time.plot()
plt.xticks([])
plt.xlabel("Time")
plt.ylabel("Correlation")
plt.title("Correlation between ETH and DOGE over time");
plt.show()



print(targets.shape[0])
print(corr_time.shape[0])
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

AVR =ta.volatility.AverageTrueRange(close = test_data['Close'],high = test_data['Hi gh'], low = test_data['Low'], window = 5,fillna=False)
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
"""
