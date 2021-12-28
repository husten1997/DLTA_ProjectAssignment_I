import os
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt

class AdvancedModel() :

    def __init__(self, coins):

        self.coins = coins

        directory = "C:/Users/Albert Nietz/PyCharm_Projects/DLTA, First Project Assignment/DLTA_ProjectAssignment_I/Data"
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

        file_path = os.path.join(directory, 'train.csv')
        data = pd.read_csv(file_path, dtype=dtypes, usecols=list(dtypes.keys()))
        data['Time'] = pd.to_datetime(data['timestamp'], unit='s')

        file_path = os.path.join(directory, 'asset_details.csv')
        self.data_details = pd.read_csv(file_path)

        self.data = pd.merge(data,
                        self.data_details,
                        on="Asset_ID",
                        how='left')

    def filterDatasets(self):

        all_coins = self.data.Asset_Name.tolist()

        df_tmp1 = pd.DataFrame()
        df_tmp2 = pd.DataFrame()

        for k in range(len(self.coins)):

            if self.coins[k] not in all_coins:
                print(self.coins[k] + ' ist nicht zulässig bzw. existiert nicht!')
                sys.exit(400)

            df_tmp1 = df_tmp1.append(self.data[self.data.Asset_Name == self.coins[k]])
            df_tmp2 = df_tmp2.append(self.data_details[self.data_details.Asset_Name == self.coins[k]])

        self.data = df_tmp1.sort_values('timestamp')
        self.data_details = df_tmp2.sort_values('Asset_ID')
        self.data.reset_index(drop = True, inplace = True)
        self.data_details.reset_index(drop = True, inplace = True)

    #Help function to create subplots
    def createPositionIndex(self):

        return range(1, len(self.data_details.Asset_ID) + 1)

    #Help function to create subplots
    def computeRows(self):

        tot = len(self.data_details.Asset_ID)
        rows = tot // 2
        rows += tot % 2

        return rows

    #Help function to create subplots
    def createFigure(self):

        fig = plt.figure(1)
        fig.set_figheight(20)
        fig.set_figwidth(20)

        return fig

    def createSubplotsReturnOverTime(self):

        rows = self.computeRows()
        position = self.createPositionIndex()
        fig = self.createFigure()

        for k in range(len(self.data_details.Asset_ID)):

            tmp_df = self.data[self.data.Asset_ID == self.data_details.Asset_ID[k]]
            ax = fig.add_subplot(rows, 2, position[k])
            ax.plot(tmp_df.Time, tmp_df.Target)
            ax.set_xlabel('Time', fontsize = 15)
            ax.set_ylabel('Target / Return', fontsize = 15)
            ax.set_title(self.data_details.Asset_Name[k], fontsize = 20)

        plt.show()
        del tmp_df

    def createSubplotsReturnDistribution(self):

        rows = self.computeRows()
        position = self.createPositionIndex()
        fig = self.createFigure()

        for k in range(len(self.data_details.Asset_ID)):
            tmp_df = self.data[self.data.Asset_ID == self.data_details.Asset_ID[k]]
            ax = fig.add_subplot(rows, 2, position[k])
            ax.hist(tmp_df.Target, bins=50)
            ax.set_xlim(-0.05, 0.05)
            ax.set_xlabel('Target / Return', fontsize = 15)
            ax.set_ylabel('Frequency', fontsize = 15)
            ax.set_title(self.data_details.Asset_Name[k], fontsize = 20)

        plt.show()
        del tmp_df

## ---------------------------- MAIN ---------------------------- ##
#Define coins that should be analysed / predicted
coins = ['Ethereum', 'Dogecoin']

#Create an instance of the model
model1 = AdvancedModel(coins)
print(model1.data.head())
print(model1.data_details.head())

#Filter data sets with all coins by only the coins that should be analysed / predicted
model1.filterDatasets()
print(model1.data.head())
print(model1.data_details.head())

#(1) Descriptive Analysis
#Create subplots: target variable (proxy for return) over time
model1.createSubplotsReturnOverTime()
model1.createSubplotsReturnDistribution()

''''' OLD CODE:
#(1) Step: Perform descriptive analysis
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

#(2) Step: Conduct feature engineering

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

#From now on: only one coin
import ta

#Take the last two plots as training period respectively test period
start_train, end_train = datetime.timestamp(timesplits[len(timesplits)-3]), \
                         datetime.timestamp(timesplits[len(timesplits)-2])
start_test, end_test = datetime.timestamp(timesplits[len(timesplits)-2]), \
                       datetime.timestamp(timesplits[len(timesplits)-1])

train_data, test_data = coin_tmp.loc[start_train:end_train], coin_tmp.loc[start_test:end_test][3600:]

upper_shadow = lambda asset: asset.High - np.maximum(asset.Close,asset.Open)
lower_shadow = lambda asset: np.minimum(asset.Close,asset.Open)- asset.Low

train_data['close_1'] = train_data.Close.diff()
train_data['close_15'] = train_data.Close.diff(15)
train_data['close_60'] = train_data.Close.diff(60)

train_data['count_1'] = train_data.Count.diff()
train_data['count_15'] = train_data.Count.diff(15)
train_data['count_60'] = train_data.Count.diff(60)

train_data['volume_1'] = train_data.Volume.diff()
train_data['volume_15'] = train_data.Volume.diff(15)
train_data['volume_60'] = train_data.Volume.diff(60)

train_data['upper_shadow'] = upper_shadow(train_data)
train_data['lower_shadow'] = lower_shadow(train_data)

train_data = ta.add_all_ta_features(train_data,
                                       open = 'Open',
                                       high = 'High',
                                       low = 'Low',
                                       close = 'Close',
                                       volume = 'Volume',
                                       fillna = False)

test_data['close_1'] = test_data.Close.diff()
test_data['close_15'] = test_data.Close.diff(15)
test_data['close_60'] = test_data.Close.diff(60)

test_data['count_1'] = test_data.Count.diff()
test_data['count_15'] = test_data.Count.diff(15)
test_data['count_60'] = test_data.Count.diff(60)

test_data['volume_1'] = test_data.Volume.diff()
test_data['volume_15'] = test_data.Volume.diff(15)
test_data['volume_60'] = test_data.Volume.diff(60)

test_data['upper_shadow'] = upper_shadow(test_data)
test_data['lower_shadow'] = lower_shadow(test_data)

test_data = ta.add_all_ta_features(test_data,
                                       open = 'Open',
                                       high = 'High',
                                       low = 'Low',
                                       close = 'Close',
                                       volume = 'Volume',
                                       fillna = False)

#Delete variables with more than 100 missing values except the target variable has more than 100 missing values too
if train_data['Target'].isnull().sum() > 100:
    train_data = train_data.drop(train_data.columns[train_data.isnull().sum() > 100].drop('Target'), axis = 1)
    test_data = test_data.drop(test_data.columns[test_data.isnull().sum() > 100].drop('Target'), axis = 1)
else:
    train_data = train_data.drop(train_data.columns[train_data.isnull().sum() > 100], axis = 1)
    test_data = test_data.drop(test_data.columns[test_data.isnull().sum() > 100], axis = 1)

#Rank the feature variables according to the correlation with the target variable
find_corr_features = train_data.drop(['Asset_ID', 'Time', 'Weight'], axis = 1).corr(method = 'spearman')['Target'].\
    abs().sort_values(ascending = False)
print(find_corr_features[0:21])

#Create heat map: visualize correlations among feature variables and the target variable
#Delete rows with missing values
train_data.dropna(inplace = True)
test_data.dropna(inplace = True)

#Filter the 20 feature variables that correlate most with the target variable
top_20_features = list(find_corr_features[:21].index)

fig, axs = plt.subplots(1, 2, figsize = (20, 10))
sns.heatmap(train_data[top_20_features].corr(method = 'spearman').abs(), ax = axs[0])
sns.heatmap(test_data[top_20_features].corr(method = 'spearman').abs(), ax = axs[1])
plt.show()

#Open To Do's
#TODO: Delete feature variables that correlate more than 0.9 with another feature varable (only vor training period)
#(TODO: Show the correlation between the feature variables and the target variable in a scatterplot in addition to the heat map)
#TODO: Stationary test for all the remaining feature variables + (graphics (development over the time) for training as well as the test period)

#(3) Step: Train and test model with suitable neuronal network

#Scale the data
from sklearn.preprocessing import MinMaxScaler

X_scaler = MinMaxScaler(feature_range = (0, 1))
X_train = train_data[top_20_features].drop(['Target'], axis = 1)
X_test = test_data[top_20_features].drop(['Target'], axis = 1)

y_train = train_data['Target'].values
y_test = test_data['Target'].values

X_train_ = X_scaler.fit_transform(X_train)
X_test_ = X_scaler.transform(X_test)

#Generate neural network (TODO: Standard Recurrent Neural Network)
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape = (X_train_.shape[1])),
    tf.keras.layers.Dense(20, activation = 'selu'),
    tf.keras.layers.Dropout(0.50),
    tf.keras.layers.Dense(10, activation = 'selu'),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(1)
])

#Show loss function for train period as well as test period
model.compile(loss = 'mean_absolute_error', optimizer = 'adam')
history = model.fit(X_train_, y_train, epochs = 5, validation_data = (X_test_, y_test))
plt.plot(history.history['loss'], label = 'training')
plt.plot(history.history['val_loss'], label = 'test')
plt.show()

#Show correlation between predictions and target realizations for training period respectively test period
fig, axs = plt.subplots(1, 2, figsize = (12,8))

axs[0].scatter(model.predict(X_train_).flatten(), y_train)
axs[1].scatter(model.predict(X_test_).flatten(), y_test)
plt.show()

print(np.corrcoef(model.predict(X_train_).flatten(), y_train)[0, 1])
print(np.corrcoef(model.predict(X_test_).flatten(), y_test)[0, 1])

#Show the feature importance of each feature variable in relation to the used model
X = tf.Variable(X_train_)
y = tf.Variable(y_train)
with tf.GradientTape() as tape:
    tape.watch(X)
    pred = model(X)
grad = tf.abs(tape.gradient(pred,X))
grad = tf.reduce_mean(grad,axis=0)
feature_importance = grad.numpy() / grad.numpy().sum()


plt.figure(figsize=(10,20))
plt.barh(X_train.columns[np.argsort(feature_importance)], np.sort(feature_importance))
plt.title('Feature importance')
plt.show()
'''''