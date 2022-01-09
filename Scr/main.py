"""""
import os
import numpy as np
import pandas as pd
import sys as sys
import matplotlib.pyplot as plt
import itertools as itt
import seaborn as sb
from datetime import datetime
from datetime import date
from dateutil.relativedelta import relativedelta
import ta
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

class AdvancedModel():

    def __init__(self, coins):

        self.coins = coins
        self.data = pd.DataFrame()
        self.data_details = pd.DataFrame()
        self.data_training = pd.DataFrame()
        self.data_test = pd.DataFrame()
        self.top_20_features = {}

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

        all_coins = self.data_details.Asset_Name.tolist()

        df_tmp1 = pd.DataFrame()
        df_tmp2 = pd.DataFrame()

        for k in range(len(self.coins)):

            if self.coins[k] not in all_coins:
                print('[' + self.coins[k] + '] is not allowed or does not exit!')
                sys.exit(400)

            df_tmp1 = df_tmp1.append(self.data[self.data.Asset_Name == self.coins[k]])
            df_tmp2 = df_tmp2.append(self.data_details[self.data_details.Asset_Name == self.coins[k]])

        #This step is only necessary for createSubplotsSevenDayCorrelation() & createHeatMap() function
        self.data_btc = self.data[self.data.Asset_Name == 'Bitcoin']
        self.data_btc_details = self.data_details[self.data_details.Asset_Name == 'Bitcoin']
        self.data_btc.sort_values('timestamp')

        self.data = df_tmp1.sort_values('timestamp')
        self.data_details = df_tmp2.sort_values('Asset_ID')
        self.data.reset_index(drop=True, inplace=True)
        self.data_details.reset_index(drop=True, inplace=True)

    def createSubplotsClosingPriceDevelopment(self):

        rows = self.computeRows()
        position = self.createPositionIndex()
        fig = self.createFigure()

        for k in range(len(self.data_details.Asset_ID)):

            tmp_df = self.data[self.data.Asset_ID == self.data_details.Asset_ID[k]]
            ax = fig.add_subplot(rows, 2, position[k])
            ax.plot(tmp_df.Time, tmp_df.Close)
            ax.set_xlabel('Time', fontsize=15)
            ax.set_ylabel('Closing Price', fontsize=15)
            ax.set_title(self.data_details.Asset_Name[k], fontsize=20)

        plt.show()
        del tmp_df

    def createSubplotsClosingPriceDevelopmentSplitUp(self):

        for k in range(len(self.data_details.Asset_ID)):

            #Set and sort the index of the new df
            tmp_df = self.data[self.data.Asset_ID == self.data_details.Asset_ID[k]]
            tmp_df.set_index('timestamp', inplace=True)
            tmp_df = tmp_df.reindex(range(tmp_df.index[0], tmp_df.index[-1] + 60, 60), method='pad')
            tmp_df.sort_index(inplace=True)

            #Calculate number of month in the new df
            starting_date = tmp_df.Time.iloc[0]
            ending_date = date(2021, 9, 21)
            numb_month = (ending_date.year - starting_date.year) * 12 + (ending_date.month - starting_date.month)

            #Calculate six month timesplits
            timesplits = [starting_date + i * relativedelta(months=6) for i in range((numb_month // 6) + 1)] + \
                         [tmp_df.Time.iloc[-1]]

            #Take the last two six-month-plots as training period respectively test period
            start_train, end_train = datetime.timestamp(timesplits[len(timesplits) - 4]), \
                                     datetime.timestamp(timesplits[len(timesplits) - 3])
            start_test, end_test = datetime.timestamp(timesplits[len(timesplits) - 3]), \
                                   datetime.timestamp(timesplits[len(timesplits) - 2])

            #The training and test dataset will generate for all analysed coins
            self.data_training = self.data_training.append(tmp_df.loc[start_train:end_train])
            self.data_test = self.data_test.append(tmp_df.loc[start_test:end_test][3600:])

            tot = len(timesplits) - 1
            cols = 2

            rows = tot // cols
            rows += tot % cols

            position = range(1, tot + 1)

            fig = self.createFigure()
            fig.suptitle(self.data_details.Asset_Name[k], fontsize=20)

            #Add every single subplot to the figure with a for loop
            for j in range(tot):
                tmp_df2 = tmp_df.loc[datetime.timestamp(timesplits[j]):datetime.timestamp(timesplits[j + 1])]
                ax = fig.add_subplot(rows, cols, position[j])
                ax.plot(tmp_df2.Time, tmp_df2.Close)
                ax.set_xlabel('Time', fontsize=15)
                ax.set_ylabel('Closing Price', fontsize=15)

            plt.show()
            del tmp_df2

        del tmp_df

        self.data_training.sort_index(inplace=True)
        self.data_test.sort_index(inplace=True)

    def caluclateTechnicalIndicators(self):

        upper_shadow = lambda asset: asset.High - np.maximum(asset.Close, asset.Open)
        lower_shadow = lambda asset: np.minimum(asset.Close, asset.Open) - asset.Low

        for k in range(len(self.data_details.Asset_ID)):

            tmp_df_tr = self.data_training[self.data_training.Asset_ID == self.data_details.Asset_ID[k]]
            tmp_df_te = self.data_test[self.data_test.Asset_ID == self.data_details.Asset_ID[k]]

            tmp_df_tr['close_1'] = tmp_df_tr.Close.diff()
            tmp_df_tr['close_15'] = tmp_df_tr.Close.diff(15)
            tmp_df_tr['close_60'] = tmp_df_tr.Close.diff(60)

            tmp_df_tr['count_1'] = tmp_df_tr.Count.diff()
            tmp_df_tr['count_15'] = tmp_df_tr.Count.diff(15)
            tmp_df_tr['count_60'] = tmp_df_tr.Count.diff(60)

            tmp_df_tr['volume_1'] = tmp_df_tr.Volume.diff()
            tmp_df_tr['volume_15'] = tmp_df_tr.Volume.diff(15)
            tmp_df_tr['volume_60'] = tmp_df_tr.Volume.diff(60)

            tmp_df_tr['upper_shadow'] = upper_shadow(tmp_df_tr)
            tmp_df_tr['lower_shadow'] = lower_shadow(tmp_df_tr)

            tmp_df_tr = ta.add_all_ta_features(tmp_df_tr,
                                                open='Open',
                                                high='High',
                                                low='Low',
                                                close='Close',
                                                volume='Volume',
                                                fillna=False)

            tmp_df_te['close_1'] = tmp_df_te.Close.diff()
            tmp_df_te['close_15'] = tmp_df_te.Close.diff(15)
            tmp_df_te['close_60'] = tmp_df_te.Close.diff(60)

            tmp_df_te['count_1'] = tmp_df_te.Count.diff()
            tmp_df_te['count_15'] = tmp_df_te.Count.diff(15)
            tmp_df_te['count_60'] = tmp_df_te.Count.diff(60)

            tmp_df_te['volume_1'] = tmp_df_te.Volume.diff()
            tmp_df_te['volume_15'] = tmp_df_te.Volume.diff(15)
            tmp_df_te['volume_60'] = tmp_df_te.Volume.diff(60)

            tmp_df_te['upper_shadow'] = upper_shadow(tmp_df_te)
            tmp_df_te['lower_shadow'] = lower_shadow(tmp_df_te)

            tmp_df_te = ta.add_all_ta_features(tmp_df_te,
                                                open='Open',
                                                high='High',
                                                low='Low',
                                                close='Close',
                                                volume='Volume',
                                                fillna=False)

            #Delete variables with more than 100 missing values (except the target variable)
            if tmp_df_tr['Target'].isnull().sum() > 100:
                tmp_df_tr = tmp_df_tr.drop(tmp_df_tr.columns[tmp_df_tr.isnull().sum() > 100].drop('Target'), axis=1)
                tmp_df_te = tmp_df_te.drop(tmp_df_te.columns[tmp_df_te.isnull().sum() > 100].drop('Target'), axis=1)
            else:
                tmp_df_tr = tmp_df_tr.drop(tmp_df_tr.columns[tmp_df_tr.isnull().sum() > 100], axis=1)
                tmp_df_te = tmp_df_te.drop(tmp_df_te.columns[tmp_df_te.isnull().sum() > 100], axis=1)

            #Update training and test dataframe with calculated technical indicators
            self.data_training = self.data_training[self.data_training.Asset_ID != self.data_details.Asset_ID[k]]
            self.data_training = self.data_training.append(tmp_df_tr)
            self.data_test = self.data_test[self.data_test.Asset_ID != self.data_details.Asset_ID[k]]
            self.data_test = self.data_test.append(tmp_df_te)

        del tmp_df_tr
        del tmp_df_te

        self.data_training.sort_index(inplace=True)
        self.data_test.sort_index(inplace=True)

    def rankFeatureVariables(self):

        for k in range(len(self.data_details.Asset_ID)):

            tmp_df_tr = self.data_training[self.data_training.Asset_ID == self.data_details.Asset_ID[k]]
            tmp_df_te = self.data_test[self.data_test.Asset_ID == self.data_details.Asset_ID[k]]

            find_corr_features = tmp_df_tr.drop(['Asset_ID', 'Time', 'Weight'], axis=1).corr(method='spearman')['Target'].\
                abs().sort_values(ascending=False)

            print('20 feature variables of [' +
                  self.data_details.Asset_Name[k] +
                  '], that correlate highest with the target variable: \n' +
                  str(find_corr_features[1:21]))

            #Insert into dictonary the top 20 features (=values) of each analysed coin (=key)
            self.top_20_features[self.data_details.Asset_ID[k]] = list(find_corr_features[:21].index)

        del tmp_df_tr
        del tmp_df_te

    def createSubplotsHeatMap(self):

        for k in range(len(self.data_details.Asset_ID)):

            tmp_df_tr = self.data_training[self.data_training.Asset_ID == self.data_details.Asset_ID[k]]
            tmp_df_te = self.data_test[self.data_test.Asset_ID == self.data_details.Asset_ID[k]]

            #Delete rows with missing values
            tmp_df_tr.dropna(inplace=True)
            tmp_df_te.dropna(inplace=True)

            fig, axs = plt.subplots(1, 2, figsize=(20, 10))
            fig.suptitle(self.data_details.Asset_Name[k], fontsize=20)
            axs[0].set_title('Training Data')
            axs[1].set_title('Test Data')
            sb.heatmap(tmp_df_tr[self.top_20_features[self.data_details.Asset_ID[k]]].corr(method='spearman').abs(),
                       ax=axs[0])
            sb.heatmap(tmp_df_te[self.top_20_features[self.data_details.Asset_ID[k]]].corr(method='spearman').abs(),
                       ax=axs[1])
            plt.show()

        del tmp_df_tr
        del tmp_df_te

    def applyNeuralNetwork(self):

        for k in range(len(self.data_details.Asset_ID)):

            #Scale the data
            tmp_df_tr = self.data_training[self.data_training.Asset_ID == self.data_details.Asset_ID[k]]
            tmp_df_te = self.data_test[self.data_test.Asset_ID == self.data_details.Asset_ID[k]]

            #Delete rows with missing values
            tmp_df_tr.dropna(inplace=True)
            tmp_df_te.dropna(inplace=True)

            x_scaler = MinMaxScaler(feature_range=(0, 1))
            x_train = tmp_df_tr[self.top_20_features[self.data_details.Asset_ID[k]]].drop(['Target'], axis=1)
            x_test = tmp_df_te[self.top_20_features[self.data_details.Asset_ID[k]]].drop(['Target'], axis=1)

            y_train = tmp_df_tr['Target'].values
            y_test = tmp_df_te['Target'].values

            x_train_ = x_scaler.fit_transform(x_train)
            x_test_ = x_scaler.transform(x_test)

            #Generate different neural networks depending on the considered coin
            if self.data_details.Asset_Name[k] == 'Dogecoin':

                model = tf.keras.Sequential([
                    tf.keras.layers.InputLayer(input_shape=(x_train_.shape[1])),
                    tf.keras.layers.Dense(20, activation='selu'),
                    tf.keras.layers.Dropout(0.50),
                    tf.keras.layers.Dense(10, activation='selu'),
                    tf.keras.layers.Dropout(0.25),
                    tf.keras.layers.Dense(1)
            ])

            elif self.data_details.Asset_Name[k] == 'Ethereum':

                model = tf.keras.Sequential([
                    tf.keras.layers.InputLayer(input_shape=(x_train_.shape[1])),
                    tf.keras.layers.Dense(20, activation='selu'),
                    tf.keras.layers.Dropout(0.50),
                    tf.keras.layers.Dense(10, activation='selu'),
                    tf.keras.layers.Dropout(0.25),
                    tf.keras.layers.Dense(1)
                ])

            else:

                print('Unfortunately, no neural network is available for [' + self.data_details[k].Asset_Name + '].')
                sys.exit(400)

            #Show loss function for train period as well as test period
            model.compile(loss='mean_absolute_error', optimizer='adam')
            history = model.fit(x_train_, y_train, epochs=5, validation_data=(x_test_, y_test))
            plt.plot(history.history['loss'], label='training')
            plt.plot(history.history['val_loss'], label='test')
            plt.legend()
            plt.show()

            #Show correlation between predictions and target realizations for training period respectively test period

            #visually
            fig, axs = plt.subplots(1, 2, figsize=(12, 8))
            axs[0].set_title('Training Data')
            axs[1].set_title('Test Data')
            axs[0].scatter(model.predict(x_train_).flatten(), y_train)
            axs[1].scatter(model.predict(x_test_).flatten(), y_test)
            plt.show()

            #quantitatively
            print('Correlation between predictions and target realizations of [' + self.data_details.Asset_Name[k] +
                  '] in the training period: ' +
                  str(np.corrcoef(model.predict(x_train_).flatten(), y_train)[0, 1]))
            print('Correlation between predictions and target realizations of [' + self.data_details.Asset_Name[k] +
                  '] in the testing period: ' +
                  str(np.corrcoef(model.predict(x_test_).flatten(), y_test)[0, 1]))

            #Show the feature importance of each feature variable in relation to the used model
            x = tf.Variable(x_train_)
            y = tf.Variable(y_train)
            with tf.GradientTape() as tape:
                tape.watch(x)
                pred = model(x)
            grad = tf.abs(tape.gradient(pred, x))
            grad = tf.reduce_mean(grad, axis=0)
            feature_importance = grad.numpy() / grad.numpy().sum()

            plt.figure(figsize=(10, 20))
            plt.barh(x_train.columns[np.argsort(feature_importance)], np.sort(feature_importance))
            plt.title('Importance of the Features of [' + self.data_details.Asset_Name[k] + ']')
            plt.show()


## ---------------------------- MAIN ---------------------------- ##

#(1) Configuration

#Define coins that should be analysed / predicted
coins = ['Ethereum']

#Create an instance of the model
model1 = AdvancedModel(coins)
print(model1.data.head())
print(model1.data_details.head())

#Filter data sets with all coins by only the coins that should be analysed / predicted
model1.filterDatasets()
print(model1.data.head())
print(model1.data_details.head())


#(3) Feature Engineering

#Create subplots: closing price development
model1.createSubplotsClosingPriceDevelopment()

#Create subplots (for each coin): split up closing price development to determine a suitable training and test period;
#In Addition to that: the training and test dataset will generate for all analysed coins
model1.createSubplotsClosingPriceDevelopmentSplitUp()

#Calucation of feature variables
#a) Technical Indicators
model1.caluclateTechnicalIndicators()
#b) tbd

#Rank feature variables according to the correlation with the target variable & print it
model1.rankFeatureVariables()

#Create subplots (for each coin): Correlation of the top 20 feature variables with the target variable
#(for training period and test period)
model1.createSubplotsHeatMap()

#(4) Train and test model with suitable neuronal network
#(scaling and result graphics included)
mode
"""""

