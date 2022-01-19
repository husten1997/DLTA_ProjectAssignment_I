import pandas as pd
from datetime import datetime
from datetime import date
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import numpy as np
import ta
import seaborn as sb
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import keras_tuner as kt
from statsmodels.tsa.stattools import adfuller

class Advanced_Model():

    #Initialize class variables
    coin_id = 0
    coin_name = ''

    all_data = pd.DataFrame()
    all_data_training = pd.DataFrame()
    all_data_test = pd.DataFrame()
    all_data_eval = pd.DataFrame()

    data = pd.DataFrame()
    data_training = pd.DataFrame()
    data_test = pd.DataFrame()
    data_eval = pd.DataFrame()

    featureSet_training = pd.DataFrame()
    featureSet_test = pd.DataFrame()
    featureSet_eval = pd.DataFrame()
    #Contains top features as well as target variable
    top_features = []

    x_train_ = []
    y_train = []
    y_test = []
    y_train_hat = []
    y_test_hat = []

    def __init__(self, coin_id, all_data, all_data_details):

        self.all_data = all_data

        self.coin_id = coin_id
        self.data = all_data[all_data.Asset_ID == coin_id]

        self.all_data_details = all_data_details[all_data_details.Asset_ID == coin_id]
        self.all_data_details.reset_index(drop=True, inplace=True)
        self.coin_name = self.all_data_details.Asset_Name[0]

        self.setupData()

    def setupData(self):

        self.all_data.set_index('timestamp', inplace=True)

        self.data.set_index('timestamp', inplace=True)
        self.data = self.data.reindex(range(self.data.index[0], self.data.index[-1] + 60, 60), method='pad')
        self.data.sort_index(inplace=True)

        training_start, test_start, eval_start = self.getPeriods()

        #This step ist necessary for the calculation of the technical market indicators
        self.all_data_training = self.all_data[(self.all_data.index >= training_start) & (self.all_data.index <= test_start)]
        self.all_data_test = self.all_data[(self.all_data.index >= test_start) & (self.all_data.index <= eval_start)]
        #self.all_data_eval = self.all_data[(self.all_data.index >= eval_start)]

        #This step is necessary for the calculation of all further feature variables
        self.data_training = self.data[(self.data.index >= training_start) & (self.data.index <= test_start)]
        self.data_test = self.data[(self.data.index >= test_start) & (self.data.index <= eval_start)]
        #self.data_eval = self.data[(self.data.index >= eval_start)]

        self.mergeFeatureSets()
        self.stationarity_transformation()
        self.setTopFeatureVariables()

    #Help function to get test, training & evaluation period (see setupData() function above)
    def getPeriods(self):

        starting_date = self.data.Time.iloc[0]
        ending_date = date(2021, 9, 21)
        numb_month = (ending_date.year - starting_date.year) * 12 + (ending_date.month - starting_date.month)

        #Calculate six month timesplits
        timesplits = [starting_date + i * relativedelta(months=6) for i in range((numb_month // 6) + 1)] + \
                     [self.data.Time.iloc[-1]]

        #Take the last two six-month-plots as training period respectively test period & the last plot as eval period
        start_train = datetime.timestamp(timesplits[len(timesplits) - 4])
        start_test = datetime.timestamp(timesplits[len(timesplits) - 3])
        start_eval = datetime.timestamp(timesplits[len(timesplits) - 2])

        #Additional: create subplots (closing price development split up)
        tot = len(timesplits) - 1
        cols = 2

        rows = tot // cols
        rows += tot % cols

        position = range(1, tot + 1)

        fig = self.createFigure()
        fig.suptitle(self.coin_name, fontsize=20)

        for j in range(tot):
            tmp_df = self.data.loc[datetime.timestamp(timesplits[j]):datetime.timestamp(timesplits[j + 1])]
            ax = fig.add_subplot(rows, cols, position[j])
            ax.plot(tmp_df.Time, tmp_df.Close)
            ax.set_xlabel('Time', fontsize=15)
            ax.set_ylabel('Closing Price', fontsize=15)

        plt.show()
        del tmp_df

        #return start_train, start_test, start_eval

        #Jan 01 2020 00:00:00, Jan 02 2020 00:00:00 & Jan 03 2020 00:00:00
        return 1577833200, 1577919600, 1578006000

        #Jan 01 2020 00:00:00, Feb 01 2020 00:00:00, Mar 01 2020 00:00:00
        #return 1577833200, 1580511600, 1583017200

    def mergeFeatureSets(self):

        #Get various feature sets
        market_movements_autoencoder_train, market_movements_autoencoder_test = self.calculateTechnicalMarketIndicators()
        tech_indicators_training, tech_indicators_test, tech_indicators_eval = self.calculateTechnicalIndicators()
        #TODO: calculate basic variables

        #Merge different feature sets to one final feature set for each period
        self.featureSet_training = tech_indicators_training.join(market_movements_autoencoder_train, how='inner')
        self.featureSet_test = tech_indicators_test.join(market_movements_autoencoder_test, how='inner')
        #self.featureSet_eval = tech_indicators_eval

        self.featureSet_training.dropna(inplace=True)
        self.featureSet_test.dropna(inplace=True)

    def stationarity_transformation(self):

        for variable in self.featureSet_training.columns:
            timeseries = self.featureSet_training[variable]
            result = adfuller(timeseries)
            p_value = result[1]
            if p_value < 0.05:
                self.featureSet_training[variable] = self.featureSet_training[variable].diff()

        for variable in self.featureSet_test.columns:
            timeseries = self.featureSet_test[variable]
            result = adfuller(timeseries)
            p_value = result[1]
            if p_value < 0.05:
                self.featureSet_test[variable] = self.featureSet_test[variable].diff()
        """""
        for variable in self.featureSet_eval.columns:
            timeseries = self.featureSet_eval[variable]
            result = adfuller(timeseries)
            p_value = result[1]
            if p_value < 0.05:
                self.featureSet_eval[variable] = self.featureSet_eval[variable].pct_change()
        """""

        self.featureSet_training.dropna(inplace=True)
        self.featureSet_test.dropna(inplace=True)

    def setTopFeatureVariables(self):

        tmp_df_training = self.mergeFinalFeatureSetAndTargetVariable()[0]

        find_corr_features = tmp_df_training.corr(method='spearman')['Target'].abs().sort_values(ascending=False)
        #Delete first row (correlation of the target variable with itself) for the print out
        find_corr_features_print = find_corr_features.drop(index=find_corr_features.index[0])


        print('Correlation of the Features of [' +
              self.coin_name +
              '] with the target variable: \n' + str((find_corr_features_print.loc[find_corr_features > 0.15])))

        self.top_features = list(find_corr_features.loc[find_corr_features > 0.15].index)

    """"" Not necessary due to the spearman correlation
    def nonlin_transform(self):

        indices = []

        for variable in self.featureSet_training.columns:
            for poly in range(2,4):
                index = f"{variable}_{poly}"
                indices.append(index)

        for variable in self.featureSet_training.columns:
            for poly in range(2,4):
                #nonlintransformation of training data
                self.featureSet_training[f"{variable}_{poly}"] = self.featureSet_training[f"{variable}"].values**poly

        for variable in self.featureSet_test.columns:
            for poly in range(2,4):
                index = f"{variable}_{poly}"
                indices.append(index)

        for variable in self.featureSet_test.columns:
            for poly in range(2,4):
                #nonlintransformation of test data
                self.featureSet_test[f"{variable}_{poly}"] = self.featureSet_test[f"{variable}"].values**poly

        #TODO: non linear transformation for evalution data

    """""

    def applyModel(self, epochs, method = "Tuner"):

        tmp_df_training = self.mergeFinalFeatureSetAndTargetVariable()[0]
        tmp_df_test = self.mergeFinalFeatureSetAndTargetVariable()[1]

        x_train = tmp_df_training[self.top_features].drop(['Target'], axis=1)
        x_test = tmp_df_test[self.top_features].drop(['Target'], axis=1)

        self.y_train = tmp_df_training['Target'].values
        self.y_test = tmp_df_test['Target'].values

        #Variable x_train_ is also used in method buildAdvModel_KerasTuner(), therefore we have to declare it as class variable
        self.x_train_ = self.scaling(x_train)
        self.x_test_ = self.scaling(x_test)

        if method == "FNN":
            config = {
                "RNN_Lookback": 15,
                "GRU": True,
                #GRU
                "GRU_L1_units": 120,
                "GRU_L1_actfun": "tanh",
                "GRU_L1_dropoutBool": True,
                "GRU_L1_dropoutUnit": 0.25,
                "GRU_L2_units": 60,
                "GRU_L2_actfun": "tanh",
                "GRU_L2_dropoutBool": True,
                "GRU_L2_dropoutUnit": 0.25,
                #LSTM
                "LSTM_L1_units": 120,
                "LSTM_L1_actfun": "tanh",
                "LSTM_L1_dropoutBool": True,
                "LSTM_L1_dropoutUnit": 0.25,
                "LSTM_L2_units": 60,
                "LSTM_L2_actfun": "tanh",
                "LSTM_L2_dropoutBool": True,
                "LSTM_L2_dropoutUnit": 0.25,
                "lr": 1e-2}

            self.adv_model = self.buildAdvModel(config)

            history = self.adv_model.fit(self.x_train_, self.y_train, epochs=epochs, validation_data=(self.x_test_, self.y_test), batch_size = 1024)
            plt.plot(history.history['loss'], label='training')
            plt.plot(history.history['val_loss'], label='test')
            plt.legend()
            plt.show()

        elif method == "Tuner": # TODO: Tuner currently broken due to missing input shape
            self.tuner = kt.RandomSearch(self.buildAdvModel_KerasTuner, objective='val_loss', max_trials=10)

            self.tuner.search(self.x_train_, self.y_train, epochs = 10, validation_data=(self.x_test_, self.y_test), batch_size = 1024)
            self.adv_model = self.tuner.get_best_models()[0]

            history = self.adv_model.fit(self.x_train_, self.y_train, epochs=epochs, validation_data=(self.x_test_, self.y_test), batch_size = 1024)
            plt.plot(history.history['loss'], label='training')
            plt.plot(history.history['val_loss'], label='test')
            plt.legend()
            plt.show()

        self.y_train_hat = self.adv_model.predict(self.x_train_).flatten()
        self.y_test_hat = self.adv_model.predict(self.x_test_).flatten()

        #Show Feature Importance after estimation of the model
        self.showFeatureImportance(x_train, self.y_train)

    def buildAdvModel(self, config):
        model = tf.keras.Sequential()

        input_shape = self.x_train_.shape[1]  # Would be x_train_.shape[1], but we dont have parameters with the Keras_tuner (or do we?)
        model.add(tf.keras.layers.InputLayer(input_shape=(input_shape)))
        
        model.add(tf.keras.layers.RepeatVector(config["RNN_Lookback"]))
        
        if config['GRU']:
            model.add(tf.keras.layers.GRU(config['GRU_L1_units'], return_sequences=True, activation=config["GRU_L1_actfun"]))
            if config['GRU_L1_dropoutBool']:
                model.add(tf.keras.layers.Dropout(config['GRU_L1_dropoutUnit']))
            model.add(tf.keras.layers.GRU(config['GRU_L2_units'], return_sequences=False, activation=config["GRU_L2_actfun"]))
            if config['GRU_L2_dropoutBool']:
                model.add(tf.keras.layers.Dropout(config['GRU_L2_dropoutUnit']))
        else:
            model.add(
                tf.keras.layers.LSTM(config['LSTM_L1_units'], return_sequences=True, activation=config["LSTM_L1_actfun"]))
            if config['LSTM_L1_dropoutBool']:
                model.add(tf.keras.layers.Dropout(config['LSTM_L1_dropoutUnit']))
            model.add(
                tf.keras.layers.LSTM(config['LSTM_L2_units'], return_sequences=False, activation=config["LSTM_L2_actfun"]))
            if config['LSTM_L2_dropoutBool']:
                model.add(tf.keras.layers.Dropout(config['LSTM_L2_dropoutUnit']))

        model.add(tf.keras.layers.Dense(1))

        learning_rate = config["lr"]

        model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))

        model.summary()

        return model

    def buildAdvModel_KerasTuner(self, hp):
        model = tf.keras.Sequential()

        input_shape = self.x_train_.shape[1]  # Would be x_train_.shape[1], but we dont have parameters with the Keras_tuner (or do we?)
        model.add(tf.keras.layers.InputLayer(input_shape=(input_shape)))

        model.add(tf.keras.layers.RepeatVector(hp.Choice('RNN_Lookback', [15, 30, 60, 120, 240])))

        if hp.Boolean('GRU'):
            model.add(
                tf.keras.layers.GRU(hp.Choice('GRU_L1_units', [60, 120, 240]), return_sequences=True, activation=hp.Choice("GRU_L1_actfun", ["relu", "tanh", "selu"])))
            if True:#hp.Boolean('GRU_L1_dropoutBool'):
                model.add(tf.keras.layers.Dropout(hp.Choice('GRU_L1_dropoutUnit', [0.12, 0.25, 0.5])))
            model.add(
                tf.keras.layers.GRU(hp.Choice('GRU_L2_units', [60, 120, 240]), return_sequences=False, activation=hp.Choice("GRU_L2_actfun", ["relu", "tanh", "selu"])))
            if True: #hp.Boolean('GRU_L2_dropoutBool'):
                model.add(tf.keras.layers.Dropout(hp.Choice('GRU_L2_dropoutUnit', [0.12, 0.25, 0.5])))
        else:
            model.add(
                tf.keras.layers.LSTM(hp.Choice('LSTM_L1_units', [60, 120, 240]), return_sequences=True,
                                    activation=hp.Choice("LSTM_L1_actfun", ["relu", "tanh", "selu"])))
            if True: #hp.Boolean('LSTM_L1_dropoutBool'):
                model.add(tf.keras.layers.Dropout(hp.Choice('LSTM_L1_dropoutUnit', [0.12, 0.25, 0.5])))
            model.add(
                tf.keras.layers.LSTM(hp.Choice('LSTM_L2_units', [60, 120, 240]), return_sequences=False,
                                    activation=hp.Choice("LSTM_L2_actfun", ["relu", "tanh", "selu"])))
            if True: #hp.Boolean('LSTM_L2_dropoutBool'):
                model.add(tf.keras.layers.Dropout(hp.Choice('LSTM_L2_dropoutUnit', [0.12, 0.25, 0.5])))

        model.add(tf.keras.layers.Dense(1))

        learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")

        model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))

        model.summary()

        return model

    def showFeatureImportance(self, x_train, y_train):

        x = tf.Variable(self.x_train_)
        y = tf.Variable(y_train)
        with tf.GradientTape() as tape:
            tape.watch(x)
            pred = self.adv_model(x)
        grad = tf.abs(tape.gradient(pred, x))
        grad = tf.reduce_mean(grad, axis=0)
        feature_importance = grad.numpy() / grad.numpy().sum()

        plt.figure(figsize=(10, 20))
        plt.barh(x_train.columns[np.argsort(feature_importance)], np.sort(feature_importance))
        plt.title('Importance of the Features of [' + self.coin_name + ']')
        plt.show()

    def calculateTechnicalIndicators(self):

        tmp_df_training = self.data_training
        tmp_df_test = self.data_test
        tmp_df_eval = self.data_eval

        upper_shadow = lambda asset: asset.High - np.maximum(asset.Close, asset.Open)
        lower_shadow = lambda asset: np.minimum(asset.Close, asset.Open) - asset.Low

        tmp_df_training['close_1'] = tmp_df_training.Close.diff()
        tmp_df_training['close_15'] = tmp_df_training.Close.diff(15)
        tmp_df_training['close_60'] = tmp_df_training.Close.diff(60)

        tmp_df_training['count_1'] = tmp_df_training.Count.diff()
        tmp_df_training['count_15'] = tmp_df_training.Count.diff(15)
        tmp_df_training['count_60'] = tmp_df_training.Count.diff(60)

        tmp_df_training['volume_1'] = tmp_df_training.Volume.diff()
        tmp_df_training['volume_15'] = tmp_df_training.Volume.diff(15)
        tmp_df_training['volume_60'] = tmp_df_training.Volume.diff(60)

        tmp_df_training['upper_shadow'] = upper_shadow(tmp_df_training)
        tmp_df_training['lower_shadow'] = lower_shadow(tmp_df_training)

        tmp_df_training = ta.add_all_ta_features(tmp_df_training,
                                                 open='Open',
                                                 high='High',
                                                 low='Low',
                                                 close='Close',
                                                 volume='Volume',
                                                 fillna=False)

        tmp_df_test['close_1'] = tmp_df_test.Close.diff()
        tmp_df_test['close_15'] = tmp_df_test.Close.diff(15)
        tmp_df_test['close_60'] = tmp_df_test.Close.diff(60)

        tmp_df_test['count_1'] = tmp_df_test.Count.diff()
        tmp_df_test['count_15'] = tmp_df_test.Count.diff(15)
        tmp_df_test['count_60'] = tmp_df_test.Count.diff(60)

        tmp_df_test['volume_1'] = tmp_df_test.Volume.diff()
        tmp_df_test['volume_15'] = tmp_df_test.Volume.diff(15)
        tmp_df_test['volume_60'] = tmp_df_test.Volume.diff(60)

        tmp_df_test['upper_shadow'] = upper_shadow(tmp_df_test)
        tmp_df_test['lower_shadow'] = lower_shadow(tmp_df_test)

        tmp_df_test = ta.add_all_ta_features(tmp_df_test,
                                             open='Open',
                                             high='High',
                                             low='Low',
                                             close='Close',
                                             volume='Volume',
                                             fillna=False)
        """"" -> eval ist derzeit nicht relevant
        tmp_df_eval['close_1'] = tmp_df_eval.Close.diff()
        tmp_df_eval['close_15'] = tmp_df_eval.Close.diff(15)
        tmp_df_eval['close_60'] = tmp_df_eval.Close.diff(60)

        tmp_df_eval['count_1'] = tmp_df_eval.Count.diff()
        tmp_df_eval['count_15'] = tmp_df_eval.Count.diff(15)
        tmp_df_eval['count_60'] = tmp_df_eval.Count.diff(60)

        tmp_df_eval['volume_1'] = tmp_df_eval.Volume.diff()
        tmp_df_eval['volume_15'] = tmp_df_eval.Volume.diff(15)
        tmp_df_eval['volume_60'] = tmp_df_eval.Volume.diff(60)

        tmp_df_eval['upper_shadow'] = upper_shadow(tmp_df_eval)
        tmp_df_eval['lower_shadow'] = lower_shadow(tmp_df_eval)

        tmp_df_eval = ta.add_all_ta_features(tmp_df_eval,
                                           open='Open',
                                           high='High',
                                           low='Low',
                                           close='Close',
                                           volume='Volume',
                                           fillna=False)
        """""

        #Delete variables that are no technical indicators
        tmp_df_training = tmp_df_training.drop(
            ['Asset_ID', 'Count', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'VWAP', 'Target', 'Time',
             'Weight', 'Asset_Name'], axis=1)
        tmp_df_test = tmp_df_test.drop(
            ['Asset_ID', 'Count', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'VWAP', 'Target', 'Time',
             'Weight', 'Asset_Name'], axis=1)
        # tmp_df_eval.drop(['Asset_ID', 'Count', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'VWAP', 'Target', 'Time',
        # 'Weight', 'Asset_Name'], axis=1)

        #Delete variables with more than 100 missing values
        tmp_df_training = tmp_df_training.drop(tmp_df_training.columns[tmp_df_training.isnull().sum() > 100], axis=1)
        tmp_df_test = tmp_df_test.drop(tmp_df_test.columns[tmp_df_test.isnull().sum() > 100], axis=1)
        #tmp_df_eval = tmp_df_eval.drop(tmp_df_eval.columns[tmp_df_eval.isnull().sum() > 100], axis=1)

        #tmp_df_training.dropna(inplace=True)
        #tmp_df_test.dropna(inplace=True)

        return tmp_df_training, tmp_df_test, tmp_df_eval

    def calculateTechnicalMarketIndicators(self):

        #variable names
        variables = ["Close", "Open", "High", "Low", "Volume"]

        #create pivot table for training data
        data_pivot_train = self.all_data_training.pivot_table(index=self.all_data_training.index, columns = 'Asset_ID')
        #TODO: is > 1000 enough or is it too less restrictive?
        data_pivot_train = data_pivot_train.drop(data_pivot_train.columns[data_pivot_train.isnull().sum() > 1000], axis=1)

        #create pivot table for testing data
        data_pivot_test = self.all_data_test.pivot_table(index=self.all_data_test.index, columns = 'Asset_ID')
        #TODO: is > 1000 enough or is it too less restrictive?
        data_pivot_test = data_pivot_test.drop(data_pivot_test.columns[data_pivot_test.isnull().sum() > 1000], axis=1)

        X_scaler = MinMaxScaler()

        #Seperate sorted variables from each other and scale them
        for variable in variables:
            globals()[str(variable) + "_data_train"] = data_pivot_train[variable]
            globals()[str(variable) + "_data_train_"] = pd.DataFrame(X_scaler.fit_transform(globals()[str(variable) + "_data_train"]), index=globals()[str(variable) + "_data_train"].index, columns=globals()[str(variable) + "_data_train"].columns)
            globals()[str(variable) + "_data_train_"] = globals()[str(variable) + "_data_train_"].dropna()


            #Autoencode the training data
            print(f"-----Autoencoding {variable} training data-----")
            encoder = tf.keras.Sequential([
                tf.keras.layers.InputLayer(input_shape = (globals()[str(variable) + "_data_train_"].shape[1])),
                tf.keras.layers.Dense(5),
                tf.keras.layers.Dense(1)
            ])

            decoder = tf.keras.Sequential([
                tf.keras.layers.InputLayer(input_shape = (1)),
                tf.keras.layers.Dense(10),
                tf.keras.layers.Dense(globals()[str(variable) + "_data_train_"].shape[1])
            ])

            autoencoder = tf.keras.Sequential([encoder, decoder])
            autoencoder.compile(loss = 'mean_squared_error', optimizer = 'adam')
            autoencoder.fit(globals()[str(variable) + "_data_train_"], globals()[str(variable) + "_data_train_"], epochs = 5, batch_size = 1024)
            globals()[str(variable) + "_ae_train"] = encoder.predict(globals()[str(variable) + "_data_train_"]).flatten()

        #create dataframe
        market_movements_autoencoder_train = pd.DataFrame(columns = [f"{variable}_market" for variable in variables], index = globals()[str(variable) + "_data_train_"].index)

        #fill dataframe
        for variable in variables:
            market_movements_autoencoder_train[f"{variable}"]  = globals()[str(variable) + "_ae_train"]

        market_movements_autoencoder_train = ta.add_all_ta_features(market_movements_autoencoder_train,
                                                                    open = 'Open',
                                                                    high = 'High',
                                                                    low = 'Low',
                                                                    close = 'Close',
                                                                    volume = 'Volume',
                                                                    fillna = False)

        market_movements_autoencoder_train.columns = [col_name + '_market' for col_name in market_movements_autoencoder_train.columns]
        market_movements_autoencoder_train.fillna(method = "pad", inplace = True)
        market_movements_autoencoder_train = market_movements_autoencoder_train.drop(market_movements_autoencoder_train.columns[market_movements_autoencoder_train.isnull().sum() > 100], axis = 1)
        #market_movements_autoencoder_train.dropna(inplace=True)

        for variable in variables:
            globals()[str(variable) + "_data_test"] = data_pivot_test[variable]
            globals()[str(variable) + "_data_test_"] = pd.DataFrame(X_scaler.fit_transform(globals()[str(variable) + "_data_test"]), index=globals()[str(variable) + "_data_test"].index, columns=globals()[str(variable) + "_data_test"].columns)
            globals()[str(variable) + "_data_test_"] = globals()[str(variable) + "_data_test_"].dropna()

            #Autoencode the testing data
            print(f"-----Autoencoding {variable} test data-----")
            encoder = tf.keras.Sequential([
                tf.keras.layers.InputLayer(input_shape = (globals()[str(variable) + "_data_test_"].shape[1])),
                tf.keras.layers.Dense(5),
                tf.keras.layers.Dense(1)
            ])
            decoder = tf.keras.Sequential([
                tf.keras.layers.InputLayer(input_shape = (1)),
                tf.keras.layers.Dense(10),
                tf.keras.layers.Dense(globals()[str(variable) + "_data_test_"].shape[1])
            ])
            autoencoder = tf.keras.Sequential([encoder, decoder])
            autoencoder.compile(loss = 'mean_squared_error', optimizer = 'adam')
            autoencoder.fit(globals()[str(variable) + "_data_test_"], globals()[str(variable) + "_data_test_"], epochs = 5, batch_size = 1024)
            globals()[str(variable) + "_ae_test"] = encoder.predict(globals()[str(variable) + "_data_test_"]).flatten()

        #create dataframe
        market_movements_autoencoder_test = pd.DataFrame(columns = [f"{variable}_market" for variable in variables], index = globals()[str(variable) + "_data_test_"].index)

        #fill dataframe
        for variable in variables:
            market_movements_autoencoder_test[f"{variable}"]  = globals()[str(variable) + "_ae_test"]

        market_movements_autoencoder_test = ta.add_all_ta_features(market_movements_autoencoder_test,
                                                                   open = 'Open',
                                                                   high = 'High',
                                                                   low = 'Low',
                                                                   close = 'Close',
                                                                   volume = 'Volume',
                                                                   fillna = False)

        market_movements_autoencoder_test.columns = [col_name + '_market' for col_name in market_movements_autoencoder_test.columns]
        market_movements_autoencoder_test.fillna(method = "pad", inplace = True)
        market_movements_autoencoder_test = market_movements_autoencoder_test.drop(market_movements_autoencoder_test.columns[market_movements_autoencoder_test.isnull().sum() > 100], axis = 1)
        #market_movements_autoencoder_test.dropna(inplace=True)

        return market_movements_autoencoder_train, market_movements_autoencoder_test

    def calculateBasicVariables(self, window = 30):
        variables = ["Close", "Open", "High", "Low", "Volume"]
        
        # Training Data
        output_training = pd.DataFrame(self.all_data_training["Time"])
        output_training.set_index(self.all_data_training.index)

        for variable in variables:
            x_vec = self.all_data_training[variable].values
            i_range = range(window, len(x_vec) + 1)
            x_matrix = []
            for i in i_range:
                x_matrix.append(x_vec[i - window:i])

            output_training[f"{variable}_MovMean_{window}"] = np.concatenate([np.repeat(np.NAN, window - 1), np.array(x_matrix).mean(axis=1)])

            i_range = range(window, len(x_vec) + 1)
            x_matrix = []
            for i in i_range:
                x_matrix.append(x_vec[i - window:i])

            output_training[f"{variable}_MovMean_{window}"] = np.concatenate([np.repeat(np.NAN, window - 1), np.array(x_matrix).var(axis=1)])

        output_training["HML"] = self.all_data_training["High"] - self.all_data_training["Low"]
        output_training["CMO"] = self.all_data_training["Close"] - self.all_data_training["Open"]


        # test Data
        output_test = pd.DataFrame(self.all_data_test["Time"])
        output_test.set_index(self.all_data_test.index)

        for variable in variables:
            x_vec = self.all_data_test[variable].values
            i_range = range(window, len(x_vec) + 1)
            x_matrix = []
            for i in i_range:
                x_matrix.append(x_vec[i - window:i])

            output_test[f"{variable}_MovMean_{window}"] = np.concatenate(
                [np.repeat(np.NAN, window - 1), np.array(x_matrix).mean(axis=1)])

            i_range = range(window, len(x_vec) + 1)
            x_matrix = []
            for i in i_range:
                x_matrix.append(x_vec[i - window:i])

            output_test[f"{variable}_MovMean_{window}"] = np.concatenate(
                [np.repeat(np.NAN, window - 1), np.array(x_matrix).var(axis=1)])

        output_test["HML"] = self.all_data_test["High"] - self.all_data_test["Low"]
        output_test["CMO"] = self.all_data_test["Close"] - self.all_data_test["Open"]


        # eval Data
        #TODO: Implement eval dataset and remote if function
        output_eval = {}
        if False:
            output_eval = pd.DataFrame(self.all_data_eval["Time"])
            output_eval.set_index(self.all_data_eval.index)

            for variable in variables:
                x_vec = self.all_data_eval[variable].values
                i_range = range(window, len(x_vec) + 1)
                x_matrix = []
                for i in i_range:
                    x_matrix.append(x_vec[i - window:i])

                output_eval[f"{variable}_MovMean_{window}"] = np.concatenate(
                    [np.repeat(np.NAN, window - 1), np.array(x_matrix).mean(axis=1)])

                i_range = range(window, len(x_vec) + 1)
                x_matrix = []
                for i in i_range:
                    x_matrix.append(x_vec[i - window:i])

                output_eval[f"{variable}_MovMean_{window}"] = np.concatenate(
                    [np.repeat(np.NAN, window - 1), np.array(x_matrix).var(axis=1)])

            output_eval["HML"] = self.all_data_eval["High"] - self.all_data_eval["Low"]
            output_eval["CMO"] = self.all_data_eval["Close"] - self.all_data_eval["Open"]

        return output_training, output_test, output_eval


    def mergeFinalFeatureSetAndTargetVariable(self):

            target_variable_training, target_variable_test = self.getTargetVariable()

            #Inner join
            tmp_df_training = target_variable_training.join(self.featureSet_training, how='inner')
            tmp_df_test = target_variable_test.join(self.featureSet_test, how='inner')
            #tmp_df_eval = target_variable_eval.join(self.featureSet_eval, how='inner')

            tmp_df_training.dropna(inplace=True)
            tmp_df_test.dropna(inplace=True)
            #tmp_df_eval.dropna(inplace=True)

            return tmp_df_training, tmp_df_test#, tmp_df_eval

    def getTargetVariable(self):

        return self.data_training['Target'].to_frame(), self.data_test['Target'].to_frame()#, self.data_eval['Target']

    def scaling(self, df):

        x_scaler = MinMaxScaler(feature_range=(0, 1))
        return x_scaler.fit_transform(df)

    #Help function to create subplots
    def createFigure(self):

        fig = plt.figure(1)
        fig.set_figheight(20)
        fig.set_figwidth(20)

        return fig

    def getCoinName(self):

        return self.coin_name

    def getFinalDataframeTraining(self):

        return self.mergeFinalFeatureSetAndTargetVariable()[0]

    def getFinalDataframeTest(self):

        return self.mergeFinalFeatureSetAndTargetVariable()[1]

    """""
    def getFinalDataFrameEval(self):

        return self.mergeFinalFeatureSetAndTargetVariable()[2]
    """""

    def getTopFeatureVariables(self):

        return self.top_features

    def getFittedData(self):

        return self.y_train_hat, self.y_test_hat

    def getRealData(self):

        return self.y_train, self.y_test
















