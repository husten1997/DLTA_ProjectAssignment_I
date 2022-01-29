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
from datetime import datetime
import time

class Indicator_Model():
    #Initialize class variables
    coin_id = 0
    coin_name = ''

    training_start = 0
    eval_start = 0

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

    encoder_dic = {}

    x_train_ = []
    x_test_ = []
    x_eval_ = []

    y_train = []
    y_test = []
    y_eval = []
    y_train_hat = []
    y_test_hat = []
    y_eval_hat = []

    modelType = ""
    minCorr = 0

    def __init__(self, coin_id, all_data, all_data_details, trainStart="25/05/2021", evalStart="01/06/2021",
                 minCorr=0.03):

        totimestamp = lambda s: np.int32(time.mktime(datetime.strptime(s, "%d/%m/%Y").timetuple()))

        timestamps = (totimestamp(trainStart), totimestamp(evalStart))

        self.training_start = timestamps[0]

        self.eval_start = timestamps[1]

        self.all_data = all_data

        self.coin_id = coin_id
        self.data = all_data[all_data.Asset_ID == coin_id]

        self.all_data_details = all_data_details[all_data_details.Asset_ID == coin_id]
        self.all_data_details.reset_index(drop=True, inplace=True)
        self.coin_name = self.all_data_details.Asset_Name[0]

        self.minCorr = minCorr

        self.setupData()

    def setupData(self):

        self.all_data.set_index('timestamp', inplace=True)

        self.data.set_index('timestamp', inplace=True)
        self.data = self.data.reindex(range(self.data.index[0], self.data.index[-1] + 60, 60), method='pad')
        self.data.sort_index(inplace=True)

        # This step is necessary for the calculation of all further feature variables
        tmp_data = self.data[(self.data.index >= self.training_start) & (self.data.index < self.eval_start)]
        train_index = int(np.floor(tmp_data.shape[0] * 0.7))
        self.data_training, self.data_test = tmp_data[:train_index], tmp_data[train_index:]

        train_timestamps = (np.min(self.data_training.index), np.max(self.data_training.index))
        test_timestamps = (np.min(self.data_test.index), np.max(self.data_test.index))

        self.data_eval = self.data[(self.data.index >= self.eval_start)]

        self.all_data_training = self.all_data[
            (self.all_data.index >= train_timestamps[0]) & (self.all_data.index <= train_timestamps[1])]
        self.all_data_test = self.all_data[
            (self.all_data.index >= test_timestamps[0]) & (self.all_data.index <= test_timestamps[1])]

        self.all_data_eval = self.all_data[(self.all_data.index >= self.eval_start)]

        self.mergeFeatureSets()
        self.stationarity_transformation()
        self.setTopFeatureVariables()

    def mergeFeatureSets(self):

        # Get various feature sets
        basic_variables_training, basic_variables_test, basic_variables_eval = self.calculateBasicVariables()
        tech_indicators_training, tech_indicators_test, tech_indicators_eval = self.calculateTechnicalIndicators()
        market_movements_autoencoder_training, market_movements_autoencoder_test, market_movements_autoencoder_eval = self.calculateTechnicalMarketIndicators()

        # Merge different feature sets to one final feature set for each period
        self.featureSet_training = basic_variables_training.join(tech_indicators_training, how='inner')
        self.featureSet_test = basic_variables_test.join(tech_indicators_test, how='inner')
        self.featureSet_eval = basic_variables_eval.join(tech_indicators_eval, how='inner')

        self.featureSet_training = self.featureSet_training.join(market_movements_autoencoder_training, how='inner')
        self.featureSet_test = self.featureSet_test.join(market_movements_autoencoder_test, how='inner')
        self.featureSet_eval = self.featureSet_eval.join(market_movements_autoencoder_eval, how='inner')

        self.featureSet_training.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.featureSet_test.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.featureSet_eval.replace([np.inf, -np.inf], np.nan, inplace=True)

        self.featureSet_training.dropna(inplace=True)
        self.featureSet_test.dropna(inplace=True)
        self.featureSet_eval.dropna(inplace=True)

    def stationarity_transformation(self):

        for variable in self.featureSet_training.columns:
            timeseries = self.featureSet_training[variable]
            result = adfuller(timeseries)
            p_value = result[1]
            if p_value > 0.05:
                self.featureSet_training[variable] = self.featureSet_training[variable].diff()

        for variable in self.featureSet_test.columns:
            timeseries = self.featureSet_test[variable]
            result = adfuller(timeseries)
            p_value = result[1]
            if p_value > 0.05:
                self.featureSet_test[variable] = self.featureSet_test[variable].diff()

        for variable in self.featureSet_eval.columns:
            timeseries = self.featureSet_eval[variable]
            result = adfuller(timeseries)
            p_value = result[1]
            if p_value > 0.05:
                self.featureSet_eval[variable] = self.featureSet_eval[variable].diff()

        self.featureSet_training.dropna(inplace=True)
        self.featureSet_test.dropna(inplace=True)
        self.featureSet_eval.dropna(inplace=True)

    def setTopFeatureVariables(self):

        tmp_df_training = self.mergeFinalFeatureSetAndTargetVariable()[0]

        find_corr_features = tmp_df_training.corr(method='spearman')['Target'].abs().sort_values(ascending=False)
        # Delete first row (correlation of the target variable with itself) for the print out
        find_corr_features_print = find_corr_features.drop(index=find_corr_features.index[0])

        print('Correlation of the Features of [' +
              self.coin_name +
              '] with the target variable: \n' + str((find_corr_features_print.loc[find_corr_features > self.minCorr])))

        self.top_features = list(find_corr_features.loc[find_corr_features > self.minCorr].index)

    def applyModel(self, epochs, method="FNN", modelType="GRU"):
        self.modelType = modelType
        tmp_df_training = self.mergeFinalFeatureSetAndTargetVariable()[0]
        tmp_df_test = self.mergeFinalFeatureSetAndTargetVariable()[1]
        tmp_df_eval = self.mergeFinalFeatureSetAndTargetVariable()[2]

        x_train = tmp_df_training[self.top_features].drop(['Target'], axis=1)
        x_test = tmp_df_test[self.top_features].drop(['Target'], axis=1)
        x_eval = tmp_df_eval[self.top_features].drop(['Target'], axis=1)

        self.y_train = tmp_df_training['Target'].values
        self.y_test = tmp_df_test['Target'].values
        self.y_eval = tmp_df_eval['Target'].values

        # Variable x_train_ is also used in method buildAdvModel_KerasTuner(), therefore we have to declare it as class variable
        self.x_train_ = self.scaling(x_train)
        self.x_test_ = self.scaling(x_test)
        self.x_eval_ = self.scaling(x_eval)

        if method == "FNN":
            config = {
                "RNN_Lookback": 15,
                # GRU
                "RNN_L1_units": 120,
                "RNN_L1_actfun": "tanh",
                "RNN_L1_dropoutBool": True,
                "RNN_L1_dropoutUnit": 0.25,
                "RNN_L2_units": 60,
                "RNN_L2_actfun": "tanh",
                "RNN_L2_dropoutBool": True,
                "RNN_L2_dropoutUnit": 0.25,
                "lr": 1e-2}

            self.adv_model = self.buildAdvModel(config)

            history = self.adv_model.fit(self.x_train_, self.y_train, epochs=epochs,
                                         validation_data=(self.x_test_, self.y_test), batch_size=1024)
            plt.plot(history.history['loss'], label='training')
            plt.plot(history.history['val_loss'], label='test')
            plt.legend()
            plt.show()

        elif method == "Tuner":
            self.tuner = kt.RandomSearch(self.buildAdvModel_KerasTuner, objective='val_loss', max_trials=10)

            self.tuner.search(self.x_train_, self.y_train, epochs=10, validation_data=(self.x_test_, self.y_test),
                              batch_size=1024)
            self.adv_model = self.tuner.get_best_models()[0]

            history = self.adv_model.fit(self.x_train_, self.y_train, epochs=epochs,
                                         validation_data=(self.x_test_, self.y_test), batch_size=1024)
            plt.plot(history.history['loss'], label='training')
            plt.plot(history.history['val_loss'], label='test')
            plt.legend()
            plt.show()

        self.y_train_hat = self.adv_model.predict(self.x_train_).flatten()
        self.y_test_hat = self.adv_model.predict(self.x_test_).flatten()
        self.y_eval_hat = self.adv_model.predict(self.x_eval_).flatten()

        # Show Feature Importance after estimation of the model
        self.showFeatureImportance(x_train, self.y_train)

    def buildAdvModel(self, config):
        model = tf.keras.Sequential()

        input_shape = self.x_train_.shape[1]
        model.add(tf.keras.layers.InputLayer(input_shape=(input_shape)))

        model.add(tf.keras.layers.RepeatVector(config["RNN_Lookback"]))

        if self.modelType == "GRU":
            model.add(
                tf.keras.layers.GRU(config['RNN_L1_units'], return_sequences=True, activation=config["RNN_L1_actfun"]))
            if config['RNN_L1_dropoutBool']:
                model.add(tf.keras.layers.Dropout(config['RNN_L1_dropoutUnit']))
            model.add(
                tf.keras.layers.GRU(config['RNN_L2_units'], return_sequences=False, activation=config["RNN_L2_actfun"]))
            if config['RNN_L2_dropoutBool']:
                model.add(tf.keras.layers.Dropout(config['RNN_L2_dropoutUnit']))
        else:
            model.add(
                tf.keras.layers.LSTM(config['RNN_L1_units'], return_sequences=True, activation=config["RNN_L1_actfun"]))
            if config['RNN_L1_dropoutBool']:
                model.add(tf.keras.layers.Dropout(config['RNN_L1_dropoutUnit']))
            model.add(
                tf.keras.layers.LSTM(config['RNN_L2_units'], return_sequences=False,
                                     activation=config["RNN_L2_actfun"]))
            if config['RNN_L2_dropoutBool']:
                model.add(tf.keras.layers.Dropout(config['RNN_L2_dropoutUnit']))

        model.add(tf.keras.layers.Dense(1))

        learning_rate = config["lr"]

        model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))

        model.summary()

        return model

    def buildAdvModel_KerasTuner(self, hp):
        model = tf.keras.Sequential()

        input_shape = self.x_train_.shape[
            1]  # Would be x_train_.shape[1], but we dont have parameters with the Keras_tuner (or do we?)
        model.add(tf.keras.layers.InputLayer(input_shape=(input_shape)))

        model.add(tf.keras.layers.RepeatVector(hp.Choice('RNN_Lookback', [15, 30, 60, 120, 240])))

        if self.modelType == "GRU":
            model.add(
                tf.keras.layers.GRU(hp.Choice('RNN_L1_units', [60, 120, 240]), return_sequences=True,
                                    activation=hp.Choice("RNN_L1_actfun", ["relu", "tanh", "selu"])))
            if hp.Boolean('RNN_L1_dropoutBool'):
                model.add(tf.keras.layers.Dropout(hp.Choice('RNN_L1_dropoutUnit', [0.12, 0.25, 0.5])))
            model.add(
                tf.keras.layers.GRU(hp.Choice('RNN_L2_units', [60, 120, 240]), return_sequences=False,
                                    activation=hp.Choice("RNN_L2_actfun", ["relu", "tanh", "selu"])))
            if hp.Boolean('RNN_L2_dropoutBool'):
                model.add(tf.keras.layers.Dropout(hp.Choice('RNN_L2_dropoutUnit', [0.12, 0.25, 0.5])))
        else:
            model.add(
                tf.keras.layers.LSTM(hp.Choice('RNN_L1_units', [60, 120, 240]), return_sequences=True,
                                     activation=hp.Choice("RNN_L1_actfun", ["relu", "tanh", "selu"])))
            if hp.Boolean('RNN_L1_dropoutBool'):
                model.add(tf.keras.layers.Dropout(hp.Choice('RNN_L1_dropoutUnit', [0.12, 0.25, 0.5])))
            model.add(
                tf.keras.layers.LSTM(hp.Choice('RNN_L2_units', [60, 120, 240]), return_sequences=False,
                                     activation=hp.Choice("RNN_L2_actfun", ["relu", "tanh", "selu"])))
            if hp.Boolean('RNN_L2_dropoutBool'):
                model.add(tf.keras.layers.Dropout(hp.Choice('RNN_L2_dropoutUnit', [0.12, 0.25, 0.5])))

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

        # Delete variables that are no technical indicators
        tmp_df_training = tmp_df_training.drop(
            ['Asset_ID', 'Count', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'VWAP', 'Target', 'Time',
             'Weight', 'Asset_Name'], axis=1)
        tmp_df_test = tmp_df_test.drop(
            ['Asset_ID', 'Count', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'VWAP', 'Target', 'Time',
             'Weight', 'Asset_Name'], axis=1)
        tmp_df_eval = tmp_df_eval.drop(
            ['Asset_ID', 'Count', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'VWAP', 'Target', 'Time',
             'Weight', 'Asset_Name'], axis=1)

        # Delete variables with more than 100 missing values
        tmp_df_training = tmp_df_training.drop(tmp_df_training.columns[tmp_df_training.isnull().sum() > 100], axis=1)
        tmp_df_test = tmp_df_test.drop(tmp_df_test.columns[tmp_df_test.isnull().sum() > 100], axis=1)
        tmp_df_eval = tmp_df_eval.drop(tmp_df_eval.columns[tmp_df_eval.isnull().sum() > 100], axis=1)

        return tmp_df_training, tmp_df_test, tmp_df_eval

    def calculateTechnicalMarketIndicators(self):

        # variable names
        variables = ["Close", "Open", "High", "Low", "Volume"]

        # create pivot table for training data
        data_pivot_train = self.all_data_training.pivot_table(index=self.all_data_training.index, columns='Asset_ID')
        data_pivot_train = data_pivot_train.drop(data_pivot_train.columns[data_pivot_train.isnull().sum() > 1000],
                                                 axis=1)

        # create pivot table for testing data
        data_pivot_test = self.all_data_test.pivot_table(index=self.all_data_test.index, columns='Asset_ID')
        data_pivot_test = data_pivot_test.drop(data_pivot_test.columns[data_pivot_test.isnull().sum() > 1000], axis=1)

        data_pivot_eval = self.all_data_eval.pivot_table(index=self.all_data_eval.index, columns='Asset_ID')
        data_pivot_eval = data_pivot_eval.drop(data_pivot_eval.columns[data_pivot_eval.isnull().sum() > 1000], axis=1)

        X_scaler = MinMaxScaler()

        # training_______________________________________________________________________________________________________

        # Seperate sorted variables from each other and scale them
        for variable in variables:
            globals()[str(variable) + "_data_train"] = data_pivot_train[variable]
            globals()[str(variable) + "_data_train_"] = pd.DataFrame(
                X_scaler.fit_transform(globals()[str(variable) + "_data_train"]),
                index=globals()[str(variable) + "_data_train"].index,
                columns=globals()[str(variable) + "_data_train"].columns)
            globals()[str(variable) + "_data_train_"] = globals()[str(variable) + "_data_train_"].dropna()

            # Autoencode the training data
            print(f"-----Autoencoding {variable} training data-----")
            encoder = tf.keras.Sequential([
                tf.keras.layers.InputLayer(input_shape=(globals()[str(variable) + "_data_train_"].shape[1])),
                tf.keras.layers.Dense(5),
                tf.keras.layers.Dense(1)
            ])

            decoder = tf.keras.Sequential([
                tf.keras.layers.InputLayer(input_shape=(1)),
                tf.keras.layers.Dense(10),
                tf.keras.layers.Dense(globals()[str(variable) + "_data_train_"].shape[1])
            ])

            autoencoder = tf.keras.Sequential([encoder, decoder])
            autoencoder.compile(loss='mean_squared_error', optimizer='adam')
            autoencoder.fit(globals()[str(variable) + "_data_train_"], globals()[str(variable) + "_data_train_"],
                            epochs=5, batch_size=1024)
            self.encoder_dic[str(variable)] = encoder
            globals()[str(variable) + "_ae_train"] = self.encoder_dic[str(variable)].predict(
                globals()[str(variable) + "_data_train_"]).flatten()

        # create dataframe
        market_movements_autoencoder_train = pd.DataFrame(columns=[f"{variable}_market" for variable in variables],
                                                          index=globals()[str(variable) + "_data_train_"].index)

        # fill dataframe
        for variable in variables:
            market_movements_autoencoder_train[f"{variable}"] = globals()[str(variable) + "_ae_train"]

        market_movements_autoencoder_train = ta.add_all_ta_features(market_movements_autoencoder_train,
                                                                    open='Open',
                                                                    high='High',
                                                                    low='Low',
                                                                    close='Close',
                                                                    volume='Volume',
                                                                    fillna=False)

        market_movements_autoencoder_train.columns = [col_name + '_market' for col_name in
                                                      market_movements_autoencoder_train.columns]
        market_movements_autoencoder_train.fillna(method="pad", inplace=True)
        market_movements_autoencoder_train = market_movements_autoencoder_train.drop(
            market_movements_autoencoder_train.columns[market_movements_autoencoder_train.isnull().sum() > 100], axis=1)

        # test___________________________________________________________________________________________________________

        for variable in variables:
            globals()[str(variable) + "_data_test"] = data_pivot_test[variable]
            globals()[str(variable) + "_data_test_"] = pd.DataFrame(
                X_scaler.fit_transform(globals()[str(variable) + "_data_test"]),
                index=globals()[str(variable) + "_data_test"].index,
                columns=globals()[str(variable) + "_data_test"].columns)
            globals()[str(variable) + "_data_test_"] = globals()[str(variable) + "_data_test_"].dropna()
            globals()[str(variable) + "_ae_test"] = self.encoder_dic[str(variable)].predict(
                globals()[str(variable) + "_data_test_"]).flatten()

        # create dataframe
        market_movements_autoencoder_test = pd.DataFrame(columns=[f"{variable}_market" for variable in variables],
                                                         index=globals()[str(variable) + "_data_test_"].index)

        # fill dataframe
        for variable in variables:
            market_movements_autoencoder_test[f"{variable}"] = globals()[str(variable) + "_ae_test"]

        market_movements_autoencoder_test = ta.add_all_ta_features(market_movements_autoencoder_test,
                                                                   open='Open',
                                                                   high='High',
                                                                   low='Low',
                                                                   close='Close',
                                                                   volume='Volume',
                                                                   fillna=False)

        market_movements_autoencoder_test.columns = [col_name + '_market' for col_name in
                                                     market_movements_autoencoder_test.columns]
        market_movements_autoencoder_test.fillna(method="pad", inplace=True)
        market_movements_autoencoder_test = market_movements_autoencoder_test.drop(
            market_movements_autoencoder_test.columns[market_movements_autoencoder_test.isnull().sum() > 100], axis=1)

        # eval___________________________________________________________________________________________________________

        for variable in variables:
            globals()[str(variable) + "_data_eval"] = data_pivot_eval[variable]
            globals()[str(variable) + "_data_eval_"] = pd.DataFrame(
                X_scaler.fit_transform(globals()[str(variable) + "_data_eval"]),
                index=globals()[str(variable) + "_data_eval"].index,
                columns=globals()[str(variable) + "_data_eval"].columns)
            globals()[str(variable) + "_data_eval_"] = globals()[str(variable) + "_data_eval_"].dropna()

            globals()[str(variable) + "_ae_eval"] = self.encoder_dic[str(variable)].predict(
                globals()[str(variable) + "_data_eval_"]).flatten()

            # create dataframe
        market_movements_autoencoder_eval = pd.DataFrame(columns=[f"{variable}_market" for variable in variables],
                                                         index=globals()[str(variable) + "_data_eval_"].index)

        # fill dataframe
        for variable in variables:
            market_movements_autoencoder_eval[f"{variable}"] = globals()[str(variable) + "_ae_eval"]

        market_movements_autoencoder_eval = ta.add_all_ta_features(market_movements_autoencoder_eval,
                                                                   open='Open',
                                                                   high='High',
                                                                   low='Low',
                                                                   close='Close',
                                                                   volume='Volume',
                                                                   fillna=False)

        market_movements_autoencoder_eval.columns = [col_name + '_market' for col_name in
                                                     market_movements_autoencoder_eval.columns]
        market_movements_autoencoder_eval.fillna(method="pad", inplace=True)
        market_movements_autoencoder_eval = market_movements_autoencoder_eval.drop(
            market_movements_autoencoder_eval.columns[market_movements_autoencoder_eval.isnull().sum() > 100], axis=1)

        return market_movements_autoencoder_train, market_movements_autoencoder_test, market_movements_autoencoder_eval

    def calculateBasicVariables(self, window=30):

        variables = ["Close", "Open", "High", "Low", "Volume"]

        # Training Data
        output_training = pd.DataFrame()

        for variable in variables:
            x_vec = self.data_training[variable].values
            i_range = range(window, len(x_vec) + 1)
            x_matrix = []
            for i in i_range:
                x_matrix.append(x_vec[i - window:i])

            output_training[f"{variable}_MovMean_{window}"] = np.concatenate(
                [np.repeat(np.NAN, window - 1), np.array(x_matrix).mean(axis=1)])

            i_range = range(window, len(x_vec) + 1)
            x_matrix = []
            for i in i_range:
                x_matrix.append(x_vec[i - window:i])

            output_training[f"{variable}_MovMean_{window}"] = np.concatenate(
                [np.repeat(np.NAN, window - 1), np.array(x_matrix).var(axis=1)])

        output_training["HML"] = self.data_training["High"].values - self.data_training["Low"].values
        output_training["CMO"] = self.data_training["Close"].values - self.data_training["Open"].values

        # Test Data
        output_test = pd.DataFrame()

        for variable in variables:
            x_vec = self.data_test[variable].values
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

        output_test["HML"] = self.data_test["High"].values - self.data_test["Low"].values
        output_test["CMO"] = self.data_test["Close"].values - self.data_test["Open"].values

        output_eval = pd.DataFrame()

        for variable in variables:
            x_vec = self.data_eval[variable].values
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

        output_eval["HML"] = self.data_eval["High"].values - self.data_eval["Low"].values
        output_eval["CMO"] = self.data_eval["Close"].values - self.data_eval["Open"].values

        output_training.set_index(self.data_training.index, inplace=True)
        output_test.set_index(self.data_test.index, inplace=True)
        output_eval.set_index(self.data_eval.index, inplace=True)

        return output_training, output_test, output_eval

    def mergeFinalFeatureSetAndTargetVariable(self):

        target_variable_training, target_variable_test, target_variable_eval = self.getTargetVariable()

        # Inner join
        tmp_df_training = target_variable_training.join(self.featureSet_training, how='inner')
        tmp_df_test = target_variable_test.join(self.featureSet_test, how='inner')
        tmp_df_eval = target_variable_eval.join(self.featureSet_eval, how='inner')

        tmp_df_training.dropna(inplace=True)
        tmp_df_test.dropna(inplace=True)
        tmp_df_eval.dropna(inplace=True)

        return tmp_df_training, tmp_df_test, tmp_df_eval

    def getTargetVariable(self):

        return self.data_training['Target'].to_frame(), self.data_test['Target'].to_frame(), self.data_eval[
            'Target'].to_frame()

    def scaling(self, df):

        x_scaler = MinMaxScaler(feature_range=(0, 1))
        return x_scaler.fit_transform(df)

    # Help function to create subplots
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

    def getFinalDataFrameEval(self):

        return self.mergeFinalFeatureSetAndTargetVariable()[2]

    def getTopFeatureVariables(self):

        return self.top_features

    def getFittedData(self):

        return self.y_train_hat, self.y_test_hat, self.y_eval_hat

    def getRealData(self):

        return self.y_train, self.y_test, self.y_eval
















