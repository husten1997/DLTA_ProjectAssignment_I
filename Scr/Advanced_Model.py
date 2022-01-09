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

class Advanced_Model():

    coin_id = 0
    coin_name = ''

    data = pd.DataFrame()

    data_training = pd.DataFrame()
    data_test = pd.DataFrame()
    data_eval = pd.DataFrame()

    featureSet_training = pd.DataFrame()
    featureSet_test = pd.DataFrame()
    featureSet_eval = pd.DataFrame()
    top_20_features = []

    y_train = []
    y_test = []
    y_train_hat = []
    y_test_hat = []

    def __init__(self, coin_id, all_data, all_data_details):

        self.coin_id = coin_id
        self.data = all_data[all_data.Asset_ID == coin_id]

        all_data_details = all_data_details[all_data_details.Asset_ID == coin_id]
        all_data_details.reset_index(drop=True, inplace=True)
        self.coin_name = all_data_details.Asset_Name[0]

        self.setupData()

    def setupData(self):

        self.data.set_index('timestamp', inplace=True)
        self.data = self.data.reindex(range(self.data.index[0], self.data.index[-1] + 60, 60), method='pad')
        self.data.sort_index(inplace=True)

        training_start, test_start, eval_start = self.getPeriods()

        self.data_training = self.data[(self.data.index >= training_start) & (self.data.index <= test_start)]
        self.data_test = self.data[(self.data.index >= test_start) & (self.data.index <= eval_start)]
        self.data_eval = self.data[(self.data.index >= eval_start)]

        self.mergeFeatureSets()
        self.setTop20FeatureVariables()

    def applyModel(self, fnn, active_func, neurons_first, dropout_first, neurons_second, dropout_second, epochs):

        tmp_df_training = self.mergeFinalFeatureSetAndTargetVariable()[0]
        tmp_df_test = self.mergeFinalFeatureSetAndTargetVariable()[1]

        x_train = tmp_df_training[self.top_20_features].drop(['Target'], axis=1)
        x_test = tmp_df_test[self.top_20_features].drop(['Target'], axis=1)

        self.y_train = tmp_df_training['Target'].values
        self.y_test = tmp_df_test['Target'].values

        x_train_ = self.scaling(x_train)
        x_test_ = self.scaling(x_test)

        config = {'fnn': fnn,
                  'active_func': active_func,
                  'neurons_first_layer': neurons_first,
                  'dropout_first_layer': dropout_first,
                  'neurons_second_layer': neurons_second,
                  'dropout_second_layer': dropout_second}

        self.adv_model = self.buildAdvModel(config, x_train_)
        history = self.adv_model.fit(x_train_, self.y_train, epochs=epochs, validation_data=(x_test_, self.y_test))
        plt.plot(history.history['loss'], label='training')
        plt.plot(history.history['val_loss'], label='test')
        plt.legend()
        plt.show()

        self.y_train_hat = self.adv_model.predict(x_train_).flatten()
        self.y_test_hat = self.adv_model.predict(x_test_).flatten()

        #Show Feature Importance after estimation of the model
        self.showFeatureImportance(x_train_, x_train, self.y_train)

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

        return start_train, start_test, start_eval

    def mergeFeatureSets(self):

        #Get various feature sets
        tech_indicators_training, tech_indicators_test, tech_indicators_eval = self.calculateTechnicalIndicators()
        #TODO: calculate further feature sets, e.g. market variables..

        #TODO: merge different feature sets to one final feature set (using pd.merge() function)..

        #Only temporary code (see TODO's)
        self.featureSet_training = tech_indicators_training
        self.featureSet_test = tech_indicators_test
        self.featureSet_eval = tech_indicators_eval

    def setTop20FeatureVariables(self):

        tmp_df_training = self.mergeFinalFeatureSetAndTargetVariable()[0]

        find_corr_features = tmp_df_training.corr(method='spearman')['Target'].abs().sort_values(ascending=False)

        print('20 feature variables of [' +
            self.coin_name +
            '], that correlate highest with the target variable: \n' +
             str(find_corr_features[1:21]))

        self.top_20_features = list(find_corr_features[:21].index)

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
        """"" -> derzeit nicht relevant
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
        tmp_df_test.drop(
            ['Asset_ID', 'Count', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'VWAP', 'Target', 'Time',
             'Weight', 'Asset_Name'], axis=1)
        # tmp_df_eval.drop(['Asset_ID', 'Count', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'VWAP', 'Target', 'Time',
        # 'Weight', 'Asset_Name'], axis=1)

        #Delete variables with more than 100 missing values
        tmp_df_training = tmp_df_training.drop(tmp_df_training.columns[tmp_df_training.isnull().sum() > 100], axis=1)
        tmp_df_test = tmp_df_test.drop(tmp_df_test.columns[tmp_df_test.isnull().sum() > 100], axis=1)
        #tmp_df_eval = tmp_df_eval.drop(tmp_df_eval.columns[tmp_df_eval.isnull().sum() > 100], axis=1)

        return tmp_df_training, tmp_df_test, tmp_df_eval

    def mergeFinalFeatureSetAndTargetVariable(self):

        target_variable_training, target_variable_test, target_variable_eval = self.getTargetVariable()

        #Inner join
        tmp_df_training = pd.merge(target_variable_training, self.featureSet_training, left_index=True, right_index=True)
        tmp_df_test = pd.merge(target_variable_test, self.featureSet_test, left_index=True, right_index=True)
        tmp_df_eval = pd.merge(target_variable_eval, self.featureSet_eval, left_index=True, right_index=True)

        tmp_df_training.dropna(inplace=True)
        tmp_df_test.dropna(inplace=True)
        tmp_df_eval.dropna(inplace=True)

        return tmp_df_training, tmp_df_test, tmp_df_eval

    def getTargetVariable(self):

        return self.data_training['Target'], self.data_test['Target'], self.data_eval['Target']

    def buildAdvModel(self, config, x_train_):

        if config['fnn']:
            model = tf.keras.Sequential([
                    tf.keras.layers.InputLayer(input_shape=(x_train_.shape[1])),
                    tf.keras.layers.Dense(config['neurons_first_layer'], activation=config['active_func']),
                    tf.keras.layers.Dropout(config['dropout_first_layer']),
                    tf.keras.layers.Dense(config['neurons_second_layer'], activation=config['active_func']),
                    tf.keras.layers.Dropout(config['dropout_second_layer']),
                    tf.keras.layers.Dense(1)
            ])
        else:
            print('A other neural network than the Forward Neural Network is currently not defined')

        model.compile(loss='mean_absolute_error', optimizer='adam')
        model.summary()

        return model

    def showFeatureImportance(self, x_train_, x_train, y_train):

        x = tf.Variable(x_train_)
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

    def getFinalDataFrameEval(self):

        return self.mergeFinalFeatureSetAndTargetVariable()[2]

    def getTop20FeatureVariables(self):

        return self.top_20_features

    def getFittedData(self):

        return self.y_train_hat, self.y_test_hat

    def getRealData(self):

        return self.y_train, self.y_test
















