import numpy as np


class AR_RNN_model:
    def __init__(self, data, arOrder, forecastSteps, coinID, dimRedMethod):
        self.dimRedMethod_dic = {   'Average': 'Average',
                                    'Autoencoder': 'Autoencoder',
                                    'RNNAutoencoder': 'RNNAutoencoder'}

        #TODO: Implement nonlinearity
        self.arOrder = arOrder
        self.forecastSteps = forecastSteps
        self.coinID = coinID

        self.setupData(data, coinID)

        self.dimRedMethod = dimRedMethod

    featureSet = None

    trainData = None
    testData = None
    evalData = None

    dimRedMethod = None
    dimRedMethod_dic = None

    weights = None

    arOrder = None
    forecastSteps = None

    Encoder = None
    Decoder = None
    Autoencoder = None

    ARRNN_model = None

    coinID = None
    coinName = None


    def setupData(self, data, coinID, timestamps = (1609459200, 1622505660), trainFraction = 0.7):
        if len(timestamps) == 2:
            data_eval = data[data.timestamp >= timestamps[1]]
            data = data[(data.timestamp >= timestamps[0]) & (data.timestamp < timestamps[1])]
        # 1.1.2021: 1609459200
        # 1.1.2020: 1577836800

        btc = data[data.Asset_ID == coinID]
        btc.set_index('timestamp', inplace=True)
        btc = btc.reindex(range(btc.index[0], btc.index[-1] + 60, 60), method='pad')
        btc.sort_index(inplace=True)

        btc_eval = data_eval[data_eval.Asset_ID == coinID]
        btc_eval.set_index('timestamp', inplace=True)
        btc_eval = btc_eval.reindex(range(btc_eval.index[0], btc_eval.index[-1] + 60, 60), method='pad')
        btc_eval.sort_index(inplace=True)

        training_size = int(np.floor(len(btc) * trainFraction))

        self.trainData, self.testData = btc[:training_size], btc[training_size:]

        # drop NAs
        self.trainData.dropna(inplace=True)
        self.testData.dropna(inplace=True)

        self.trainData = self.trainData.drop(['Asset_ID', 'Weight', 'Asset_Name'], axis=1)
        self.testData = self.testData.drop(['Asset_ID', 'Weight', 'Asset_Name'], axis=1)


    def setupAutoencoder(self, trainData, testData, outputDim, epochs = 20):
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        import tensorflow as tf

        self.Encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(outputDim, activation = 'linear', input_shape=[trainData.shape[1]],
                                  use_bias=False)
        ])

        self.Decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(trainData.shape[1], activation = 'linear', input_shape=[outputDim],
                                  use_bias=False)
        ])

        self.Autoencoder = tf.keras.Sequential([self.Encoder, self.Decoder])
        self.Autoencoder.compile(loss='mse', optimizer='adam')
        self.Autoencoder.summary()
        self.Autoencoder.fit(trainData, trainData, epochs = epochs, verbose = 1, validation_data = (testData, testData), batch_size=32)

    def setupRNNAutoencoder(self, trainData, testData, outputDim, epochs = 20):
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        #TODO: Fix RNNAutoencoder
        import tensorflow as tf
        print(trainData.shape)

        self.Encoder = tf.keras.Sequential([
            tf.keras.layers.GRU(outputDim, return_sequences=True, input_shape=(trainData.shape[1], 1)),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1))
        ])
        self.Encoder.summary()

        self.Decoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(outputDim, 1)),
            #tf.keras.layers.RepeatVector(lookback),
            #tf.keras.layers.GRU(5, return_sequences=True),
            tf.keras.layers.GRU(1, return_sequences=True),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1))

        ])
        self.Decoder.summary()

        self.Autoencoder = tf.keras.Sequential([self.Encoder, self.Decoder])
        self.Autoencoder.compile(loss='mse', optimizer='adam')
        self.Autoencoder.summary()

        print("Started fit")
        self.Autoencoder.fit(trainData, trainData, epochs = epochs, verbose = 1, validation_data = (testData, testData), batch_size=32)


    def generateFeatureSet(self, variable = 'Target'):
        import numpy as np

        trainData = self.trainData[variable].values
        testData = self.testData[variable].values

        # trainFeatures
        index_range = range(self.arOrder, len(trainData) - self.forecastSteps - 1)

        X_featuresTrain = []
        Y_featuresTrain = []
        for i in index_range:
            X_featuresTrain.append(trainData[(i - self.arOrder):i])
            # Y_data_train.append(trainData[(i+1):(i + self.forecast_steps)])
            # Y_data_train.append(trainData[[(i + 1), (i + self.forecast_steps + 1)]])
            Y_featuresTrain.append(trainData[[(i + 1)]])


        # testFeatures
        index_range = range(self.arOrder, len(testData) - self.forecastSteps - 1)
        X_featuresTest = []
        Y_featuresTest = []
        for i in index_range:
            X_featuresTest.append(testData[(i - self.arOrder):i])
            # Y_data_test.append(testData[(i+1):(i + self.forecastSteps)])
            # Y_data_test.append(testData[[(i + 1), (i + self.forecastSteps + 1]])
            Y_featuresTest.append(testData[[(i + 1)]])


        # evalFeatures
        #TODO: evalFeatures


        X_featuresTrain = np.array(X_featuresTrain)
        X_featuresTest = np.array(X_featuresTest)
        Y_featuresTrain = np.array(Y_featuresTrain)
        Y_featuresTest = np.array(Y_featuresTest)

        return X_featuresTrain, X_featuresTest, Y_featuresTrain, Y_featuresTest


    def generateWeightMatrix(self, dimRedMethod, outputDim = None):
        import numpy as np

        weights = None

        if outputDim is None:
            outputDim = int(self.arOrder / 60)

        if dimRedMethod == self.dimRedMethod_dic["Autoencoder"]:
            if self.Encoder is None or self.Decoder is None or self.Autoencoder is None:
                X_featuresTrain, X_featuresTest =  self.generateFeatureSet()[:2]
                self.setupAutoencoder(trainData = X_featuresTrain, testData = X_featuresTest, outputDim = outputDim)

            weights = np.matrix(self.Encoder.get_weights()[0])

        elif dimRedMethod == self.dimRedMethod_dic["RNNAutoencoder"]:
            if self.Encoder is None or self.Decoder is None or self.Autoencoder is None:
                X_featuresTrain, X_featuresTest =  self.generateFeatureSet()[:2]
                self.setupRNNAutoencoder(trainData = X_featuresTrain, testData = X_featuresTest, outputDim = outputDim)


        elif dimRedMethod == self.dimRedMethod_dic["Average"]:
            weights = np.zeros((3600, 60))

            for j in range(60):
                for i in range(60):
                    weights[(j * 60) + i, j] = 1 / 60

        else:
            print("Unknown Method")

        return weights


    def generateReducedFeatureSet(self, dimRedMethod):
        import numpy as np

        X_featuresTrain, X_featuresTest = self.generateFeatureSet()[:2]
        weights = self.generateWeightMatrix(dimRedMethod = dimRedMethod)

        X_redFeaturesTrain = np.array(np.matmul(X_featuresTrain.reshape((-1, 3600)), weights))
        X_redFeaturesTest = np.array(np.matmul(X_featuresTest.reshape((-1, 3600)), weights))

        return X_redFeaturesTrain, X_redFeaturesTest


    def generateFitFeaturesSet(self, dimRedMethod):
        X_redFeaturesTrain, X_redFeaturesTest = self.generateReducedFeatureSet(dimRedMethod)
        X_featuresTrain, X_featuresTest, Y_featuresTrain, Y_featuresTest = self.generateFeatureSet()
        return X_redFeaturesTrain, X_redFeaturesTest, Y_featuresTrain, Y_featuresTest


    def buildARRNN_KerasTuner(self, hp):
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        import tensorflow as tf
        X_featuresTrain = self.generateReducedFeatureSet(self.dimRedMethod)[0]
        input_shape = X_featuresTrain.shape[1]

        model = tf.keras.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=(input_shape, 1)))

        if hp.Boolean("LSTM"):
            model.add(tf.keras.layers.LSTM(hp.Choice('LSTMunits', [60, 120, 240]), return_sequences=False,
                                           activation=hp.Choice("activation", ["relu", "tanh"])))
        else:
            model.add(tf.keras.layers.GRU(hp.Choice('GRUunits', [60, 120, 240])))

        if hp.Boolean("dropout"):
            model.add(tf.keras.layers.Dropout(rate=0.25))

        model.add(tf.keras.layers.Dense(1))

        learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")

        model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
        return model


    def buildARRNN(self, config):
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        import tensorflow as tf
        X_featuresTrain = self.generateReducedFeatureSet(self.dimRedMethod)[0]
        input_shape = X_featuresTrain.shape[1]

        model = tf.keras.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=(input_shape, 1)))

        if config['LSTM']:
            model.add(tf.keras.layers.LSTM(config['LSTMunits'], return_sequences=False,
                                           activation= config['LSTMactivation'] ))
        else:
            model.add(tf.keras.layers.GRU(config['GRUunits'] ))

        if config['dropout']:
            model.add(tf.keras.layers.Dropout(rate=0.25))

        model.add(tf.keras.layers.Dense(1))

        learning_rate = config['lr']

        model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
        model.summary()

        return model


    def setARRNN_model(self, method = "Config", config = None, dimRedMethod = 'Average', epochs = 20):
        X_redFeaturesTrain, X_redFeaturesTest, Y_featuresTrain, Y_featuresTest = self.generateFitFeaturesSet(
            dimRedMethod = self.dimRedMethod)

        X_redFeaturesTrain = np.reshape(X_redFeaturesTrain, X_redFeaturesTrain.shape + (1,))
        X_redFeaturesTest = np.reshape(X_redFeaturesTest, X_redFeaturesTest.shape + (1,))

        Y_featuresTrain = np.reshape(Y_featuresTrain, Y_featuresTrain.shape + (1,))
        Y_featuresTest = np.reshape(Y_featuresTest, Y_featuresTest.shape + (1,))

        if method == "Tuner":
            import keras_tuner as kt

            tuner = kt.RandomSearch(self.buildARRNN_KerasTuner, objective='val_loss', max_trials=5)

            tuner.search(X_redFeaturesTrain, Y_featuresTrain, epochs = 10, validation_data=(X_redFeaturesTest, Y_featuresTest),
                         batch_size = 1024)
            self.ARRNN_model = tuner.get_best_models()[0]

        elif method == "Config":
            if config is None:
                config = {'LSTM': True,
                          'LSTMunits': 60,
                          'LSTMactivation': 'relu',
                          'GRUunits': 60,
                          'dropout': False,
                          'lr': 1e-2 } # 0.00068464

            self.ARRNN_model = self.buildARRNN(config)
            history = self.ARRNN_model.fit(X_redFeaturesTrain, Y_featuresTrain, epochs = epochs, validation_data = (X_redFeaturesTest, Y_featuresTest), batch_size=1024)

            # History Plot
            import matplotlib.pyplot as plt

            plt.plot(history.history['loss'], label='loss')
            plt.plot(history.history['val_loss'], label='validation_loss')
            plt.legend()
            plt.show()

        else:
            print("Unknown Method")

    def getARRNN_model(self):
        return self.ARRNN_model

    def getFittedData(self):
        X_redFeaturesTrain, X_redFeaturesTest = self.generateFitFeaturesSet(
            dimRedMethod=self.dimRedMethod)[:2]

        X_redFeaturesTrain = np.reshape(X_redFeaturesTrain, X_redFeaturesTrain.shape + (1,))
        X_redFeaturesTest = np.reshape(X_redFeaturesTest, X_redFeaturesTest.shape + (1,))

        Y_train_hat = self.ARRNN_model.predict(X_redFeaturesTrain)
        Y_test_hat = self.ARRNN_model.predict(X_redFeaturesTest)

        return Y_train_hat, Y_test_hat