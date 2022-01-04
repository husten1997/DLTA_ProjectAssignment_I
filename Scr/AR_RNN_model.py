import numpy as np


class AR_RNN_model:

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


    def __init__(self):
        self.dimRedMethod_dic = {   'Average': 'Average',
                                    'Autoencoder': 'Autoencoder'}
        None


    def setupAutoencoder(self, trainData, testData, outputDim, epochs = 20):
        import tensorflow as tf

        self.Encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(outputDim, activation='linear', input_shape=[trainData.shape[1]],
                                  use_bias=False)
        ])

        self.Decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(trainData.shape[1], activation='linear', input_shape=[outputDim],
                                  use_bias=False)
        ])

        self.Autoencoder = tf.keras.Sequential([self.Encoder, self.Decoder])
        self.Autoencoder.compile(loss='mse', optimizer='adam')
        self.Autoencoder.fit(trainData, trainData, epochs = 20, verbose = 1, validation_data = (testData, testData))


    def generateFeatureSet(self):
        import numpy as np

        # trainFeatures
        index_range = range(self.arOrder, len(self.trainData) - self.forecastSteps - 1)

        X_featuresTrain = []
        Y_featuresTrain = []
        for i in index_range:
            X_featuresTrain.append(self.trainData[(i - self.arOrder):i])
            # Y_data_train.append(price_train[(i+1):(i+forecast_steps)])
            # Y_data_train.append(price_train[[(i + 1), (i + forecast_steps + 1)]])
            Y_featuresTrain.append(self.trainData[[(i + 1)]])


        # testFeatures
        index_range = range(self.arOrder, len(self.testData) - self.forecastSteps - 1)
        X_featuresTest = []
        Y_featuresTest = []
        for i in index_range:
            X_featuresTest.append(self.testData[(i - self.arOrder):i])
            # Y_data_test.append(self.testData[(i+1):(i + self.forecastSteps)])
            # Y_data_test.append(self.testData[[(i + 1), (i + self.forecastSteps + 1]])
            Y_featuresTest.append(self.testData[[(i + 1)]])


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

        if dimRedMethod == self.dimRedMethod_dic["Average"]:
            if self.Encoder is None or self.Decoder is None or self.Autoencoder is None:
                X_featuresTrain, X_featuresTest =  self.generateFeatureSet()
                self.setupAutoencoder(trainData = X_featuresTrain, testData = X_featuresTest, outputDim = outputDim)

            weights = np.matrix(self.Encoder.get_weights()[0])

        elif dimRedMethod == self.dimRedMethod_dic["Autoencoder"]:
            weights = np.zeros((3600, 60))

            for j in range(60):
                for i in range(60):
                    weights[(j * 60) + i, j] = 1 / 60

        else:
            print("Unknown Method")

        return weights


    def generateReducedFeatureSet(self, dimRedMethod):
        import numpy as np

        X_featuresTrain, X_featuresTest = self.generateFeatureSet()
        weights = self.generateWeightMatrix(dimRedMethod = dimRedMethod)

        X_redFeaturesTrain = np.array(np.matmul(X_featuresTrain.reshape((-1, 3600)), weights))
        X_redFeaturesTest = np.array(np.matmul(X_featuresTest.reshape((-1, 3600)), weights))

        return X_redFeaturesTrain, X_redFeaturesTest


    def generateFitFeaturesSet(self, dimRedMethod):
        X_redFeaturesTrain, X_redFeaturesTest = self.generateReducedFeatureSet(dimRedMethod)
        X_featuresTrain, X_featuresTest, Y_featuresTrain, Y_featuresTest = self.generateFeatureSet()
        return X_redFeaturesTrain, X_redFeaturesTest, Y_featuresTrain, Y_featuresTest


    def buildARRNN_KerasTuner(hp):
        import tensorflow as tf
        X_featuresTrain = self.generateReducedFeatureSet('Average')
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
        import tensorflow as tf
        X_featuresTrain = self.generateReducedFeatureSet('Average')
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


    def setARRNN_model(self, method):
        X_redFeaturesTrain, X_redFeaturesTest, Y_featuresTrain, Y_featuresTest = self.generateFitFeaturesSet(
            dimRedMethod='Average')

        if method == "Tuner":
            import keras_tuner as kt

            tuner = kt.RandomSearch(self.buildARRNN_KerasTuner, objective='val_loss', max_trials=5)

            tuner.search(X_redFeaturesTrain, Y_featuresTrain, epochs=10, validation_data=(X_redFeaturesTest, Y_featuresTest),
                         batch_size=512)
            self.ARRNN_model = tuner.get_best_models()[0]

        elif method == "Config":
            config = {} #TODO: Write config dic
            self.ARRNN_model = self.buildARRNN(config)
            history = self.ARRNN_model.fit(X_redFeaturesTrain, Y_featuresTrain, epochs = 20, validation_data = (X_redFeaturesTest, Y_featuresTest), batch_size=1024)


    def getARRNN_model(self):
        return self.ARRNN_model