import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFECV
import matplotlib.pyplot as plt


class AR_RNN_model:

    featureSet = None

    trainData = None
    testData = None
    evalData = None

    trainDF = None
    testDF = None
    evalDF = None

    dimRedMethod = None
    dimRedMethod_dic = None

    weights = None

    arOrder = None
    forecastSteps = None
    outputDim = None

    Encoder = None
    Decoder = None
    Autoencoder = None

    ARRNN_model = None

    coinID = None
    coinName = None


    def __init__(self, data, arOrder, forecastSteps, coinID, dimRedMethod, outputDim = None, dimRedRatio = None):
        self.dimRedMethod_dic = {   'Average': 'Average',
                                    'Autoencoder': 'Autoencoder',
                                    'RNNAutoencoder': 'RNNAutoencoder',
                                    'RandomForest': 'RandomForest'}

        #TODO: Implement nonlinearity

        self.arOrder = arOrder
        self.forecastSteps = forecastSteps
        self.coinID = coinID

        self.setupData(data, coinID)
        self.generateFeatureSet()

        self.dimRedMethod = dimRedMethod

        if dimRedMethod != "None":
            self.setOutputDim(outputDim, dimRedRatio)


    def setOutputDim(self, outputDim = None, dimRedRatio = None):
        if outputDim is None and dimRedRatio is None:
            print("Please supply outputdim or dimReductionRatio")
            self.outputDim = 1
        elif outputDim is None and dimRedRatio is not None:
            if self.arOrder % dimRedRatio != 0:
                print("The dim reduction ratio must be an integer factor of arOrder")
            else:
                self.outputDim = self.arOrder / dimRedRatio
        elif outputDim is not None and dimRedRatio is None:
            self.outputDim = outputDim
            #TODO: Implement test for integer devision


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
        self.evalData = btc_eval

        # drop NAs
        self.trainData.dropna(inplace=True)
        self.testData.dropna(inplace=True)
        self.evalData.dropna(inplace=True)

        self.trainData = self.trainData.drop(['Asset_ID', 'Weight', 'Asset_Name'], axis=1)
        self.testData = self.testData.drop(['Asset_ID', 'Weight', 'Asset_Name'], axis=1)
        self.evalData = self.evalData.drop(['Asset_ID', 'Weight', 'Asset_Name'], axis=1)


    def generateFeatureSet(self):
        trainData_X = self.trainData['Close'].values
        testData_X = self.testData['Close'].values
        evalData_X = self.evalData['Close'].values

        trainData_Y = self.trainData['Target'].values
        testData_Y = self.testData['Target'].values
        evalData_Y = self.evalData['Target'].values

        self.trainDF = pd.DataFrame(np.column_stack([trainData_X[15:-1], trainData_X[:-16], trainData_Y[16:]]))
        self.trainDF.columns = ['P1', 'P16', 'Target']

        self.testDF = pd.DataFrame(np.column_stack([testData_X[15:-1], testData_X[:-16], testData_Y[16:]]))
        self.testDF.columns = ['P1', 'P16', 'Target']

        self.evalDF = pd.DataFrame(np.column_stack([evalData_X[15:-1], evalData_X[:-16], evalData_Y[16:]]))
        self.evalDF.columns = ['P1', 'P16', 'Target']


    def setupAutoencoder(self, config):
        # define a recurrent network with Gated Recurrent Units

        input_dim = self.arOrder
        output_dim = self.outputDim

        encoder_input = tf.keras.layers.Input(shape=(1,))

        encoder_output = encoder_input
        encoder_output = tf.keras.layers.RepeatVector(input_dim)(encoder_output)
        encoder_output = tf.keras.layers.Flatten()(encoder_output)
        encoder_output = tf.keras.layers.Dense(output_dim)(encoder_output)

        #TODO: Ausprobieren GRU
        #TODO: Research machen ob Encoder/Decoder symetrisch aufgebaut sein müssen

        encoder_output = tf.keras.layers.Reshape((output_dim, 1))(encoder_output)

        self.Encoder = tf.keras.Model(inputs=encoder_input, outputs=encoder_output)
        self.Encoder._name = "Encoder"

        self.Encoder.summary()

        decoder_input = tf.keras.layers.Input(shape=(output_dim, 1,))

        decoder_output = tf.keras.layers.GRU(1)(decoder_input)
        decoder_output = tf.keras.layers.Dense(1)(decoder_output)

        self.Decoder = tf.keras.Model(inputs=decoder_input, outputs=decoder_output)
        self.Decoder._name = "Decoder"
        self.Decoder.summary()

        self.Autoencoder = tf.keras.Sequential([self.Encoder, self.Decoder])
        self.Autoencoder._name = "Autoencoder"

        self.Autoencoder.compile(loss='mse', optimizer='adam')

        # self.Autoencoder.fit(trainData, trainData, epochs = epochs, verbose = 1, validation_data = (testData, testData), batch_size=32)

    def setupAutoencoder_KerasTuner(self, hp):
        # define a recurrent network with Gated Recurrent Units

        input_dim = self.arOrder
        output_dim = self.outputDim

        encoder_input = tf.keras.layers.Input(shape=(1,))

        encoder_output = encoder_input
        encoder_output = tf.keras.layers.RepeatVector(input_dim)(encoder_output)
        encoder_output = tf.keras.layers.Flatten()(encoder_output)
        encoder_output = tf.keras.layers.Dense(output_dim)(encoder_output)
        encoder_output = tf.keras.layers.Reshape((output_dim, 1))(encoder_output)

        self.Encoder = tf.keras.Model(inputs=encoder_input, outputs=encoder_output)
        self.Encoder._name = "Encoder"

        self.Encoder.summary()

        decoder_input = tf.keras.layers.Input(shape=(output_dim, 1,))

        decoder_output = tf.keras.layers.GRU(1)(decoder_input)
        decoder_output = tf.keras.layers.Dense(1)(decoder_output)

        self.Decoder = tf.keras.Model(inputs=decoder_input, outputs=decoder_output)
        self.Decoder._name = "Decoder"
        self.Decoder.summary()

        self.Autoencoder = tf.keras.Sequential([self.Encoder, self.Decoder])
        self.Autoencoder._name = "Autoencoder"

        self.Autoencoder.compile(loss='mse', optimizer='adam')

        # self.Autoencoder.fit(trainData, trainData, epochs = epochs, verbose = 1, validation_data = (testData, testData), batch_size=32)


    def featureSelectionRF(self, dataTrain, dataTest):
        rf_model = RandomForestRegressor(random_state = 0)
        rf_model.fit(dataTrain, dataTest.flatten())
        # sel = SelectFromModel(RandomForestClassifier(n_estimators = 100))
        # sel.fit(dataTrain, dataTest)
        features = range(0, dataTrain.shape[1])

        f_i = list(zip(features, rf_model.feature_importances_))
        f_i.sort(key=lambda x: x[1])
        plt.barh([x[0] for x in f_i], [x[1] for x in f_i])

        plt.show()

        rfe = RFECV(rf_model, cv=5, scoring="neg_mean_squared_error")

        rfe.fit(dataTrain, dataTest)

        return rfe.get_support()


    def buildP1Model_KerasTuner(self, hp):
        model_P1_input = tf.keras.layers.Input(shape=(self.outputDim, 1,))

        model_P1_output = model_P1_input


        model_P1_output = tf.keras.layers.LSTM(hp.Choice('P1_L1_LSTMUnits', [60, 120, 240]), return_sequences=True)(model_P1_output)

        if hp.Boolean("P1_L1_dropoutBool"):
            model_P1_output = tf.keras.layers.Dropout(rate = hp.Choice('P1_L1_dropoutUnits', [0.1, 0.25, 0.5]))(model_P1_output)

        #if hp.Boolean("P1_L2_LSTMBool"):

        model_P1_output = tf.keras.layers.LSTM(hp.Choice('P1_L2_LSTMUnits', [60, 120, 240]), return_sequences=False)(model_P1_output)

        model_P1_output = tf.keras.layers.Dense(1)(model_P1_output)

        model_P1 = tf.keras.Model(inputs=model_P1_input, outputs=model_P1_output)
        model_P1._name = "Model_P1"

        model_P1.compile(loss='mean_squared_error', optimizer='adam')
        model_P1.summary()

        return model_P1


    def buildP16Model_KerasTuner(self, hp):
        model_P16_input = tf.keras.layers.Input(shape=(self.outputDim, 1,))

        model_P16_output = model_P16_input


        model_P16_output = tf.keras.layers.LSTM(hp.Choice('P16_L1_LSTMUnits', [60, 120, 240]), return_sequences=True)(model_P16_output)

        if hp.Boolean("P16_L1_dropoutBool"):
            model_P16_output = tf.keras.layers.Dropout(rate = hp.Choice('P16_L1_dropoutUnits', [0.1, 0.25, 0.5]))(model_P16_output)

        #if hp.Boolean("P16_L2_LSTMBool"):

        model_P16_output = tf.keras.layers.LSTM(hp.Choice('P16_L2_LSTMUnits', [60, 120, 240]), return_sequences=False)(model_P16_output)

        model_P16_output = tf.keras.layers.Dense(1)(model_P16_output)

        model_P16 = tf.keras.Model(inputs=model_P16_input, outputs=model_P16_output)
        model_P16._name = "Model_P2"

        model_P16.compile(loss='mean_squared_error', optimizer='adam')
        model_P16.summary()

        return model_P16


    def buildARRNN_KerasTuner(self, hp):
        for layer in self.Encoder.layers:
            layer.trainable = False

        model_P1 = self.buildP1Model_KerasTuner(hp)
        model_P16 = self.buildP16Model_KerasTuner(hp)

        if False: #hp.Boolean("ARRNN_preFitP1P16"):
            model_P1.fit(self.trainData['Close'].values[:-1], self.trainData['Close'].values[1:], epochs = 20, validation_data = (self.testData['Close'].values[:-1], self.testData['Close'].values[1:]), batch_size=1024)
            model_P16.fit(self.trainData['Close'].values[:-16], self.trainData['Close'].values[16:], epochs = 20, validation_data = (self.testData['Close'].values[:-16], self.testData['Close'].values[16:]), batch_size=1024)

            for layer in model_P1.layers:
                layer.trainable = False

            for layer in model_P16.layers:
                layer.trainable = False

        self.Encoder._name = "Encoder_P1"
        model_P1_c = tf.keras.Sequential([self.Encoder, model_P1])

        self.Encoder._name = "Encoder_P16"
        model_P16_c = tf.keras.Sequential([self.Encoder, model_P16])

        model_P1P16 = tf.keras.layers.concatenate([model_P1_c.output, model_P16_c.output])

        model_out = model_P1P16

        if hp.Boolean("ARRNN_TargetFNN_RNNLayer"):
            model_out = tf.keras.layers.RepeatVector(15)(model_out)
            model_out = tf.keras.layers.GRU(5, return_sequences=True)(model_out)
            model_out = tf.keras.layers.GRU(1, return_sequences=False)(model_out)

        model_out = tf.keras.layers.Dense(hp.Choice('ARRNN_TargetFNN_FL1Units', [60, 120, 240]), activation=hp.Choice('ARRNN_TargetFNN_FL1Activation', ["relu", "tanh", "sigmoid", "linear"]))(model_out)
        model_out = tf.keras.layers.Dense(hp.Choice('ARRNN_TargetFNN_FL2Units', [60, 120, 240]), activation=hp.Choice('ARRNN_TargetFNN_FL2Activation', ["relu","tanh", "sigmoid", "linear"]))(model_out)
        model_out = tf.keras.layers.Dense(1, activation="linear")(model_out)

        learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")

        model = tf.keras.Model(inputs=[model_P1_c.input, model_P16_c.input], outputs=model_out)
        model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))

        return model


    def buildP1Model(self, config):
        model_P1_input = tf.keras.layers.Input(shape=(self.outputDim, 1,))

        model_P1_output = model_P1_input

        model_P1_output = tf.keras.layers.LSTM(config['P1_L1_LSTMUnits'], return_sequences=True)(
            model_P1_output)

        if config["P1_L1_dropoutBool"]:
            model_P1_output = tf.keras.layers.Dropout(rate=config['P1_L1_dropoutUnits'])(
                model_P1_output)

        if True: #config["P1_L2_LSTMBool"]:
            # TODO: Research was mehrere RNN Layer machen
            # TODO: Konsolidationseffekt? Ist dann Encoder überhaupt vernünftig?
            model_P1_output = tf.keras.layers.LSTM(config['P1_L2_LSTMUnits'],
                                                   return_sequences=False)(model_P1_output)

        model_P1_output = tf.keras.layers.Dense(1)(model_P1_output)

        model_P1 = tf.keras.Model(inputs=model_P1_input, outputs=model_P1_output)
        model_P1._name = "Model_P1"

        model_P1.compile(loss='mean_squared_error', optimizer='adam')
        model_P1.summary()

        return model_P1


    def buildP16Model(self, config):
        model_P16_input = tf.keras.layers.Input(shape=(self.outputDim, 1,))

        model_P16_output = model_P16_input

        model_P16_output = tf.keras.layers.LSTM(config['P16_L1_LSTMUnits'], return_sequences=True)(
            model_P16_output)

        if config["P16_L1_dropoutBool"]:
            model_P16_output = tf.keras.layers.Dropout(rate=config['P16_L1_dropoutUnits'])(
                model_P16_output)

        if True: #config["P16_L2_LSTMBool"]:
            model_P16_output = tf.keras.layers.LSTM(config['P16_L2_LSTMUnits'],
                                                    return_sequences=False)(model_P16_output)

        model_P16_output = tf.keras.layers.Dense(1)(model_P16_output)

        model_P16 = tf.keras.Model(inputs=model_P16_input, outputs=model_P16_output)
        model_P16._name = "Model_P16"

        model_P16.compile(loss='mean_squared_error', optimizer='adam')
        model_P16.summary()

        return model_P16


    def buildARRNN(self, config):
        for layer in self.Encoder.layers:
            layer.trainable = False

        model_P1 = self.buildP1Model(config)
        model_P16 = self.buildP16Model(config)

        if False: #config["ARRNN_preFitP1P16"]:
            model_P1.fit(self.trainData['Close'].values[:-1], self.trainData['Close'].values[1:], epochs=20,
                         validation_data=(self.testData['Close'].values[:-1], self.testData['Close'].values[1:]),
                         batch_size=1024)
            model_P16.fit(self.trainData['Close'].values[:-16], self.trainData['Close'].values[16:], epochs=20,
                          validation_data=(self.testData['Close'].values[:-16], self.testData['Close'].values[16:]),
                          batch_size=1024)

            for layer in model_P1.layers:
                layer.trainable = False

            for layer in model_P16.layers:
                layer.trainable = False

        self.Encoder._name = "Encoder_P1"
        model_P1_c = tf.keras.Sequential([self.Encoder, model_P1])

        self.Encoder._name = "Encoder_P16"
        model_P16_c = tf.keras.Sequential([self.Encoder, model_P16])

        model_P1P16 = tf.keras.layers.concatenate([model_P1_c.output, model_P16_c.output])

        model_out = model_P1P16

        if config["ARRNN_TargetFNN_RNNLayer"]:
            model_out = tf.keras.layers.RepeatVector(15)(model_out)
            model_out = tf.keras.layers.GRU(5, return_sequences=True)(model_out)
            model_out = tf.keras.layers.GRU(1, return_sequences=False)(model_out)

        model_out = tf.keras.layers.Dense(config['ARRNN_TargetFNN_FL1Units'],
                                          activation=config['ARRNN_TargetFNN_FL1Activation'])(model_out)
        model_out = tf.keras.layers.Dense(config['ARRNN_TargetFNN_FL2Units'],
                                          activation=config['ARRNN_TargetFNN_FL2Activation'])(model_out)
        model_out = tf.keras.layers.Dense(1, activation="linear")(model_out)

        learning_rate = config['lr']

        model = tf.keras.Model(inputs=[model_P1_c.input, model_P16_c.input], outputs=model_out)
        model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))

        return model


    def setARRNN_model(self, method = "Config", config = None, epochs = 20):
        self.setupAutoencoder(None)

        if self.dimRedMethod == "Average":
            dimRedRatio = int(self.arOrder / self.outputDim)

            weights = np.zeros((self.arOrder, self.outputDim))

            for j in range(self.outputDim):
                for i in range(dimRedRatio):
                    weights[(j * dimRedRatio) + i, j] = 1 / dimRedRatio

            bias = np.zeros(self.outputDim)

            self.Encoder.set_weights([weights, bias])

        elif self.dimRedMethod == "Autoencoder":
            #TODO: Implement Keras Tuner config and change the config dic accordingly
            self.Autoencoder.fit(self.trainDF['P1'], self.trainDF['P1'], epochs=20, validation_data=(self.testDF['P1'], self.testDF['P1']),
                            batch_size=1024)

        elif self.dimRedMethod == "None":
            None
            #TODO

        elif self.dimRedMethod == "RF":
            None
            #TODO

        else:
            print("Unknown Method")

        if method == "Tuner":
            import keras_tuner as kt

            tuner = kt.RandomSearch(self.buildARRNN_KerasTuner, objective='val_loss', max_trials=10)

            tuner.search(x = [self.trainDF['P1'], self.trainDF['P16']], y = self.trainDF['Target'], epochs = 10, validation_data=([self.testDF['P1'], self.testDF['P16']], self.testDF['Target']),
                         batch_size = 1024)
            self.ARRNN_model = tuner.get_best_models()[0]
            self.tuner = tuner

        elif method == "Config":
            if config is None:
                config = {
                        # Model P1
                        'P1_L1_LSTMUnits': 240,
                        "P1_L1_dropoutBool": True,
                        'P1_L1_dropoutUnits': 0.25,
                        "P1_L2_LSTMBool": True,
                        'P1_L2_LSTMUnits': 120,
                        # Model P16
                        'P16_L1_LSTMUnits': 240,
                        "P16_L1_dropoutBool": True,
                        'P16_L1_dropoutUnits': 0.25,
                        "P16_L2_LSTMBool": True,
                        'P16_L2_LSTMUnits': 120,
                        # Model TargetFNN
                        "ARRNN_preFitP1P16": True,
                        "ARRNN_TargetFNN_RNNLayer": True,
                        'ARRNN_TargetFNN_FL1Units': 120,
                        'ARRNN_TargetFNN_FL1Activation': "sigmoid",
                        'ARRNN_TargetFNN_FL2Units': 60,
                        'ARRNN_TargetFNN_FL2Activation': "sigmoid",

                        'lr': 1e-2 } # 0.00068464

            self.ARRNN_model = self.buildARRNN(config)
            history = self.ARRNN_model.fit(x = [self.trainDF['P1'], self.trainDF['P16']], y = self.trainDF['Target'], epochs = epochs, validation_data = ([self.testDF['P1'], self.testDF['P16']], self.testDF['Target']), batch_size=1024)

            # History Plot
            plt.plot(history.history['loss'], label='loss')
            plt.plot(history.history['val_loss'], label='validation_loss')
            plt.legend()
            plt.show()

        else:
            print("Unknown Method")


    def getARRNN_model(self):
        return self.ARRNN_model


    def getFittedData(self):
        Y_train_hat = self.ARRNN_model.predict([self.trainDF['P1'], self.trainDF['P16']])
        Y_test_hat = self.ARRNN_model.predict([self.testDF['P1'], self.testDF['P16']])
        Y_eval_hat = self.ARRNN_model.predict([self.evalDF['P1'], self.evalDF['P16']])

        return Y_train_hat, Y_test_hat, Y_eval_hat