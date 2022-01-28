import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFECV
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import time
from datetime import datetime


class AR_RNN_model:

    featureSet = None

    trainData, testData, evalData = None, None, None

    trainDF, testDF, evalDF = None, None, None

    dimRedMethod = None

    weights = None

    arOrder, outputDim, forecastSteps = None, None, None

    Encoder, Decoder, Autoencoder = None, None, None

    ARRNN_model = None

    coinID, coinName = None, None

    # Constructor, used to initialize the model object with all the necesarry parameters for the data preperation and
    # also starts the data preperation
    def __init__(self, data, arOrder, forecastSteps, coinID, dimRedMethod, outputDim = None, dimRedRatio = None, trainStart = "01/05/2021", evalStart = "01/06/2021"):  #timestamps = (1622505660 - (600000 * 0.5), 1622505660)

        totimestamp = lambda s: np.int32(time.mktime(datetime.strptime(s, "%d/%m/%Y").timetuple()))

        timestamps = (totimestamp(trainStart), totimestamp(evalStart))

        self.arOrder = arOrder
        self.forecastSteps = forecastSteps
        self.coinID = coinID

        self.setupData(data, coinID, timestamps)
        self.generateFeatureSet()

        self.dimRedMethod = dimRedMethod

        if dimRedMethod != "None":
            self.setOutputDim(outputDim, dimRedRatio)
        else:
            self.outputDim = arOrder


    # Function to handle the logic of dim reduction, i.e. test if the desired output dimension is even possible,
    # determine the reduction ratio (required for the average approach etc.)
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


    # Handles the data preparation, i.e. selecting the desired coin, selecting the desired range of data,
    # splitting the dataset into train, test and evaluation datasets, data cleaning like dripping NAs and dropping
    # not required columns
    def setupData(self, data, coinID, timestamps, trainFraction = 0.7):
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


    # Handels the preparation of the three different featuresets, and combine X and Y data into a easy and accesible
    # dataframe (including selection of the correct time frames)
    def generateFeatureSet(self):
        self.scaler_X = StandardScaler()
        self.scaler_Y = StandardScaler()

        trainData_X = self.trainData['Close'].diff().values.reshape(-1, 1)
        testData_X = self.testData['Close'].diff().values.reshape(-1, 1)
        evalData_X = self.evalData['Close'].diff().values.reshape(-1, 1)

        trainData_X = self.scaler_X.fit_transform(trainData_X).flatten()[1:]
        testData_X = self.scaler_X.transform(testData_X).flatten()[1:]
        evalData_X = self.scaler_X.transform(evalData_X).flatten()[1:]

        trainData_Y = self.trainData['Target'].values.reshape(-1, 1)
        testData_Y = self.testData['Target'].values.reshape(-1, 1)
        evalData_Y = self.evalData['Target'].values.reshape(-1, 1)

        trainData_Y = self.scaler_Y.fit_transform(trainData_Y).flatten()[1:]
        testData_Y = self.scaler_Y.transform(testData_Y).flatten()[1:]
        evalData_Y = self.scaler_Y.transform(evalData_Y).flatten()[1:]

        self.trainDF = pd.DataFrame(np.column_stack([trainData_X[15:-1], trainData_X[:-16], trainData_Y[16:]]))
        self.trainDF.columns = ['P1', 'P16', 'Target']

        self.testDF = pd.DataFrame(np.column_stack([testData_X[15:-1], testData_X[:-16], testData_Y[16:]]))
        self.testDF.columns = ['P1', 'P16', 'Target']

        self.evalDF = pd.DataFrame(np.column_stack([evalData_X[15:-1], evalData_X[:-16], evalData_Y[16:]]))
        self.evalDF.columns = ['P1', 'P16', 'Target']


    # Function which sets up the autoencoder required to reduce the feature matrix, aka the matrix with the sequences
    def setupAutoencoder(self, config, input_dim = None, output_dim = None):
        # define a recurrent network with Gated Recurrent Units

        if input_dim is None:
            input_dim = self.arOrder

        if output_dim is None:
            output_dim = self.outputDim

        encoder_input = tf.keras.layers.Input(shape=(1,))

        encoder_output = encoder_input
        encoder_output = tf.keras.layers.RepeatVector(input_dim)(encoder_output)
        encoder_output = tf.keras.layers.Flatten()(encoder_output)
        encoder_output = tf.keras.layers.Dense(output_dim, activation = "linear")(encoder_output)

        #TODO: Ausprobieren GRU
        #TODO: Research machen ob Encoder/Decoder symetrisch aufgebaut sein müssen

        encoder_output = tf.keras.layers.Reshape((output_dim, 1))(encoder_output)

        self.Encoder = tf.keras.Model(inputs=encoder_input, outputs=encoder_output)
        self.Encoder._name = "Encoder"

        self.Encoder.summary()

        decoder_input = tf.keras.layers.Input(shape=(output_dim, 1,))
        decoder_output = decoder_input
        decoder_output = tf.keras.layers.Flatten()(decoder_output)
        #decoder_output = tf.keras.layers.GRU(1)(decoder_output)
        decoder_output = tf.keras.layers.Dense(1, activation = "linear")(decoder_output)

        self.Decoder = tf.keras.Model(inputs=decoder_input, outputs=decoder_output)
        self.Decoder._name = "Decoder"
        self.Decoder.summary()

        self.Autoencoder = tf.keras.Sequential([self.Encoder, self.Decoder])
        self.Autoencoder._name = "Autoencoder"

        self.Autoencoder.compile(loss='mse', optimizer='adam')


    # Function which sets up the autoencoder, this one is desinged be used with the KerasTuner
    def setupAutoencoder_KerasTuner(self, hp, input_dim = None, output_dim = None):
        # define a recurrent network with Gated Recurrent Units

        if input_dim is None:
            input_dim = self.arOrder

        if output_dim is None:
            output_dim = self.outputDim

        encoder_input = tf.keras.layers.Input(shape=(1,))

        encoder_output = encoder_input
        encoder_output = tf.keras.layers.RepeatVector(input_dim)(encoder_output)
        encoder_output = tf.keras.layers.Flatten()(encoder_output)
        encoder_output = tf.keras.layers.Dense(output_dim, activation = "linear")(encoder_output)
        encoder_output = tf.keras.layers.Reshape((output_dim, 1))(encoder_output)

        self.Encoder = tf.keras.Model(inputs=encoder_input, outputs=encoder_output)
        self.Encoder._name = "Encoder"

        self.Encoder.summary()

        decoder_input = tf.keras.layers.Input(shape=(output_dim, 1,))
        decoder_output = decoder_input
        decoder_output = tf.keras.layers.Flatten()(decoder_output)
        decoder_output = tf.keras.layers.Dense(1, activation = "linear")(decoder_output)

        self.Decoder = tf.keras.Model(inputs=decoder_input, outputs=decoder_output)
        self.Decoder._name = "Decoder"
        self.Decoder.summary()

        self.Autoencoder = tf.keras.Sequential([self.Encoder, self.Decoder])
        self.Autoencoder._name = "Autoencoder"

        self.Autoencoder.compile(loss='mse', optimizer='adam')

    # Function for feature selection via a RandomForest (depricated)
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

    # Function which sets up the Model for prediction the information equivalent of P_t+1, should be used in the context
    # of the KarasTuner which required a HyperParameter object as parameter
    def buildP1Model_KerasTuner(self, hp):
        model_P1_input = tf.keras.layers.Input(shape=(self.outputDim, 1,))

        model_P1_output = model_P1_input

        # RNN 1
        if self.modelType == "GRU":
            model_P1_output = tf.keras.layers.GRU(hp.Choice('P1_L1_RNNUnits', [60, 120, 240]), return_sequences=True)(model_P1_output)
        else:
            model_P1_output = tf.keras.layers.LSTM(hp.Choice('P1_L1_RNNUnits', [60, 120, 240]), return_sequences=True)(model_P1_output)

        # Droput RNN 1
        if hp.Boolean("P1_L1_dropoutBool"):
            model_P1_output = tf.keras.layers.Dropout(rate = hp.Choice('P1_L1_dropoutUnits', [0.1, 0.25, 0.5], parent_name="P1_L1_dropoutBool", parent_values = True))(model_P1_output)

        # RNN 2
        if self.modelType == "GRU":
            model_P1_output = tf.keras.layers.GRU(hp.Choice('P1_L2_RNNUnits', [60, 120, 240]), return_sequences=False)(model_P1_output)
        else:
            model_P1_output = tf.keras.layers.LSTM(hp.Choice('P1_L2_RNNUnits', [60, 120, 240]), return_sequences=False)(model_P1_output)

        model_P1_output = tf.keras.layers.Dense(1)(model_P1_output)

        model_P1 = tf.keras.Model(inputs=model_P1_input, outputs=model_P1_output)
        model_P1._name = "Model_P1"

        model_P1.compile(loss='mean_squared_error', optimizer='adam')
        model_P1.summary()

        return model_P1

    # Function which sets up the Model for prediction the information equivalent of P_t+16, should be used in the context
    # of the KarasTuner which required a HyperParameter object as parameter
    def buildP16Model_KerasTuner(self, hp):
        model_P16_input = tf.keras.layers.Input(shape=(self.outputDim, 1,))

        model_P16_output = model_P16_input

        # RNN 1
        if self.modelType == "GRU":
            model_P16_output = tf.keras.layers.LSTM(hp.Choice('P16_L1_RNNUnits', [60, 120, 240]), return_sequences=True)(model_P16_output)
        else:
            model_P16_output = tf.keras.layers.GRU(hp.Choice('P16_L1_RNNUnits', [60, 120, 240]),return_sequences=True)(model_P16_output)

        # Drouput RNN 1
        if hp.Boolean("P16_L1_dropoutBool"):
            model_P16_output = tf.keras.layers.Dropout(rate = hp.Choice('P16_L1_dropoutUnits', [0.1, 0.25, 0.5], parent_name="P16_L1_dropoutBool", parent_values = True))(model_P16_output)

        # RNN 2
        if self.modelType == "GRU":
            model_P16_output = tf.keras.layers.GRU(hp.Choice('P16_L2_RNNUnits', [60, 120, 240]), return_sequences=False)(model_P16_output)
        else:
            model_P16_output = tf.keras.layers.LSTM(hp.Choice('P16_L2_RNNUnits', [60, 120, 240]),
                                                   return_sequences=False)(model_P16_output)

        model_P16_output = tf.keras.layers.Dense(1)(model_P16_output)

        model_P16 = tf.keras.Model(inputs=model_P16_input, outputs=model_P16_output)
        model_P16._name = "Model_P2"

        model_P16.compile(loss='mean_squared_error', optimizer='adam')
        model_P16.summary()

        return model_P16

    # Function which combines the price models with the autoencoder and combines the outputs of the price models and
    # adds a FNN on top to approximate the functional relationship between price forecasts and target variable
    def buildARRNN_KerasTuner(self, hp):
        learning_rate_priceModels = hp.Float("lr_priceModels", min_value=1e-4, max_value=1e-2, sampling="log")
        learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")

        for layer in self.Encoder.layers:
            layer.trainable = False

        model_P1 = self.buildP1Model_KerasTuner(hp)
        model_P16 = self.buildP16Model_KerasTuner(hp)

        self.Encoder._name = "Encoder_P1"
        model_P1_c = tf.keras.Sequential([self.Encoder, model_P1])

        self.Encoder._name = "Encoder_P16"
        model_P16_c = tf.keras.Sequential([self.Encoder, model_P16])

        print("---- PreTraining P1/p16 Model ----")
        model_P1_c.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate_priceModels))
        model_P16_c.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate_priceModels))

        model_P1_c.fit(self.trainData['Close'].values[:-1], self.trainData['Close'].values[1:], epochs = 20, validation_data = (self.testData['Close'].values[:-1], self.testData['Close'].values[1:]), batch_size=1024)
        model_P16_c.fit(self.trainData['Close'].values[:-16], self.trainData['Close'].values[16:], epochs = 20, validation_data = (self.testData['Close'].values[:-16], self.testData['Close'].values[16:]), batch_size=1024)

        for layer in model_P1_c.layers:
            layer.trainable = False

        for layer in model_P16_c.layers:
            layer.trainable = False

        model_P1P16 = tf.keras.layers.concatenate([model_P1_c.output, model_P16_c.output])

        model_out = model_P1P16

        if hp.Boolean("ARRNN_TargetFNN_RNNLayer"):
            model_out = tf.keras.layers.RepeatVector(60)(model_out)
            if self.modelType == "GRU":
                model_out = tf.keras.layers.GRU(5, return_sequences=True)(model_out)
                model_out = tf.keras.layers.GRU(1, return_sequences=False)(model_out)
            else:
                model_out = tf.keras.layers.LSTM(5, return_sequences=True)(model_out)
                model_out = tf.keras.layers.LSTM(1, return_sequences=False)(model_out)

        model_out = tf.keras.layers.Dense(hp.Choice('ARRNN_TargetFNN_FL1Units', [60, 120, 240]), activation=hp.Choice('ARRNN_TargetFNN_FL1Activation', ["relu", "tanh", "linear"]))(model_out)
        model_out = tf.keras.layers.Dense(hp.Choice('ARRNN_TargetFNN_FL2Units', [60, 120, 240]), activation=hp.Choice('ARRNN_TargetFNN_FL2Activation', ["relu","tanh", "linear"]))(model_out)
        model_out = tf.keras.layers.Dense(1, activation="linear")(model_out)

        model = tf.keras.Model(inputs=[model_P1_c.input, model_P16_c.input], outputs=model_out)
        model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))

        return model


    def buildP1Model(self, config):
        model_P1_input = tf.keras.layers.Input(shape=(self.outputDim, 1,))

        model_P1_output = model_P1_input

        # RNN 1
        if self.modelType == "GRU":
            model_P1_output = tf.keras.layers.GRU(config['P1_L1_RNNUnits'], return_sequences=True)(model_P1_output)
        else:
            model_P1_output = tf.keras.layers.LSTM(config['P1_L1_RNNUnits'], return_sequences=True)(model_P1_output)

        # Dropout RNN 1
        if config["P1_L1_dropoutBool"]:
            model_P1_output = tf.keras.layers.Dropout(rate=config['P1_L1_dropoutUnits'])(
                model_P1_output)

        # RNN 2
        if self.modelType == "GRU":
            model_P1_output = tf.keras.layers.GRU(config['P1_L2_RNNUnits'],return_sequences=False)(model_P1_output)
        else:
            model_P1_output = tf.keras.layers.LSTM(config['P1_L2_RNNUnits'], return_sequences=False)(model_P1_output)


        model_P1_output = tf.keras.layers.Dense(1)(model_P1_output)

        model_P1 = tf.keras.Model(inputs=model_P1_input, outputs=model_P1_output)
        model_P1._name = "Model_P1"

        model_P1.compile(loss='mean_squared_error', optimizer='adam')
        model_P1.summary()

        return model_P1


    def buildP16Model(self, config):
        model_P16_input = tf.keras.layers.Input(shape=(self.outputDim, 1,))

        model_P16_output = model_P16_input

        # RNN 1
        if self.modelType == "GRU":
            model_P16_output = tf.keras.layers.GRU(config['P16_L1_RNNUnits'], return_sequences=True)(model_P16_output)
        else:
            model_P16_output = tf.keras.layers.LSTM(config['P16_L1_RNNUnits'], return_sequences=True)(model_P16_output)

        # Dropout RNN 1
        if config["P16_L1_dropoutBool"]:
            model_P16_output = tf.keras.layers.Dropout(rate=config['P16_L1_dropoutUnits'])(
                model_P16_output)

        # RNN 2
        if self.modelType == "GRU":
            model_P16_output = tf.keras.layers.GRU(config['P16_L2_RNNUnits'],return_sequences=False)(model_P16_output)
        else:
            model_P16_output = tf.keras.layers.LSTM(config['P16_L2_RNNUnits'], return_sequences=False)(model_P16_output)

        model_P16_output = tf.keras.layers.Dense(1)(model_P16_output)

        model_P16 = tf.keras.Model(inputs=model_P16_input, outputs=model_P16_output)
        model_P16._name = "Model_P16"

        model_P16.compile(loss='mean_squared_error', optimizer='adam')
        model_P16.summary()

        return model_P16


    def buildARRNN(self, config):
        learning_rate_priceModels = config["lr_priceModels"]
        learning_rate = config['lr']

        for layer in self.Encoder.layers:
            layer.trainable = False

        model_P1 = self.buildP1Model(config)
        model_P16 = self.buildP16Model(config)

        self.Encoder._name = "Encoder_P1"
        model_P1_c = tf.keras.Sequential([self.Encoder, model_P1])

        self.Encoder._name = "Encoder_P16"
        model_P16_c = tf.keras.Sequential([self.Encoder, model_P16])

        print("---- PreTraining P1/p16 Model ----")
        model_P1_c.compile(loss='mean_squared_error',
                           optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate_priceModels))
        model_P16_c.compile(loss='mean_squared_error',
                            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate_priceModels))

        model_P1_c.fit(self.trainData['Close'].values[:-1], self.trainData['Close'].values[1:], epochs=20,
                     validation_data=(self.testData['Close'].values[:-1], self.testData['Close'].values[1:]),
                     batch_size=1024)
        model_P16_c.fit(self.trainData['Close'].values[:-16], self.trainData['Close'].values[16:], epochs=20,
                      validation_data=(self.testData['Close'].values[:-16], self.testData['Close'].values[16:]),
                      batch_size=1024)

        for layer in model_P1_c.layers:
            layer.trainable = False

        for layer in model_P16_c.layers:
            layer.trainable = False

        model_P1P16 = tf.keras.layers.concatenate([model_P1_c.output, model_P16_c.output])

        model_out = model_P1P16

        if config["ARRNN_TargetFNN_RNNLayer"]:
            model_out = tf.keras.layers.RepeatVector(60)(model_out)
            if self.modelType == "GRU":
                model_out = tf.keras.layers.GRU(5, return_sequences=True)(model_out)
                model_out = tf.keras.layers.GRU(1, return_sequences=False)(model_out)
            else:
                model_out = tf.keras.layers.LSTM(5, return_sequences=True)(model_out)
                model_out = tf.keras.layers.LSTM(1, return_sequences=False)(model_out)

        model_out = tf.keras.layers.Dense(config['ARRNN_TargetFNN_FL1Units'],
                                          activation=config['ARRNN_TargetFNN_FL1Activation'])(model_out)
        model_out = tf.keras.layers.Dense(config['ARRNN_TargetFNN_FL2Units'],
                                          activation=config['ARRNN_TargetFNN_FL2Activation'])(model_out)
        model_out = tf.keras.layers.Dense(1, activation="linear")(model_out)

        model = tf.keras.Model(inputs=[model_P1_c.input, model_P16_c.input], outputs=model_out)
        corr_loss = lambda y_true, y_pred: 1 - np.abs(np.corrcoef(y_true.flatten(), y_pred.flatten())[1, 0])

        model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
        #model.compile(loss=corr_loss, optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))

        model.summary()
        tf.keras.utils.plot_model(model)

        return model

    # function which handels all the steps necessary for model building and estimation. Should be accesed from the "outside"
    def setARRNN_model(self, method = "Config", config = None, epochs = 20, modelType = "GRU"):
        self.modelType = modelType
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
            self.Autoencoder.fit(self.trainDF['P1'], self.trainDF['P1'], epochs=10, validation_data=(self.testDF['P1'], self.testDF['P1']),
                            batch_size=1024)

        elif self.dimRedMethod == "None":
            #self.setupAutoencoder(None, output_dim=self.arOrder)
            weights = np.zeros((self.arOrder, self.arOrder))

            for j in range(self.outputDim):
                weights[j, j] = 1

            bias = np.zeros(self.outputDim)

            self.Encoder.set_weights([weights, bias])

        else:
            print("Unknown Method")

        if method == "Tuner":
            import keras_tuner as kt

            #tuner = kt.RandomSearch(self.buildARRNN_KerasTuner, objective='val_loss', max_trials=10)
            tuner = kt.BayesianOptimization(self.buildARRNN_KerasTuner, objective='val_loss', max_trials=5, overwrite=True, project_name="ARRNN_tune")

            tuner.search(x = [self.trainDF['P1'], self.trainDF['P16']], y = self.trainDF['Target'], epochs = 10, validation_data=([self.testDF['P1'], self.testDF['P16']], self.testDF['Target']),
                         batch_size = 1024)
            self.ARRNN_model = tuner.get_best_models()[0]
            self.tuner = tuner

            history = self.ARRNN_model.fit(x=[self.trainDF['P1'], self.trainDF['P16']], y=self.trainDF['Target'],
                                           epochs=epochs, validation_data=([self.testDF['P1'], self.testDF['P16']], self.testDF['Target']), batch_size=1024)

            # History Plot
            plt.plot(history.history['loss'], label='loss')
            plt.plot(history.history['val_loss'], label='validation_loss')
            plt.legend()
            plt.show()

        elif method == "Config":
            if config is None:
                config = {
                        # Model P1
                        'P1_L1_RNNUnits': 240,
                        "P1_L1_dropoutBool": True,
                        'P1_L1_dropoutUnits': 0.25,
                        'P1_L2_RNNUnits': 120,
                        # Model P16
                        'P16_L1_RNNUnits': 240,
                        "P16_L1_dropoutBool": True,
                        'P16_L1_dropoutUnits': 0.25,
                        'P16_L2_RNNUnits': 120,
                        # Model TargetFNN
                        "ARRNN_TargetFNN_RNNLayer": True,
                        'ARRNN_TargetFNN_FL1Units': 120,
                        'ARRNN_TargetFNN_FL1Activation': "tanh",
                        'ARRNN_TargetFNN_FL2Units': 60,
                        'ARRNN_TargetFNN_FL2Activation': "tanh",

                        'lr': 1e-2,
                        "lr_priceModels": 1e-2} # 0.00068464

            self.ARRNN_model = self.buildARRNN(config)
            history = self.ARRNN_model.fit(x = [self.trainDF['P1'], self.trainDF['P16']], y = self.trainDF['Target'], epochs = epochs, validation_data = ([self.testDF['P1'], self.testDF['P16']], self.testDF['Target']), batch_size=1024)

            # History Plot
            plt.plot(history.history['loss'], label='loss')
            plt.plot(history.history['val_loss'], label='validation_loss')
            plt.legend()
            plt.show()

        else:
            print("Unknown Method")

    # function which returns the estimated model
    def getARRNN_model(self):
        return self.ARRNN_model

    # function which returns the predicted values for the training, test and eval time series
    def getFittedData(self, scaled = True):
        Y_train_hat = self.ARRNN_model.predict([self.trainDF['P1'], self.trainDF['P16']])
        Y_test_hat = self.ARRNN_model.predict([self.testDF['P1'], self.testDF['P16']])
        Y_eval_hat = self.ARRNN_model.predict([self.evalDF['P1'], self.evalDF['P16']])

        if not scaled:
            Y_train_hat = self.scaler_Y.inverse_transform(Y_train_hat)
            Y_test_hat = self.scaler_Y.inverse_transform(Y_test_hat)
            Y_eval_hat = self.scaler_Y.inverse_transform(Y_eval_hat)

        return Y_train_hat, Y_test_hat, Y_eval_hat

    # function which returns the predicted values for the training, test and eval time series
    def getFittedTrainData(self, scaled=True):
        Y_train_hat = self.ARRNN_model.predict([self.trainDF['P1'], self.trainDF['P16']])

        if not scaled:
            Y_train_hat = self.scaler_Y.inverse_transform(Y_train_hat)

        return Y_train_hat

    # function which returns the predicted values for the training, test and eval time series
    def getFittedTestData(self, scaled=True):
        Y_test_hat = self.ARRNN_model.predict([self.testDF['P1'], self.testDF['P16']])

        if not scaled:
            Y_test_hat = self.scaler_Y.inverse_transform(Y_test_hat)

        return Y_test_hat

    # function which returns the predicted values for the training, test and eval time series
    def getFittedEvalData(self, scaled=True):
        Y_eval_hat = self.ARRNN_model.predict([self.evalDF['P1'], self.evalDF['P16']])

        if not scaled:
            Y_eval_hat = self.scaler_Y.inverse_transform(Y_eval_hat)

        return Y_eval_hat