#%%
from Scr.AR_RNN_model_2 import AR_RNN_model
from Scr.functions import import_data
from Scr.functions import performanceEval
from Scr.functions import corr
from Scr.functions import Bias
from Scr.functions import Var
from Scr.functions import MSE

import numpy as np

data = import_data("Data/")[0]

rep_times = 10

# CoinID 4

#%%
### Autoencoder LSTM
result_dic_autoencoderDogge = {}

result_array_TrainCorr = []
result_array_TrainBias = []
result_array_TrainVar = []
result_array_TrainMSE = []

result_array_TestCorr = []
result_array_TestBias = []
result_array_TestVar = []
result_array_TestMSE = []

for i in range(rep_times):
    print(f"---- Starting rep {i} ----")
    ARRNN_mod = AR_RNN_model(data, arOrder = 60 * 1, forecastSteps = 15, coinID = 4, dimRedMethod = 'Autoencoder', outputDim=60, trainStart = "25/05/2021", evalStart = "01/06/2021")
    
    ARRNN_mod.setARRNN_model(method = "Config", modelType="LSTM")
    
    Y_train_hat, Y_test_hat = ARRNN_mod.getFittedTrainData(scaled = True), ARRNN_mod.getFittedTestData(scaled = True)
    Y_train, Y_test = ARRNN_mod.trainDF['Target'], ARRNN_mod.testDF['Target']
    
    result_array_TrainCorr.append(corr(Y_train, Y_train_hat))
    result_array_TrainBias.append(Bias(Y_train, Y_train_hat))
    result_array_TrainVar.append(Var(Y_train, Y_train_hat))
    result_array_TrainMSE.append(MSE(Y_train, Y_train_hat))

    result_array_TestCorr.append(corr(Y_test, Y_test_hat))
    result_array_TestBias.append(Bias(Y_test, Y_test_hat))
    result_array_TestVar.append(Var(Y_test, Y_test_hat))
    result_array_TestMSE.append(MSE(Y_test, Y_test_hat))
    
result_dic_autoencoderDogge["LSTMTrainCorr"] = np.mean(result_array_TrainCorr)
result_dic_autoencoderDogge["LSTMTrainBias"] = np.mean(result_array_TrainBias)
result_dic_autoencoderDogge["LSTMTrainVar"] = np.mean(result_array_TrainVar)
result_dic_autoencoderDogge["LSTMTrainMSE"] = np.mean(result_array_TrainMSE)

result_dic_autoencoderDogge["LSTMTestCorr"] = np.mean(result_array_TestCorr)
result_dic_autoencoderDogge["LSTMTestBias"] = np.mean(result_array_TestBias)
result_dic_autoencoderDogge["LSTMTestVar"] = np.mean(result_array_TestVar)
result_dic_autoencoderDogge["LSTMTestMSE"] = np.mean(result_array_TestMSE)

### Autoencoder GRU
result_array_TrainCorr = []
result_array_TrainBias = []
result_array_TrainVar = []
result_array_TrainMSE = []

result_array_TestCorr = []
result_array_TestBias = []
result_array_TestVar = []
result_array_TestMSE = []

for i in range(rep_times):
    print(f"---- Starting rep {i} ----")
    ARRNN_mod = AR_RNN_model(data, arOrder=60 * 1, forecastSteps=15, coinID=4, dimRedMethod='Autoencoder', outputDim=60,
                             trainStart="25/05/2021", evalStart="01/06/2021")

    ARRNN_mod.setARRNN_model(method="Config", modelType="GRU")

    Y_train_hat, Y_test_hat = ARRNN_mod.getFittedTrainData(scaled=True), ARRNN_mod.getFittedTestData(scaled=True)
    Y_train, Y_test = ARRNN_mod.trainDF['Target'], ARRNN_mod.testDF['Target']

    result_array_TrainCorr.append(corr(Y_train, Y_train_hat))
    result_array_TrainBias.append(Bias(Y_train, Y_train_hat))
    result_array_TrainVar.append(Var(Y_train, Y_train_hat))
    result_array_TrainMSE.append(MSE(Y_train, Y_train_hat))

    result_array_TestCorr.append(corr(Y_test, Y_test_hat))
    result_array_TestBias.append(Bias(Y_test, Y_test_hat))
    result_array_TestVar.append(Var(Y_test, Y_test_hat))
    result_array_TestMSE.append(MSE(Y_test, Y_test_hat))

result_dic_autoencoderDogge["GRUTrainCorr"] = np.mean(result_array_TrainCorr)
result_dic_autoencoderDogge["GRUTrainBias"] = np.mean(result_array_TrainBias)
result_dic_autoencoderDogge["GRUTrainVar"] = np.mean(result_array_TrainVar)
result_dic_autoencoderDogge["GRUTrainMSE"] = np.mean(result_array_TrainMSE)

result_dic_autoencoderDogge["GRUTestCorr"] = np.mean(result_array_TestCorr)
result_dic_autoencoderDogge["GRUTestBias"] = np.mean(result_array_TestBias)
result_dic_autoencoderDogge["GRUTestVar"] = np.mean(result_array_TestVar)
result_dic_autoencoderDogge["GRUTestMSE"] = np.mean(result_array_TestMSE)

#%%
### Average LSTM
result_dic_averageDogge = {}

result_array_TrainCorr = []
result_array_TrainBias = []
result_array_TrainVar = []
result_array_TrainMSE = []

result_array_TestCorr = []
result_array_TestBias = []
result_array_TestVar = []
result_array_TestMSE = []

for i in range(rep_times):
    print(f"---- Starting rep {i} ----")
    ARRNN_mod = AR_RNN_model(data, arOrder=60 * 1, forecastSteps=15, coinID=4, dimRedMethod='Average', outputDim=60,
                             trainStart="25/05/2021", evalStart="01/06/2021")

    ARRNN_mod.setARRNN_model(method="Config", modelType="LSTM")

    Y_train_hat, Y_test_hat = ARRNN_mod.getFittedTrainData(scaled=True), ARRNN_mod.getFittedTestData(scaled=True)
    Y_train, Y_test = ARRNN_mod.trainDF['Target'], ARRNN_mod.testDF['Target']

    result_array_TrainCorr.append(corr(Y_train, Y_train_hat))
    result_array_TrainBias.append(Bias(Y_train, Y_train_hat))
    result_array_TrainVar.append(Var(Y_train, Y_train_hat))
    result_array_TrainMSE.append(MSE(Y_train, Y_train_hat))

    result_array_TestCorr.append(corr(Y_test, Y_test_hat))
    result_array_TestBias.append(Bias(Y_test, Y_test_hat))
    result_array_TestVar.append(Var(Y_test, Y_test_hat))
    result_array_TestMSE.append(MSE(Y_test, Y_test_hat))

result_dic_averageDogge["LSTMTrainCorr"] = np.mean(result_array_TrainCorr)
result_dic_averageDogge["LSTMTrainBias"] = np.mean(result_array_TrainBias)
result_dic_averageDogge["LSTMTrainVar"] = np.mean(result_array_TrainVar)
result_dic_averageDogge["LSTMTrainMSE"] = np.mean(result_array_TrainMSE)

result_dic_averageDogge["LSTMTestCorr"] = np.mean(result_array_TestCorr)
result_dic_averageDogge["LSTMTestBias"] = np.mean(result_array_TestBias)
result_dic_averageDogge["LSTMTestVar"] = np.mean(result_array_TestVar)
result_dic_averageDogge["LSTMTestMSE"] = np.mean(result_array_TestMSE)


### Average GRU
result_array_TrainCorr = []
result_array_TrainBias = []
result_array_TrainVar = []
result_array_TrainMSE = []

result_array_TestCorr = []
result_array_TestBias = []
result_array_TestVar = []
result_array_TestMSE = []

for i in range(rep_times):
    print(f"---- Starting rep {i} ----")
    ARRNN_mod = AR_RNN_model(data, arOrder=60 * 1, forecastSteps=15, coinID=4, dimRedMethod='Average', outputDim=60,
                             trainStart="25/05/2021", evalStart="01/06/2021")

    ARRNN_mod.setARRNN_model(method="Config", modelType="GRU")

    Y_train_hat, Y_test_hat = ARRNN_mod.getFittedTrainData(scaled=True), ARRNN_mod.getFittedTestData(scaled=True)
    Y_train, Y_test = ARRNN_mod.trainDF['Target'], ARRNN_mod.testDF['Target']

    result_array_TrainCorr.append(corr(Y_train, Y_train_hat))
    result_array_TrainBias.append(Bias(Y_train, Y_train_hat))
    result_array_TrainVar.append(Var(Y_train, Y_train_hat))
    result_array_TrainMSE.append(MSE(Y_train, Y_train_hat))

    result_array_TestCorr.append(corr(Y_test, Y_test_hat))
    result_array_TestBias.append(Bias(Y_test, Y_test_hat))
    result_array_TestVar.append(Var(Y_test, Y_test_hat))
    result_array_TestMSE.append(MSE(Y_test, Y_test_hat))

result_dic_averageDogge["GRUTrainCorr"] = np.mean(result_array_TrainCorr)
result_dic_averageDogge["GRUTrainBias"] = np.mean(result_array_TrainBias)
result_dic_averageDogge["GRUTrainVar"] = np.mean(result_array_TrainVar)
result_dic_averageDogge["GRUTrainMSE"] = np.mean(result_array_TrainMSE)

result_dic_averageDogge["GRUTestCorr"] = np.mean(result_array_TestCorr)
result_dic_averageDogge["GRUTestBias"] = np.mean(result_array_TestBias)
result_dic_averageDogge["GRUTestVar"] = np.mean(result_array_TestVar)
result_dic_averageDogge["GRUTestMSE"] = np.mean(result_array_TestMSE)



#%%
### None LSTM
result_dic_noneDogge = {}

result_array_TrainCorr = []
result_array_TrainBias = []
result_array_TrainVar = []
result_array_TrainMSE = []

result_array_TestCorr = []
result_array_TestBias = []
result_array_TestVar = []
result_array_TestMSE = []

for i in range(rep_times):
    print(f"---- Starting rep {i} ----")
    ARRNN_mod = AR_RNN_model(data, arOrder=60 * 1, forecastSteps=15, coinID=4, dimRedMethod='None', outputDim=60,
                             trainStart="25/05/2021", evalStart="01/06/2021")

    ARRNN_mod.setARRNN_model(method="Config", modelType="LSTM")

    Y_train_hat, Y_test_hat = ARRNN_mod.getFittedTrainData(scaled=True), ARRNN_mod.getFittedTestData(scaled=True)
    Y_train, Y_test = ARRNN_mod.trainDF['Target'], ARRNN_mod.testDF['Target']

    result_array_TrainCorr.append(corr(Y_train, Y_train_hat))
    result_array_TrainBias.append(Bias(Y_train, Y_train_hat))
    result_array_TrainVar.append(Var(Y_train, Y_train_hat))
    result_array_TrainMSE.append(MSE(Y_train, Y_train_hat))

    result_array_TestCorr.append(corr(Y_test, Y_test_hat))
    result_array_TestBias.append(Bias(Y_test, Y_test_hat))
    result_array_TestVar.append(Var(Y_test, Y_test_hat))
    result_array_TestMSE.append(MSE(Y_test, Y_test_hat))

result_dic_noneDogge["LSTMTrainCorr"] = np.mean(result_array_TrainCorr)
result_dic_noneDogge["LSTMTrainBias"] = np.mean(result_array_TrainBias)
result_dic_noneDogge["LSTMTrainVar"] = np.mean(result_array_TrainVar)
result_dic_noneDogge["LSTMTrainMSE"] = np.mean(result_array_TrainMSE)

result_dic_noneDogge["LSTMTestCorr"] = np.mean(result_array_TestCorr)
result_dic_noneDogge["LSTMTestBias"] = np.mean(result_array_TestBias)
result_dic_noneDogge["LSTMTestVar"] = np.mean(result_array_TestVar)
result_dic_noneDogge["LSTMTestMSE"] = np.mean(result_array_TestMSE)


### None GRU
result_array_TrainCorr = []
result_array_TrainBias = []
result_array_TrainVar = []
result_array_TrainMSE = []

result_array_TestCorr = []
result_array_TestBias = []
result_array_TestVar = []
result_array_TestMSE = []

for i in range(rep_times):
    print(f"---- Starting rep {i} ----")
    ARRNN_mod = AR_RNN_model(data, arOrder=60 * 1, forecastSteps=15, coinID=4, dimRedMethod='None', outputDim=60,
                             trainStart="25/05/2021", evalStart="01/06/2021")

    ARRNN_mod.setARRNN_model(method="Config", modelType="GRU")

    Y_train_hat, Y_test_hat = ARRNN_mod.getFittedTrainData(scaled=True), ARRNN_mod.getFittedTestData(scaled=True)
    Y_train, Y_test = ARRNN_mod.trainDF['Target'], ARRNN_mod.testDF['Target']

    result_array_TrainCorr.append(corr(Y_train, Y_train_hat))
    result_array_TrainBias.append(Bias(Y_train, Y_train_hat))
    result_array_TrainVar.append(Var(Y_train, Y_train_hat))
    result_array_TrainMSE.append(MSE(Y_train, Y_train_hat))

    result_array_TestCorr.append(corr(Y_test, Y_test_hat))
    result_array_TestBias.append(Bias(Y_test, Y_test_hat))
    result_array_TestVar.append(Var(Y_test, Y_test_hat))
    result_array_TestMSE.append(MSE(Y_test, Y_test_hat))

result_dic_noneDogge["GRUTrainCorr"] = np.mean(result_array_TrainCorr)
result_dic_noneDogge["GRUTrainBias"] = np.mean(result_array_TrainBias)
result_dic_noneDogge["GRUTrainVar"] = np.mean(result_array_TrainVar)
result_dic_noneDogge["GRUTrainMSE"] = np.mean(result_array_TrainMSE)

result_dic_noneDogge["GRUTestCorr"] = np.mean(result_array_TestCorr)
result_dic_noneDogge["GRUTestBias"] = np.mean(result_array_TestBias)
result_dic_noneDogge["GRUTestVar"] = np.mean(result_array_TestVar)
result_dic_noneDogge["GRUTestMSE"] = np.mean(result_array_TestMSE)




# CoinID 6

# %%
### Autoencoder LSTM
result_dic_autoencoderEther = {}

result_array_TrainCorr = []
result_array_TrainBias = []
result_array_TrainVar = []
result_array_TrainMSE = []

result_array_TestCorr = []
result_array_TestBias = []
result_array_TestVar = []
result_array_TestMSE = []

for i in range(rep_times):
    print(f"---- Starting rep {i} ----")
    ARRNN_mod = AR_RNN_model(data, arOrder=60 * 1, forecastSteps=15, coinID=6, dimRedMethod='Autoencoder', outputDim=60,
                             trainStart="25/05/2021", evalStart="01/06/2021")

    ARRNN_mod.setARRNN_model(method="Config", modelType="LSTM")

    Y_train_hat, Y_test_hat = ARRNN_mod.getFittedTrainData(scaled=True), ARRNN_mod.getFittedTestData(scaled=True)
    Y_train, Y_test = ARRNN_mod.trainDF['Target'], ARRNN_mod.testDF['Target']

    result_array_TrainCorr.append(corr(Y_train, Y_train_hat))
    result_array_TrainBias.append(Bias(Y_train, Y_train_hat))
    result_array_TrainVar.append(Var(Y_train, Y_train_hat))
    result_array_TrainMSE.append(MSE(Y_train, Y_train_hat))

    result_array_TestCorr.append(corr(Y_test, Y_test_hat))
    result_array_TestBias.append(Bias(Y_test, Y_test_hat))
    result_array_TestVar.append(Var(Y_test, Y_test_hat))
    result_array_TestMSE.append(MSE(Y_test, Y_test_hat))

result_dic_autoencoderEther["LSTMTrainCorr"] = np.mean(result_array_TrainCorr)
result_dic_autoencoderEther["LSTMTrainBias"] = np.mean(result_array_TrainBias)
result_dic_autoencoderEther["LSTMTrainVar"] = np.mean(result_array_TrainVar)
result_dic_autoencoderEther["LSTMTrainMSE"] = np.mean(result_array_TrainMSE)

result_dic_autoencoderEther["LSTMTestCorr"] = np.mean(result_array_TestCorr)
result_dic_autoencoderEther["LSTMTestBias"] = np.mean(result_array_TestBias)
result_dic_autoencoderEther["LSTMTestVar"] = np.mean(result_array_TestVar)
result_dic_autoencoderEther["LSTMTestMSE"] = np.mean(result_array_TestMSE)


### Autoencoder GRU
result_array_TrainCorr = []
result_array_TrainBias = []
result_array_TrainVar = []
result_array_TrainMSE = []

result_array_TestCorr = []
result_array_TestBias = []
result_array_TestVar = []
result_array_TestMSE = []

for i in range(rep_times):
    print(f"---- Starting rep {i} ----")
    ARRNN_mod = AR_RNN_model(data, arOrder=60 * 1, forecastSteps=15, coinID=6, dimRedMethod='Autoencoder', outputDim=60,
                             trainStart="25/05/2021", evalStart="01/06/2021")

    ARRNN_mod.setARRNN_model(method="Config", modelType="GRU")

    Y_train_hat, Y_test_hat = ARRNN_mod.getFittedTrainData(scaled=True), ARRNN_mod.getFittedTestData(scaled=True)
    Y_train, Y_test = ARRNN_mod.trainDF['Target'], ARRNN_mod.testDF['Target']

    result_array_TrainCorr.append(corr(Y_train, Y_train_hat))
    result_array_TrainBias.append(Bias(Y_train, Y_train_hat))
    result_array_TrainVar.append(Var(Y_train, Y_train_hat))
    result_array_TrainMSE.append(MSE(Y_train, Y_train_hat))

    result_array_TestCorr.append(corr(Y_test, Y_test_hat))
    result_array_TestBias.append(Bias(Y_test, Y_test_hat))
    result_array_TestVar.append(Var(Y_test, Y_test_hat))
    result_array_TestMSE.append(MSE(Y_test, Y_test_hat))

result_dic_autoencoderEther["GRUTrainCorr"] = np.mean(result_array_TrainCorr)
result_dic_autoencoderEther["GRUTrainBias"] = np.mean(result_array_TrainBias)
result_dic_autoencoderEther["GRUTrainVar"] = np.mean(result_array_TrainVar)
result_dic_autoencoderEther["GRUTrainMSE"] = np.mean(result_array_TrainMSE)

result_dic_autoencoderEther["GRUTestCorr"] = np.mean(result_array_TestCorr)
result_dic_autoencoderEther["GRUTestBias"] = np.mean(result_array_TestBias)
result_dic_autoencoderEther["GRUTestVar"] = np.mean(result_array_TestVar)
result_dic_autoencoderEther["GRUTestMSE"] = np.mean(result_array_TestMSE)

# %%
### Average LSTM
result_dic_averageEther = {}

result_array_TrainCorr = []
result_array_TrainBias = []
result_array_TrainVar = []
result_array_TrainMSE = []

result_array_TestCorr = []
result_array_TestBias = []
result_array_TestVar = []
result_array_TestMSE = []

for i in range(rep_times):
    print(f"---- Starting rep {i} ----")
    ARRNN_mod = AR_RNN_model(data, arOrder=60 * 1, forecastSteps=15, coinID=6, dimRedMethod='Average', outputDim=60,
                             trainStart="25/05/2021", evalStart="01/06/2021")

    ARRNN_mod.setARRNN_model(method="Config", modelType="LSTM")

    Y_train_hat, Y_test_hat = ARRNN_mod.getFittedTrainData(scaled=True), ARRNN_mod.getFittedTestData(scaled=True)
    Y_train, Y_test = ARRNN_mod.trainDF['Target'], ARRNN_mod.testDF['Target']

    result_array_TrainCorr.append(corr(Y_train, Y_train_hat))
    result_array_TrainBias.append(Bias(Y_train, Y_train_hat))
    result_array_TrainVar.append(Var(Y_train, Y_train_hat))
    result_array_TrainMSE.append(MSE(Y_train, Y_train_hat))

    result_array_TestCorr.append(corr(Y_test, Y_test_hat))
    result_array_TestBias.append(Bias(Y_test, Y_test_hat))
    result_array_TestVar.append(Var(Y_test, Y_test_hat))
    result_array_TestMSE.append(MSE(Y_test, Y_test_hat))

result_dic_averageEther["LSTMTrainCorr"] = np.mean(result_array_TrainCorr)
result_dic_averageEther["LSTMTrainBias"] = np.mean(result_array_TrainBias)
result_dic_averageEther["LSTMTrainVar"] = np.mean(result_array_TrainVar)
result_dic_averageEther["LSTMTrainMSE"] = np.mean(result_array_TrainMSE)

result_dic_averageEther["LSTMTestCorr"] = np.mean(result_array_TestCorr)
result_dic_averageEther["LSTMTestBias"] = np.mean(result_array_TestBias)
result_dic_averageEther["LSTMTestVar"] = np.mean(result_array_TestVar)
result_dic_averageEther["LSTMTestMSE"] = np.mean(result_array_TestMSE)


### Average GRU
result_array_TrainCorr = []
result_array_TrainBias = []
result_array_TrainVar = []
result_array_TrainMSE = []

result_array_TestCorr = []
result_array_TestBias = []
result_array_TestVar = []
result_array_TestMSE = []

for i in range(rep_times):
    print(f"---- Starting rep {i} ----")
    ARRNN_mod = AR_RNN_model(data, arOrder=60 * 1, forecastSteps=15, coinID=6, dimRedMethod='Average', outputDim=60,
                             trainStart="25/05/2021", evalStart="01/06/2021")

    ARRNN_mod.setARRNN_model(method="Config", modelType="GRU")

    Y_train_hat, Y_test_hat = ARRNN_mod.getFittedTrainData(scaled=True), ARRNN_mod.getFittedTestData(scaled=True)
    Y_train, Y_test = ARRNN_mod.trainDF['Target'], ARRNN_mod.testDF['Target']

    result_array_TrainCorr.append(corr(Y_train, Y_train_hat))
    result_array_TrainBias.append(Bias(Y_train, Y_train_hat))
    result_array_TrainVar.append(Var(Y_train, Y_train_hat))
    result_array_TrainMSE.append(MSE(Y_train, Y_train_hat))

    result_array_TestCorr.append(corr(Y_test, Y_test_hat))
    result_array_TestBias.append(Bias(Y_test, Y_test_hat))
    result_array_TestVar.append(Var(Y_test, Y_test_hat))
    result_array_TestMSE.append(MSE(Y_test, Y_test_hat))

result_dic_averageEther["GRUTrainCorr"] = np.mean(result_array_TrainCorr)
result_dic_averageEther["GRUTrainBias"] = np.mean(result_array_TrainBias)
result_dic_averageEther["GRUTrainVar"] = np.mean(result_array_TrainVar)
result_dic_averageEther["GRUTrainMSE"] = np.mean(result_array_TrainMSE)

result_dic_averageEther["GRUTestCorr"] = np.mean(result_array_TestCorr)
result_dic_averageEther["GRUTestBias"] = np.mean(result_array_TestBias)
result_dic_averageEther["GRUTestVar"] = np.mean(result_array_TestVar)
result_dic_averageEther["GRUTestMSE"] = np.mean(result_array_TestMSE)

# %%
### None LSTM
result_dic_noneEther = {}

result_array_TrainCorr = []
result_array_TrainBias = []
result_array_TrainVar = []
result_array_TrainMSE = []

result_array_TestCorr = []
result_array_TestBias = []
result_array_TestVar = []
result_array_TestMSE = []

for i in range(rep_times):
    print(f"---- Starting rep {i} ----")
    ARRNN_mod = AR_RNN_model(data, arOrder=60 * 1, forecastSteps=15, coinID=6, dimRedMethod='None', outputDim=60,
                             trainStart="25/05/2021", evalStart="01/06/2021")

    ARRNN_mod.setARRNN_model(method="Config", modelType="LSTM")

    Y_train_hat, Y_test_hat = ARRNN_mod.getFittedTrainData(scaled=True), ARRNN_mod.getFittedTestData(scaled=True)
    Y_train, Y_test = ARRNN_mod.trainDF['Target'], ARRNN_mod.testDF['Target']

    result_array_TrainCorr.append(corr(Y_train, Y_train_hat))
    result_array_TrainBias.append(Bias(Y_train, Y_train_hat))
    result_array_TrainVar.append(Var(Y_train, Y_train_hat))
    result_array_TrainMSE.append(MSE(Y_train, Y_train_hat))

    result_array_TestCorr.append(corr(Y_test, Y_test_hat))
    result_array_TestBias.append(Bias(Y_test, Y_test_hat))
    result_array_TestVar.append(Var(Y_test, Y_test_hat))
    result_array_TestMSE.append(MSE(Y_test, Y_test_hat))

result_dic_noneEther["LSTMTrainCorr"] = np.mean(result_array_TrainCorr)
result_dic_noneEther["LSTMTrainBias"] = np.mean(result_array_TrainBias)
result_dic_noneEther["LSTMTrainVar"] = np.mean(result_array_TrainVar)
result_dic_noneEther["LSTMTrainMSE"] = np.mean(result_array_TrainMSE)

result_dic_noneEther["LSTMTestCorr"] = np.mean(result_array_TestCorr)
result_dic_noneEther["LSTMTestBias"] = np.mean(result_array_TestBias)
result_dic_noneEther["LSTMTestVar"] = np.mean(result_array_TestVar)
result_dic_noneEther["LSTMTestMSE"] = np.mean(result_array_TestMSE)


### None GRU
result_array_TrainCorr = []
result_array_TrainBias = []
result_array_TrainVar = []
result_array_TrainMSE = []

result_array_TestCorr = []
result_array_TestBias = []
result_array_TestVar = []
result_array_TestMSE = []

for i in range(rep_times):
    print(f"---- Starting rep {i} ----")
    ARRNN_mod = AR_RNN_model(data, arOrder=60 * 1, forecastSteps=15, coinID=6, dimRedMethod='None', outputDim=60,
                             trainStart="25/05/2021", evalStart="01/06/2021")

    ARRNN_mod.setARRNN_model(method="Config", modelType="GRU")

    Y_train_hat, Y_test_hat = ARRNN_mod.getFittedTrainData(scaled=True), ARRNN_mod.getFittedTestData(scaled=True)
    Y_train, Y_test = ARRNN_mod.trainDF['Target'], ARRNN_mod.testDF['Target']

    result_array_TrainCorr.append(corr(Y_train, Y_train_hat))
    result_array_TrainBias.append(Bias(Y_train, Y_train_hat))
    result_array_TrainVar.append(Var(Y_train, Y_train_hat))
    result_array_TrainMSE.append(MSE(Y_train, Y_train_hat))

    result_array_TestCorr.append(corr(Y_test, Y_test_hat))
    result_array_TestBias.append(Bias(Y_test, Y_test_hat))
    result_array_TestVar.append(Var(Y_test, Y_test_hat))
    result_array_TestMSE.append(MSE(Y_test, Y_test_hat))

result_dic_noneEther["GRUTrainCorr"] = np.mean(result_array_TrainCorr)
result_dic_noneEther["GRUTrainBias"] = np.mean(result_array_TrainBias)
result_dic_noneEther["GRUTrainVar"] = np.mean(result_array_TrainVar)
result_dic_noneEther["GRUTrainMSE"] = np.mean(result_array_TrainMSE)

result_dic_noneEther["GRUTestCorr"] = np.mean(result_array_TestCorr)
result_dic_noneEther["GRUTestBias"] = np.mean(result_array_TestBias)
result_dic_noneEther["GRUTestVar"] = np.mean(result_array_TestVar)
result_dic_noneEther["GRUTestMSE"] = np.mean(result_array_TestMSE)






