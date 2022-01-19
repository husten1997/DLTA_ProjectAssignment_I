#%%
from Scr.AR_RNN_model_2 import AR_RNN_model
from Scr.functions import import_data
from Scr.functions import performanceEval

data = import_data("Data/")

#%%
ARRNN_mod = AR_RNN_model(data, arOrder = 60 * 4, forecastSteps = 15, coinID = 4, dimRedMethod = 'Autoencoder', outputDim=15)

#%%
ARRNN_mod.setARRNN_model(method = "Tuner")

#%%
Y_train_hat, Y_test_hat = ARRNN_mod.getFittedData()
Y_train, Y_test = ARRNN_mod.trainDF['Target'], ARRNN_mod.testDF['Target']

performanceEval(Y_train, Y_train_hat, "In-Sample ")
performanceEval(Y_test, Y_test_hat, "Out-of-Sample ")


