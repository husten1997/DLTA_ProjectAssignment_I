#%%
from Scr.AR_RNN_model_2 import AR_RNN_model
from Scr.functions import import_data
from Scr.functions import performanceEval

data = import_data("Data/")[0]

result_dic = {}

#%%
ARRNN_mod = AR_RNN_model(data, arOrder = 60 * 1, forecastSteps = 15, coinID = 4, dimRedMethod = 'Autoencoder', outputDim=15, trainStart = "25/05/2021", evalStart = "01/06/2021")

#%%
ARRNN_mod.setARRNN_model(method = "Config", modelType = "GRU")
ARRNN_mod.tuner.get_best_hyperparameters()[0].values

#%%
Y_train_hat, Y_test_hat, Y_eval_hat = ARRNN_mod.getFittedData(scaled = True)
Y_train, Y_test, Y_eval = ARRNN_mod.trainDF['Target'], ARRNN_mod.testDF['Target'], ARRNN_mod.evalDF['Target']

performanceEval(Y_train, Y_train_hat, "In-Sample ")
performanceEval(Y_test, Y_test_hat, "Out-of-Sample ")
performanceEval(Y_eval, Y_eval_hat, "True Out-of-Sample ")


