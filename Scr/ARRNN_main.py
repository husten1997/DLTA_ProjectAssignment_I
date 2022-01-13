#%%
from Scr.AR_RNN_model import AR_RNN_model
from Scr.functions import import_data
from Scr.functions import performanceEval

data = import_data("Data/")

#%%
ARRNN_mod = AR_RNN_model(data, arOrder = 15, forecastSteps = 15, coinID = 4, dimRedMethod = 'None')

#%%
ARRNN_mod.setARRNN_model(method = "Config")

#%%
Y_train_hat, Y_test_hat = ARRNN_mod.getFittedData()
Y_train, Y_test = ARRNN_mod.generateFitFeaturesSet("Average")[2:]

print("In-sample corr: " + performanceEval(Y_train, Y_train_hat, "In-Sample "))
print("Out-of-sample corr: " + performanceEval(Y_test, Y_test_hat, "Out-of-Sample "))
