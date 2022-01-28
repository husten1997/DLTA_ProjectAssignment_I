#%%
from Scr.Advanced_Model import Advanced_Model
from Scr.functions import import_data
from Scr.functions import createHeatMapTopFeatureVariables
from Scr.functions import performanceEval

#Import whole dataset
all_data, all_data_details = import_data("Data")
#Define one coin that should be analysed / predicted with his ID => Currently: Ethereum
coin_id = 6

#Create an instance of advanced model
advanced_model = Advanced_Model(coin_id, all_data, all_data_details, trainStart = "25/05/2021", evalStart = "01/06/2021")

#%%
#Create heat map of the top feature variables
coin_name = advanced_model.getCoinName()
finalDataframe_training = advanced_model.getFinalDataframeTraining()
finalDataFrame_test = advanced_model.getFinalDataframeTest()
finaleDataFrame_eval = advanced_model.getFinalDataFrameEval()
top_features = advanced_model.getTopFeatureVariables()

createHeatMapTopFeatureVariables(coin_name, finalDataframe_training, finalDataFrame_test, top_features)

#%%
#Apply / set advanced model
advanced_model.applyModel(20)

#Conduct performance evaluation
y_train, y_test, y_eval = advanced_model.getRealData()
y_train_hat, y_test_hat, y_eval_hat = advanced_model.getFittedData()

performanceEval(y_train, y_train_hat, "In-Sample")
performanceEval(y_test, y_test_hat, "Out-of-Sample")
performanceEval(y_eval, y_eval_hat, "True Out-of-Sample")











