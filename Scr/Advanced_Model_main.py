from Scr.Advanced_Model import Advanced_Model
from Scr.functions import import_data
from Scr.functions import createHeatMapTopFeatureVariables
from Scr.functions import performanceEval

#Import whole dataset
all_data, all_data_details = import_data("Data")
#Define one coin that should be analysed / predicted with his ID => Currently: Ethereum
coin_id = 6

#Create an instance of advanced model
advanced_model = Advanced_Model(coin_id, all_data, all_data_details)

#Create heat map of the top feature variables
coin_name = advanced_model.getCoinName()
finalDataframe_training = advanced_model.getFinalDataframeTraining()
finalDataFrame_test = advanced_model.getFinalDataframeTest()
top_features = advanced_model.getTopFeatureVariables()

createHeatMapTopFeatureVariables(coin_name, finalDataframe_training, finalDataFrame_test, top_features)

#Apply / set advanced model
advanced_model.applyModel(20)

#Conduct performance evaluation
y_train, y_test = advanced_model.getRealData()
y_train_hat, y_test_hat = advanced_model.getFittedData()

performanceEval(y_train, y_train_hat, "In-Sample ")
performanceEval(y_test, y_test_hat, "Out-of-Sample ")










