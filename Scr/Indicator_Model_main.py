#%%
from Scr.Indicator_Model import Indicator_Model
from Scr.functions import import_data
from Scr.functions import createHeatMapTopFeatureVariables
from Scr.functions import performanceEval
import numpy as np
import pandas as pd

#Import whole dataset
all_data, all_data_details = import_data("C:/Users/Fabia/pythoninput/DLTA/Data")
#Define one coin that should be analysed / predicted with his ID => Currently: Ethereum
coin_id = 6

#Create an instance of advanced model
indicator_model = Indicator_Model(coin_id, all_data, all_data_details, trainStart="25/05/2021", evalStart="01/06/2021")

#%%
#Create heat map of the top feature variables
coin_name = indicator_model.getCoinName()
finalDataframe_training = indicator_model.getFinalDataframeTraining()
finalDataFrame_test = indicator_model.getFinalDataframeTest()
finaleDataFrame_eval = indicator_model.getFinalDataFrameEval()
top_features = indicator_model.getTopFeatureVariables()

createHeatMapTopFeatureVariables(coin_name, finalDataframe_training, finalDataFrame_test, top_features)

#%%
#Apply / set advanced model
indicator_model.applyModel(20)

#Conduct performance evaluation
y_train, y_test, y_eval = indicator_model.getRealData()
y_train_hat, y_test_hat, y_eval_hat = indicator_model.getFittedData()

performanceEval(y_train, y_train_hat, "In-Sample")
performanceEval(y_test, y_test_hat, "Out-of-Sample")
performanceEval(y_eval, y_eval_hat, "True Out-of-Sample")











