from Scr.Descriptive_Analysis import Descriptive_Analysis
from Scr.functions import import_data

#Import the whole dataset
all_data, all_data_details = import_data("C:/Users/Albert Nietz/PyCharm_Projects/DLTA, First Project Assignment/DLTA_ProjectAssignment_I/Data")
#Define coins that should be analysed
coin_ids = [4, 6]

#Create an instance of the descriptive analysis class
da = Descriptive_Analysis(all_data, all_data_details, coin_ids)

#Create subplots: target variable (proxy for return) over time
da.createSubplotsReturnOverTime()
#Create one plot: target variable (proxy for return) over time of all analysed coins
da.createOnePlotReturnOverTime()
#Create subplots: return distribution / histogram
da.createSubplotsReturnDistribution()

#Create dataframe for the following correlation plots
dataframe_corr_plots = da.prepareDataFrameforCorrelationPlots()
#Create subplots: seven day correlation
da.createSubplotsSevenDayCorrelation(dataframe_corr_plots)
#Create heat map
da.createHeatMap(dataframe_corr_plots)




