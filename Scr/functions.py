import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

def import_data(dir):

    dtypes = {
        'timestamp': np.int64,
        'Asset_ID': np.int8,
        'Count': np.int32,
        'Open': np.float64,
        'High': np.float64,
        'Low': np.float64,
        'Close': np.float64,
        'Volume': np.float64,
        'VWAP': np.float64,
        'Target': np.float64,
    }

    file_path = os.path.join(dir, 'train.csv')
    all_data = pd.read_csv(file_path, dtype=dtypes, usecols=list(dtypes.keys()))
    all_data['Time'] = pd.to_datetime(all_data['timestamp'], unit='s')

    file_path = os.path.join(dir, 'asset_details.csv')
    all_data_details = pd.read_csv(file_path)

    all_data = pd.merge(all_data,
                         all_data_details,
                         on="Asset_ID",
                         how='left')

    return all_data, all_data_details

def createHeatMapTopFeatureVariables(coin_name, df_training, df_test, top_features):

    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    fig.suptitle(coin_name, fontsize=20)
    axs[0].set_title('Training Data')
    axs[1].set_title('Test Data')

    sb.heatmap(df_training[top_features].corr(method='spearman').abs(), ax=axs[0])
    sb.heatmap(df_test[top_features].corr(method='spearman').abs(), ax=axs[1])
    plt.show()

def performanceEval(Y, Y_hat):

    var = lambda x: (1 / (len(x) - 1)) * (np.sum(x * x) - (1 / len(x)) * (np.sum(x) ** 2))
    cov = lambda x, y: (1 / (len(x) - 1)) * (np.sum(x * y) - (1 / len(x)) * np.sum(x) * np.sum(y))
    corr = lambda x, y: (cov(x, y)) / np.sqrt(var(x) * var(y))

    u = Y.reshape((-1)) - Y_hat.reshape((-1))

    plt.plot(u, label='Residuals')
    plt.legend()
    plt.show()

    return str(corr(Y.reshape((-1)), Y_hat.reshape((-1))))


