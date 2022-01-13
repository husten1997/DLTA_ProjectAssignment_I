def import_data(dir):
    import os
    import numpy as np
    import pandas as pd

    file_path = os.path.join(dir, 'train.csv')
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
    data = pd.read_csv(file_path, dtype=dtypes, usecols=list(dtypes.keys()))
    data['Time'] = pd.to_datetime(data['timestamp'], unit='s')

    file_path = os.path.join(dir, 'asset_details.csv')
    details = pd.read_csv(file_path)

    data = pd.merge(data,
                    details,
                    on="Asset_ID",
                    how='left')

    return data

def performanceEval(Y, Y_hat, plotTitleAddition = ""):
    import numpy as np
    import matplotlib.pyplot as plt
    from statistics import mean

    Y = np.array(Y)
    Y_hat = np.array(Y_hat)

    # var = lambda x: (1 / (len(x) - 1)) * (np.sum(x * x) - (1 / len(x)) * (np.sum(x) ** 2))
    # cov = lambda x, y: (1 / (len(x) - 1)) * (np.sum(x * y) - (1 / len(x)) * np.sum(x) * np.sum(y))
    # corr = lambda x, y: (cov(x, y)) / np.sqrt(var(x) * var(y))

    cov_matrix = np.cov(Y.flatten(), Y_hat.flatten())

    corr = cov_matrix[0, 1] / np.sqrt(cov_matrix[0, 0] * cov_matrix[1, 1])

    u = Y.reshape((-1)) - Y_hat.reshape((-1))

    plt.plot(u, label='Residuals')
    plt.title(plotTitleAddition + 'Residuals over time (u x t)')
    plt.legend()
    plt.show()

    plt.plot(u, Y_hat, 'bo', label='Residuals')
    plt.title(plotTitleAddition + 'Residual Plot (u x y.hat)')
    plt.legend()
    plt.show()

    return str(corr)
    #return str(corr(Y.reshape((-1)), Y_hat.reshape((-1))))