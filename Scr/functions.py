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

def performanceEval(Y, Y_hat, preFix = ""):
    import numpy as np
    import matplotlib.pyplot as plt

    Y = np.array(Y)
    Y_hat = np.array(Y_hat)

    u = Y.reshape((-1)) - Y_hat.reshape((-1))

    print(preFix + " corr(Y, Y_hat): " + str(corr(Y, Y_hat)))
    print(preFix + " Bias(u): " + str(Bias(Y, Y_hat)))
    print(preFix + " Var(u): " + str(Var(Y, Y_hat)))
    print(preFix + " MSE(u): " + str(MSE(Y, Y_hat)))

    plt.plot(u, label='Residuals')
    plt.title(preFix + 'Residuals over time (u x t)')
    plt.legend()
    plt.show()

    plt.plot(u, Y_hat, 'bo', label='Residuals')
    plt.title(preFix + 'Residual Plot (u x y.hat)')
    plt.legend()
    plt.show()

    #return str(corr)
    #return str(corr(Y.reshape((-1)), Y_hat.reshape((-1))))

def corr(Y, Y_hat):
    import numpy as np

    Y = np.array(Y).flatten()
    Y_hat = np.array(Y_hat).flatten()

    cov_matrix = np.cov(Y, Y_hat)

    return cov_matrix[0, 1] / np.sqrt(cov_matrix[0, 0] * cov_matrix[1, 1])

def Bias(Y, Y_hat):
    import numpy as np

    Y = np.array(Y).flatten()
    Y_hat = np.array(Y_hat).flatten()

    u = Y - Y_hat

    return np.mean(u)

def Var(Y, Y_hat):
    import numpy as np

    Y = np.array(Y).flatten()
    Y_hat = np.array(Y_hat).flatten()

    u = Y - Y_hat

    return np.var(u)

def MSE(Y, Y_hat):
    return Bias(Y, Y_hat) + Var(Y, Y_hat)
