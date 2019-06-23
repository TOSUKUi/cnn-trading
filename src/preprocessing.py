import pandas as pd
from pipeline import Procedure
from datetime import datetime
import numpy as np
from tqdm import tqdm
from numba import njit, jit
as_strided = np.lib.stride_tricks.as_strided  



class PreprocessingProcedure1D(Procedure):
    
    def run(self, x):
        ohlc_dict = {
            'Open':'first',
            'High':'max',
            'Low':'min',
            'Close':'last',
            'Volume_(Currency)':'sum'
        }
        return self.preprocessing(x, ohlc_dict)

    @jit
    def preprocessing(self, df, how):
        dataset = []
        df.loc[:, "datetime"] = pd.to_datetime(df['Timestamp'], unit='s')
        df_d = df.set_index("datetime")

        df_accept = df_d.loc[datetime(2013, 8, 1):]
        df_b_a = df_accept.bfill()
        df_resampled_1min = df_b_a[["Open", "High", "Low", "Close", "Volume_(Currency)"]]
        X = []
        y = []
        for n in tqdm(range(1800, len(df_resampled_1min) - 60, 60)):
            x_base = df_resampled_1min.iloc[n-1800:n, :]
            x_base_normalize = normalize(x_base)
            y.append((df_resampled_1min["Close"].iloc[n+60] - df_resampled_1min["Close"].iloc[n]) / df_resampled_1min["Close"].iloc[n])
            X.append(x_base_normalize.T.values)
        return np.array(X), np.array(y)


def normalize(df):
    return ( df - df.mean() ) / df.std()


class TrainingPreprocessingProcedure1D(Procedure):
    
    def run(self, x):
        ohlc_dict = {
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume_(Currency)': 'sum'
        }
        return self.preprocessing(x, ohlc_dict)

    @jit
    def preprocessing(self, df, how):
        dataset = []
        df.loc[:, "datetime"] = pd.to_datetime(df['Timestamp'], unit='s')
        df_d = df.set_index("datetime")

        df_accept = df_d.loc[datetime(2013, 8, 1):]
        
        df_b_a = df_accept.bfill()
        df_resampled_1min = df_b_a[["Open", "High", "Low", "Close", "Volume_(Currency)"]]
        X = []
        y = []
        for n in tqdm(range(60, len(df_resampled_1min) - 60, 60)):
            x_base = df_resampled_1min.iloc[n-60:n, :]
            x_base_normalize = normalize(x_base)
            y.append((df_resampled_1min["Close"].iloc[n+60] - df_resampled_1min["Close"].iloc[n]) / df_resampled_1min["Close"].iloc[n])
            X.append(x_base_normalize.values)
        return np.array(X, dtype=np.float16), np.array(y, dtype=np.float16)


class TrainingPreprocessingProcedure1DBinary(Procedure):
    
    def run(self, x):
        ohlc_dict = {
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume_(Currency)': 'sum'
        }
        return self.preprocessing(x, ohlc_dict)

    def preprocessing(self, df, how):

        df.loc[:, "datetime"] = pd.to_datetime(df['Timestamp'], unit='s')
        df_d = df.set_index("datetime")

        df_accept = df_d.loc[datetime(2013, 8, 1):]
        
        df_b_a = df_accept.bfill()
        df_resampled_1min = df_b_a[["Open", "High", "Low", "Close", "Volume_(Currency)"]]
        array = df_resampled_15min.values 
        X, y = dataset(array)
        return np.array(X), np.array(y) 
    
@njit
def dataset(array):
    X = []
    y = []
    count = 0
    for n in range(60, len(array) - 60, 60):
        count += 1
        x_base = array[n-60:n, :]
        x_base_normalize = (x_base - x_base.mean()) / x_base.std()
        y.append(1 if array[n+60, 3] - array[n, 3] > 0 else 0)
        X.append(x_base_normalize)
    return X, y


class GramMatrixPreprocessing(Procedure):

    def run(self, x):
        return self.preprocessing(x)
    
    def preprocessing(self, df):
        df.loc[:, "datetime"] = pd.to_datetime(df['Timestamp'], unit='s')
        df_d = df.set_index("datetime")
        df_accept = df_d.loc[datetime(2013, 8, 1):]
        df_b_a = df_accept.bfill()
        df_b_a_ocv = df_b_a[["Open", "Close", "Volume_(Currency)"]]
        df_resampled_15min = df_b_a_ocv
        array = df_resampled_15min.values 
        X, y = dataset_gram_matrix(array)
        X_reshape = np.reshape(X, (X.shape[0], X.shape[2], X.shape[3], X.shape[1]))
        return X_reshape, y


@jit
def dataset_gram_matrix(array):
    X = []
    y = []
    for n in range(250, len(array)-1000000, 250):
        matrix_list = []
        base = array[n-250:n, :]
        base_normalize = ((base - base.max()) - (base - base.min())) / (base.max() - base.min()) 
        for i in range(3):
            matrix_list.append(np.multiply(base_normalize[:, [i]], base_normalize[:, [i]].T))
        matrixes = np.array(matrix_list)
        X.append(matrixes.astype(np.float16))
        y.append(1 if array[n+1, 1] - array[n, 1] > 0 else 0)
    return np.array(X), np.array(y)
