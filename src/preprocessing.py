import pandas as pd
from pipeline import Procedure
from datetime import datetime
import numpy as np
from tqdm import tqdm
from numba import njit, jit
from tensorflow.python.keras.utils import to_categorical
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
        how = {
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume_(Currency)': 'sum'
        }
        df.loc[:, "datetime"] = pd.to_datetime(df['Timestamp'], unit='s')
        df_d = df.set_index("datetime")
        df_accept = df_d.loc[datetime(2014, 8, 1):]
        df_b_a = df_accept.bfill()
        df_b_a_ocv = df_b_a[["Open", "Close", "Volume_(Currency)"]]
        df_resampled_15min = df_b_a_ocv.resample('15min', how=how)
        array = df_resampled_15min.values.astype(np.float32) 
        X, y = dataset_gram_matrix(array)
        X_reshape = np.reshape(X, (X.shape[0], X.shape[2], X.shape[3], X.shape[1]))
        y_categorical = to_categorical(y)
        print(y_categorical)
        return X_reshape, y_categorical.astype(np.uint8)


@jit
def dataset_gram_matrix(array, binary=True):
    X = []
    y = []
    for n in range(250, len(array)-1, 25):
        matrix_list = []
        base = array[n - 250:n, :]
        for i in range(3):
            base_col = base[:, [i]]
            base_normalize = ((base_col - base_col.min()) / (base_col.max() - base_col.min())).astype(np.float16)
            matrix_list.append(np.multiply(base_normalize, base_normalize.T))
        matrixes = np.array(matrix_list)
        X.append(matrixes)
        y.append( array[n+1, 1] - array[n, 1] )
    return np.array(X), np.array(y)


class GramMatrixPreprocessingRegression(Procedure):

    def run(self, x):
        return self.preprocessing(x)
    
    def preprocessing(self, df):
        how = {
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume_(Currency)': 'sum'
        }
        df.loc[:, "datetime"] = pd.to_datetime(df['Timestamp'], unit='s')
        df_d = df.set_index("datetime")
        df_accept = df_d.loc[datetime(2014, 8, 1):]
        df_b_a = df_accept.bfill()
        df_b_a_ocv = df_b_a[["Open", "Close", "Volume_(Currency)"]]
        df_resampled_15min = df_b_a_ocv.resample('15min', how=how)
        array = df_resampled_15min.values.astype(np.float32) 
        X, y = dataset_gram_matrix(array)
        X_reshape = np.transpose(X, [0, 2, 3, 1])
        return X_reshape, y
