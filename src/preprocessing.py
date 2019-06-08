import pandas as pd
from pipeline import Procedure
from datetime import datetime
import numpy as np
from chainer.datasets import TupleDataset
from tqdm import tqdm
from numba import jit
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

    @jit
    def preprocessing(self, df, how):
        win = 60

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
            y.append(1 if df_resampled_1min["Close"].iloc[n+60] - df_resampled_1min["Close"].iloc[n] > 0 else 0)
            X.append(x_base_normalize.values)
        return np.array(X, dtype=np.float16), np.array(y, dtype=np.float16) 


def normalize(df):
    return ( df - df.mean() ) / df.std()