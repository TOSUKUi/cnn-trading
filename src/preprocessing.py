import pandas as pd
from core.pipeline import Procedure
from datetime import datetime
import numpy as np
from chainer.datasets import TupleDataset
from tqdm import tqdm
from numba import jit


class PreprocessingProcedure1D(Procedure):
    """
    Explain as Procedures.
    
    Args:
        df (pd.DataFrame): dataframe which has at least 1800 length and contain ohlcv informations 

    Returns:
        (np.ndarray shape (1, 5, 1800)): Normalizationed dataframe which length is just a 1800.
    """

    def __init__(self, ohlcv_columns):
        """this is preprocessing for predicate next legs
        
        Args:
            ohlcv_columns (List[str]): List of columns which specify the each of ohlcv column.
        """
        self.columns = ohlcv_columns
    
    def run(self, x):
        return self.preprocessing(x)

    @jit
    def preprocessing(self, df):
        # how = {
        #     'Open':'first',
        #     'High':'max',
        #     'Low':'min',
        #     'Close':'last',
        #     'Volume_(Currency)':'sum'
        # }
        df["datetime"] = pd.to_datetime(df['Timestamp'], unit='s')
        df_d = df.set_index("datetime")
        df_d_a = df_d.tail(1800)
        df_d_a_b = df_d_a.bfill()
        df_d_a_b_ohlcv = df_d_a_b[self.ohlcv_columns]
        return np.array([normalize(df_d_a_b_ohlcv).values.T])


def normalize(df):
    return (df - df.mean()) / df.std()


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
        for n in tqdm(range(1800, len(df_resampled_1min) - 60, 60)):
            x_base = df_resampled_1min.iloc[n-1800:n, :]
            x_base_normalize = normalize(x_base)
            y.append((df_resampled_1min["Close"].iloc[n+60] - df_resampled_1min["Close"].iloc[n]) / df_resampled_1min["Close"].iloc[n])
            X.append(x_base_normalize.T.values)
        return np.array(X), np.array(y)


def normalize(df):
    return ( df - df.mean() ) / df.std()