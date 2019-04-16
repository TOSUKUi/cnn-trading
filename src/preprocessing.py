import pandas as pd
from pipeline import Procedure



class PreprocessingProcedure(Procedure):
    
    def run(self, x):
        ohlc_dict = {
            'Open':'first',
            'High':'max',
            'Low':'min',
            'Close':'last',
            'Volume_(Currency)':'sum'
        }
        return self.preprocessing(x, ohlc_dict)

    def preprocessing(self, df, how):
        df_resampled = df.resample('60min', how=how)
        for n in range(500):
            x_base = df_resampled.iloc[n:n+60, :]
            x_base = x_base - x_base.head(1).iloc[0]
            min_date = x_base.index[0]
            max_date = x_base.index[-1]
            unit = df.loc[min_date:max_date]
            unit = unit - unit.head(1).iloc[0]
            x = []
            x.append(x_base)
            x.append(unit.resample('1min', how=how))
            x.append(unit.resample('5min', how=how))
            x.append(unit.resample('15min', how=how))
            x.append(unit.resample('30min', how=how))
            y = df_resampled.iloc[n+61] - unit.iloc[0]
            X = pd.concat(x)
            yield X, y
