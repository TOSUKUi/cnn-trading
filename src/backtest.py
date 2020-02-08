from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from datetime import datetime
import backtrader as bt
import pandas as pd
import torch
import os.path
import sys


class TestStrategy(bt.Strategy):
    def log(self, txt, dt=None):
        ''' Logging function for this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close

    def next(self):
        # Simply log the closing price of the series from the reference
        self.log('Close, %.2f' % self.dataclose[0])


if __name__ == "__main__":
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(100000.0)

    cerebro.addstrategy(TestStrategy)
    how = {
        "open": 'first',
        "high": 'max',
        "low": "min",
        "close": "last",
        "volume": 'sum'
    }
    df = pd.read_csv(
        "jupyter/bitstampUSD_1-min_data_2012-01-01_to_2019-08-12.csv")
    df.loc[:, "datetime"] = pd.to_datetime(df["Timestamp"], unit='s')
    df = df[["datetime", "Open", "High", "Low", "Close", "Volume_(Currency)"]]
    print(df)

    df.columns = ["datetime", "open", "high", "low", "close", "volume"]
    df = df.set_index("datetime")
    df = df.resample("60min", how=how)
    df = df.bfill()
    print(df)

    data = bt.feeds.pandafeed.PandasData(dataname=df, fromdate=datetime(
        2018, 1, 1), todate=datetime(2019, 12, 1), )

#     data = bt.feeds.YahooFinanceCSVData(
#         dataname=datapath,
#         fromdate=datetime(2000, 1, 1),
#         todate=datetime(2000, 12, 31),
#         reverse=False
#     )

    cerebro.adddata(data)
    cerebro.broker.setcash(100000.0)
    print("starting portfolio value : %.2f" % cerebro.broker.getvalue())
    cerebro.run()
    print("Final Portfolio value : %.2f" % cerebro.broker.getvalue())
