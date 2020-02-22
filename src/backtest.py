from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import torch.nn.functional as F
from torch import optim
import torch
from torch import zeros
import torch.nn as nn
import numpy as np
import pandas as pd
from datetime import datetime
import backtrader as bt
import matplotlib
matplotlib.use('Agg')


PATH = "models/Model_16LSTMwindow.pkl"


class MinMaxNormalizer():

    def min_max(self, x, axis=None):
        self.min = x.min(axis=axis, keepdims=True)
        self.max = x.max(axis=axis, keepdims=True)
        result = (x-self.min)/(self.max-self.min)
        return result

    def inverse_min_max(self, x, axis=None, index=None):
        return x*(self.max - self.min) + self.min


class GRUNet(nn.Module):
    a = 1

    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.5):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.gru = nn.LSTM(input_dim, hidden_dim, n_layers,
                           batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x, h=None):
        out, h = self.gru(x, h)
        out = self.sigmoid(self.fc(self.relu(out[:, -1])))
        return out


class TestStrategy(bt.Strategy):

    def log(self, txt, dt=None):
        ''' Logging function for this strategy'''
        dt = dt or self.datas[0].datetime.datetime(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close
        self.dataopen = self.datas[0].open
        self.datahigh = self.datas[0].high
        self.datalow = self.datas[0].low
        self.datavolume = self.datas[0].volume
        self.order = None
        self.window = 16
        self.count = 0
        self.cuda = True if torch.cuda.is_available() else False

        self.model = torch.load(PATH)

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log('BUY EXECUTED, %.2f' % order.executed.price)
            elif order.issell():
                self.log('SELL EXECUTED, %.2f' % order.executed.price)
            elif order.isclose():
                self.log('order closed, %.2f' % order.executed.price)

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        self.order = None

    def next(self):
        self.count += 1
        if self.order:
            return
        if self.position.size > 0:
            self.close()
        if self.count > 16:
            arr = np.zeros((1, 16, 5), dtype=np.float32)

            input_seq = np.diff(np.array([self.dataopen.get(size=self.window), self.datahigh.get(size=self.window), self.datalow.get(
                size=self.window), self.dataclose.get(size=self.window), self.datavolume.get(size=self.window)]).transpose(1, 0).astype(np.float32), axis=0)
            # self.log(arr[0, :-1].shape)
            arr[0, :-1] = input_seq
            arr[0, -1] = input_seq.mean(axis=0)
            sc = MinMaxNormalizer()
            arr_min_max = sc.min_max(arr, axis=1)
            # self.log(arr_min_max)
            y = arr_min_max[0, -1, 3]
            x = arr_min_max[:, :-1]
            if self.cuda:
                x, y = torch.tensor(x, dtype=torch.float32, device='cuda'), torch.tensor(
                    y, dtype=torch.float32, device='cuda')
            else:
                x, y = torch.tensor(x, dtype=torch.float32), torch.tensor(
                    y, dtype=torch.float32)
            # self.log(x)
            pred = self.model(x)
            predcpu = pred.detach().cpu().numpy()
            cp_arr_min_max = arr_min_max.copy()
            cp_arr_min_max[0, -1, 3] = predcpu
            cp_arr = sc.inverse_min_max(cp_arr_min_max)
            out = cp_arr[0, -1, 3]
            self.log(self.dataclose[0])
            self.log(out)
            if abs(out) > 30:
                if out < 0:
                    self.order = self.sell()
                elif out > 0:
                    self.order = self.buy()


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
        "jupyter/coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv")
    # df = pd.read_csv(
    #     "jupyter/bitstampUSD_1-min_data_2012-01-01_to_2019-08-12.csv")
    df.loc[:, "datetime"] = pd.to_datetime(df["Timestamp"], unit='s')
    df = df[["datetime", "Open", "High", "Low", "Close", "Volume_(Currency)"]]

    df.columns = ["datetime", "open", "high", "low", "close", "volume"]
    df = df.set_index("datetime")
    df = df.resample("60min").agg(how)
    df = df.bfill()

    data = bt.feeds.pandafeed.PandasData(dataname=df, fromdate=datetime(
        2018, 8, 1), todate=datetime(2019, 1, 9), )

    cerebro.adddata(data)
    cerebro.broker.setcash(100000.0)
    cerebro.broker.setcommission(commission=0.0075)
    print("starting portfolio value : %.2f" % cerebro.broker.getvalue())
    cerebro.run(maxcpu=16)
    print("Final Portfolio value : %.2f" % cerebro.broker.getvalue())
    cerebro.plot(style='candle', subtxtsize=7, dpi=500)

    matplotlib.pyplot.savefig("unko.png")
