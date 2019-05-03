from read_data import ReadData
from core.pipeline import PipeLine 
from preprocessing import TrainingPreprocessingProcedure1D, PreprocessingProcedure1D 
from strategies.training import TrainProcedureChainer, TrainProcedureKeras
from strategies.neural_net import KerasLinear1D
from strategies.trading_strategies import TradingStrategy


import sys
sys.path.append("/Users/TOSUKUi/Documents/workspace/trading-deeplearning")


def train_pipeline():
    saved_model_path = "models/saved_model.h5"
    data_path = 'data/bitstampUSD_1-min_data_2012-01-01_to_2019-03-13.csv',
    PipeLine(
        data_path,
        ReadData(),
        TrainingPreprocessingProcedure1D(),
        TrainProcedureKeras(
            KerasLinear1D(
                saved_model_path=saved_model_path
            )
        )
    ).execute()


def strategy_order_pipeline():
    """
    It is kicked by cron or some scheduler, but you can use it by simple while loop.
    """

    saved_model_path = "models/saved_model.h5"

    # for example

    ohlcv_columns = ["Open", "High", "Low", "Close", "Volume_(Currency)"]
    data_path = 'data/bitstampUSD_1-min_data_2012-01-01_to_2019-03-13.csv',
    p = PipeLine(
        data_path,
        ReadData(),
        PreprocessingProcedure1D(ohlcv_columns),
        KerasLinear1D(
            saved_model_path=saved_model_path,
            model=open(saved_model_path, "rb")
        ),
        TradingStrategy()
    )
    print(p.execute())


def backtest_pipeline():
    data_path = 'data/bitstampUSD_1-min_data_2012-01-01_to_2019-03-13.csv',
    saved_model_path = "models/saved_model.h5"
    PipeLine(
        data_path,
        ReadData,
        PreprocessingProcedure1D(ohlcv_columns=["Open", "High", "Low", "Close", "Volume_(Currency)"]),
        PredicateProcedureKeras(
            KerasLinear1D(
                saved_model_path=saved_model_path,
                model=open(saved_model_path, "rb")
            )
        )
    ).execute()
