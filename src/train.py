from pipeline import PipeLine, TrainPipeLine
from read_data import ReadData
from preprocessing import PreprocessingProcedure
from strategies.training import TrainProcedure
from strategies.neural_net import TradingModel1D



def main():
    TrainPipeLine(
        'data/bitstampUSD_1-min_data_2012-01-01_to_2019-03-13.csv', 
        ReadData(), 
        PreprocessingProcedure(), 
        TrainProcedure(TradingModel1D())
        )

main()
