from pipeline import PipeLine, TrainPipeLine
from read_data import ReadData
from preprocessing import PreprocessingProcedure2D
from strategies.training import TrainProcedure
from strategies.neural_net import TradingModel1D, TradingModel2D, MLP
import sys
sys.path.append("/Users/TOSUKUi/Documents/workspace/trading-deeplearning")



def main():
    TrainPipeLine(
        'data/bitstampUSD_1-min_data_2012-01-01_to_2019-03-13.csv', 
        ReadData(), 
        PreprocessingProcedure2D(), 
        TrainProcedure(MLP(200, 1))
    ).execute()


main()
