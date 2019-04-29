from pipeline import PipeLine, TrainPipeLine
from read_data import ReadData
from preprocessing import PreprocessingProcedure2D, PreprocessingProcedure1D
from strategies.training import TrainProcedureChainer, TrainProcedureKeras
from strategies.neural_net import *

import sys
sys.path.append("/Users/TOSUKUi/Documents/workspace/trading-deeplearning")



def main():
    TrainPipeLine(
        'data/bitstampUSD_1-min_data_2012-01-01_to_2019-03-13.csv', 
        ReadData(), 
        PreprocessingProcedure1D(), 
        TrainProcedureKeras(KerasLinear1D(saved_model_path="models/saved_model.h5"))
    ).execute()


main()
