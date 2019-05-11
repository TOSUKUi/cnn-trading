from read_data import ReadData
from pipeline import PipeLine 
from preprocessing import PreprocessingProcedure1D, TrainingPreprocessingProcedure1D
from strategies.training import TrainProcedureChainer, TrainProcedureKeras
from strategies.neural_net import KerasLinear1D
from strategies.predicate import PredicateProcedureKeras




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


def backtest_pipeline():
    data_path = 'data/bitstampUSD_1-min_data_2012-01-01_to_2019-03-13.csv',
    saved_model_path = "models/saved_model.h5"
    PipeLine(
        data_path,
        ReadData,
        PreprocessingProcedure1D(),
        PredicateProcedureKeras(
            KerasLinear1D(
                saved_model_path=saved_model_path,
                model=open(saved_model_path, "rb")
            )
        )
    ).execute()
