import argparse
from pipeline import  Procedure



class TrainProcedureKeras(Procedure):

    def __init__(self, model, **kwargs): 
        self.model = model
        self.kwargs = kwargs
    
    def run(self, *data):
        return self.train_model(*data)

    def train_model(self, *data):
        return self.model.train(*data, **self.kwargs)
    