from functools import reduce
from abc import ABCMeta, abstractmethod

class PipeLine():

    def execute(self):
        return reduce(lambda x,y: y.run(x), self.pipeline)


class TrainPipeLine(PipeLine):

    def __init__(self, *pipeline):
        self.pipeline = pipeline 


class Procedure(metaclass=ABCMeta):

    @abstractmethod
    def run(self, x):
        pass
