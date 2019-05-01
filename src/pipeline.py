from functools import reduce
from abc import ABCMeta, abstractmethod
from typing import List


class Procedure(metaclass=ABCMeta):

    @abstractmethod
    def run(self, x):
        pass


class PipeLine():

    def execute(self):
        return reduce(lambda x, y: y.run(x), self.pipeline)

    def __init__(self, first, *pipeline: List[Procedure]):
        self.pipeline = first + pipeline 
