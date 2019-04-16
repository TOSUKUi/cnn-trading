from functools import reduce
class PipeLine():

    def execute(self, x):
        return reduce(lambda x,y: y.run(x), self.pipeline)


class TrainPipeLine(PipeLine):

    def __init__(self, *pipeline):
        self.pipeline = pipeline 


class Procedure():

    def run(self, x):
        pass
