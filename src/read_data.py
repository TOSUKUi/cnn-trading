from pipeline import Procedure
import pandas as pd

class ReadData(Procedure):

    def run(self, x):
        return self.execute(x)

    def execute(self, path):
        return  pd.read_csv(path)
        
