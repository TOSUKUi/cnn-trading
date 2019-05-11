## cnn-trading
This is neural net trading application.

Now development but 1st iteration of training is already done.

Now development feature of #1.

This is not a library but customizable easily via add the pipeline code like this.

```python
src/pipelines.py

from core.pipeline import PipeLine 

def train_pipeline():
    saved_model_path = "models/saved_model.h5"
    data_path = 'data/bitstampUSD_1-min_data_2012-01-01_to_2019-03-13.csv',

    # pipeline definition
    PipeLine(
        data_path,
        ReadData(),
        PreprocessingProcedure1D(),
        TrainProcedureKeras(
            KerasLinear1D(
                saved_model_path=saved_model_path
            )
        )
    ).execute()
```
pipeline is defined at src/pipeline.py


PipeLine constructor never take argument which are not extend Procedure except 1st argument.

Each procedure are implement Procedure class.

So you can add procedure easily.
