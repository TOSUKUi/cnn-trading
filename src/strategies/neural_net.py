import chainer
import chainer.functions as F
import chainer.links as L
from pipeline import Procedure
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Conv2D, Flatten, Dense, MaxPool2D, BatchNormalization
from tensorflow.python.keras.layers import Conv1D, MaxPool1D 
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.utils import plot_model, to_categorical
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf
import numpy as np


class CNNModel:

    def train(self, X, y, saved_model_path, batch_size=1, epochs=100,  train_split=0.8, verbose=1, min_delta=.0005, patience=5, use_early_stop=True):
        """
        Args:
            train: list of traininig data
            validate: list of validation data
            saved_model_path: saved previous model path
        """
        # checkpoint to save model after each epoch
        save_best = ModelCheckpoint(saved_model_path,
                                    monitor='val_loss', 
                                    verbose=verbose,
                                    save_best_only=True,
                                    mode='min')

        # stop training if the validation error stops improving.
        early_stop = EarlyStopping(monitor='val_loss',
                                    min_delta=min_delta,
                                    patience=patience,
                                    verbose=verbose,
                                    mode='auto')

        callbacks_list = [save_best]
        if use_early_stop:
            callbacks_list.append(early_stop)

        hist = self.model.fit(
            X,
            y,
            #batch_size=batch_size,
            epochs=epochs,
            #verbose=1,
            #validation_split=1 - train_split,
            #callbacks=callbacks_list,
        )
        return hist


class ModelProcedureChainer(chainer.Chain, Procedure):

    
    def run(self, x):
        return self.__call__(x)


class ModelProcedureKeras():

    def train(self, X, y, saved_model_path, batch_size=8, epochs=100,  train_split=0.8, verbose=1, min_delta=.0005, patience=5, use_early_stop=True):
        """
        Args:
            train: list of traininig data
            validate: list of validation data
            saved_model_path: saved previous model path
        """
        # checkpoint to save model after each epoch
        save_best = ModelCheckpoint(saved_model_path,
                                    monitor='val_loss', 
                                    verbose=verbose,
                                    save_best_only=True,
                                    mode='min')

        # stop training if the validation error stops improving.
        early_stop = EarlyStopping(monitor='val_loss',
                                    min_delta=min_delta,
                                    patience=patience,
                                    verbose=verbose,
                                    mode='auto')

        callbacks_list = [save_best]
        if use_early_stop:
            callbacks_list.append(early_stop)

        hist = self.model.fit(
            X,
            y,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_split=0.2,
            callbacks=callbacks_list,
        )
        return hist

    def run(self, x):
        return self.model.predict(x)


class TradingModel2D(ModelProcedureChainer):

    def __init__(self):
        super(TradingModel2D, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 64, ksize=4, stride=4)
            # self.normalize1 = L.BatchNormalization(None)
            self.conv2 = L.Convolution2D(64, 512, ksize=4, stride=4)
            self.conv3 = L.Convolution2D(512, 1920, 4, 4)
            self.linear1 = L.Linear(128, 128)
            self.linear2 = L.Linear(128, 128)
            self.linear3 = L.Linear(128, 1)
        
    def __call__(self, x):
        h1 = self.conv1(x)
        # h2 = self.normalize1(h1)
        h3 = F.relu(h1)
        h4 = F.max_pooling_2d(h3, 4, 2)
        h5 = F.max_pooling_2d(F.relu(self.conv2(h4)), 4, 2)
        h6 = F.max_pooling_2d(F.relu(self.conv3(h5)), 4, 2)
        l1 = F.sigmoid(self.linear1(h6))
        l2 = F.sigmoid(self.linear2(l1))
        return self.linear3(l2)


class TradingModel1D(ModelProcedureChainer):

    def __init__(self):
        super(TradingModel1D, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution1D(None, 64, ksize=4, stride=4)
            self.conv2 = L.Convolution1D(None, 512, ksize=4, stride=4)
            self.conv3 = L.Convolution1D(None, 1920, 4, 4)
            self.linear1 = L.Linear(None, 1280)
            self.linear2 = L.Linear(1280, 1280)
            self.linear3 = L.Linear(1280, 1)
        
    def __call__(self, x):
        h1 = self.conv1(x)
        # h2 = self.normalize1(h1)
        h3 = F.relu(h1)
        h4 = F.max_pooling_2d(h3, 4, 2)
        h5 = F.max_pooling_2d(F.relu(self.conv2(h4)), 4, 2)
        h6 = F.max_pooling_2d(F.relu(self.conv3(h5)), 4, 2)
        l1 = F.relu(self.linear1(h6))
        l2 = F.relu(self.linear2(l1))
        return self.linear3(l2)

    def run(self, x):
        return self.__call__(x)


class KerasLinear(CNNModel):
    def __init__(self, model=None, num_outputs=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if model:
            self.model = model
        elif num_outputs is not None:
            self.model = self.linear()
        else:
            self.model = self.linear()
    

    def run(self, img_arr):
        img_arr = img_arr.reshape((1,) + img_arr.shape)
        output = self.model.predict(img_arr)
        return np.argmax(output[0])

    def train(self, X, y, *args, **kwargs):
        return super().train(X, y, *args, **kwargs)

    def linear(self):
        inputs = Input(shape=(5, 60, 1), name='inputs')
        x = Conv2D(filters=15, kernel_size=3, strides=1, activation='relu')(inputs)
        x = MaxPool2D(pool_size=3, strides=2)(x)
        x = BatchNormalization()(x)
        x = Conv2D(filters=24, kernel_size=3, strides=2, activation='relu')(x)
        x = MaxPool2D(pool_size=3, strides=2)(x)
        x = Flatten(name='flattened')(x)
        x = Dense(units=100, activation='linear')(x)
        x = Dense(units=100, activation='linear')(x)
        handling = Dense(units=1, activation='', name='handling')(x)
        model = Model(inputs=[inputs], outputs=[handling])
        model.compile(optimizer='adam', loss={'handling':'mean_squared_error'}, loss_weights={'handling': 0.05})
        return model


class KerasLinear1D(CNNModel):

    def __init__(self, saved_model_path, model=None, num_outputs=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.saved_model_path = saved_model_path
        if model:
            self.model = model
        elif num_outputs is not None:
            self.model = self.linear()
        else:
            self.model = self.linear()

    def run(self, img_arr):
        output = self.model.predict(img_arr)
        return np.argmax(output[0])

    def train(self, X, y, *args, **kwargs):
        return super().train(X, y, self.saved_model_path, *args, **kwargs)

    def linear(self):
        inputs = Input(shape=(5, 1800), name='inputs')
        x = Conv1D(filters=6, kernel_size=3, strides=1, activation=tf.nn.relu)(inputs)
        # x = MaxPool1D(pool_size=3, strides=2)(x)
        # x = BatchNormalization()(x)
        # # x = Conv1D(filters=6, kernel_size=3, strides=2, activation='relu')(x)
        # # x = MaxPool1D(pool_size=3, strides=2)(x)
        x = Flatten(name='flattened')(x)
        # x = Dense(units=100, activation=tf.nn.relu)(x)
        # x = Dense(units=100)(x)
        handling = Dense(units=1, name='output')(x)
        model = Model(inputs=[inputs], outputs=[handling])
        model.compile(optimizer='adam', loss={'output': 'mean_squared_error'}, loss_weights={'output': 0.05})
        return model