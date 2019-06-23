from pipeline import Procedure
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Conv2D, Flatten, Dense, MaxPool2D, BatchNormalization, Dropout, GlobalAveragePooling2D
from tensorflow.python.keras.layers import Conv1D, MaxPool1D 
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU
from tensorflow.python.keras.applications.inception_v3 import InceptionV3
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.utils import plot_model, to_categorical
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf
import numpy as np
import os



class CNNModel(Procedure):

    def train(self, X, y, saved_model_path, batch_size=64, epochs=100,  train_split=0.8, verbose=1, min_delta=.0005, patience=5, use_early_stop=True):
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
            validation_split=1 - train_split,
            callbacks=callbacks_list,
        )
        return hist


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
        inputs = Input(shape=(1800, 5), name='inputs')
        x = Conv1D(filters=30, kernel_size=5, strides=2, activation='relu')(inputs)
        x = MaxPool1D(pool_size=3, strides=1)(x)
        x = BatchNormalization()(x)
        x = Conv1D(filters=6, kernel_size=3, strides=1, activation='relu')(x)
        x = Conv1D(filters=6, kernel_size=3, strides=1, activation='relu')(x)
        x = MaxPool1D(pool_size=3, strides=2)(x)
        x = Dropout(rate=0.25)(x)
        x = Flatten(name='flattened')(x)
        x = Dense(units=100, activation=tf.nn.relu)(x)
        x = Dense(units=100)(x)
        handling = Dense(units=1, name='output')(x)
        model = Model(inputs=[inputs], outputs=[handling])
        model.compile(optimizer='adam', loss={'output': 'mean_squared_error'})
        return model


class KerasLinear1DSoftMax(CNNModel):

    def __init__(self, saved_model_path, model=None, num_outputs=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.saved_model_path = saved_model_path
        if model:
            self.model = model
        elif num_outputs is not None:
            self.model = self.softmax()
        else:
            self.model = self.softmax()

    def run(self, img_arr):
        output = self.model.predict(img_arr)
        return np.argmax(output[0])

    def train(self, X, y, *args, **kwargs):
        return super().train(X, y, self.saved_model_path, *args, **kwargs)

    def softmax(self):
        inputs = Input(shape=(60, 5), name='inputs')
        x = Conv1D(filters=6, kernel_size=3, strides=1, activation='relu')(inputs)
        x = Conv1D(filters=6, kernel_size=3, strides=1, activation='relu')(x)
        x = MaxPool1D(pool_size=2, strides=1)(x)
        x = Conv1D(filters=6, kernel_size=3, strides=1, activation='relu')(x)
        x = Flatten(name='flattened')(x)
        x = Dropout(rate=0.25)(x)
        x = Dense(units=100, activation=tf.nn.relu)(x)
        x = Dense(units=100)(x)
        handling = Dense(units=1, name='output', activation='sigmoid')(x)
        model = Model(inputs=[inputs], outputs=[handling])
        model.compile(optimizer='adam', loss={'output': 'binary_crossentropy'}, metrics=['accuracy'])
        return model

        
class ImageConvVGG16(CNNModel):

    def __init__(self, model=None):
        if model:
            self.model = model
        else:
            self.model = self.image_net()

    def run(self, img_arr):
        output = self.model.predict(img_arr)
        return np.argmax(output[0])

    def image_net(self):
        input_shape = (250, 250, 3)
        inputs = Input(shape=input_shape)
        x = Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu')(inputs)
        x = Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu')(x)
        x = MaxPool2D(pool_size=2, strides=1)(x)
        x = Conv2D(filters=24, kernel_size=3, strides=2, activation='relu', padding='same')(x)
        x = Conv2D(filters=24, kernel_size=3, strides=2, activation='relu', padding='same')(x)
        x = MaxPool2D(pool_size=2, strides=2)(x)
        x = Flatten(name='flattened')(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(rate=0.25)(x)
        predictions = Dense(2, activation='softmax')(x)
        model = Model(inputs = inputs, outputs=predictions)
        model.compile(optimizer=tf.train.RMSPropOptimizer(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
        # 環境変数に登録されているTPUサーバーへ接続
        TPU_WORKER = "grpc://" + os.environ["COLAB_TPU_ADDR"]
        strategy = tf.contrib.tpu.TPUDistributionStrategy(
            tf.contrib.cluster_resolver.TPUClusterResolver(TPU_WORKER)
        )

        # Keras ModelをTPU形式へ変換
        tpu_model = tf.contrib.tpu.keras_to_tpu_model(
            model,
            strategy=strategy
        )

        return model