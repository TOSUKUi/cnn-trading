import chainer
import chainer.functions as F
import chainer.links as L
from pipeline import Procedure



class ModelProcedure(chainer.Chain, Procedure):
    
    def run(self, x):
        return self.__call__(x)


class TradngModel2D(ModelProcedure):

    def __init__(self):
        super(TradingModel2D, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 64, ksize=4, strides=4)
            self.normalize1 = L.BatchNormalization(None)
            self.conv2 = L.Convolution2D(64, 512, ksize=4, strides=4)
            self.conv3 = L.Convolution2D(512, 1920, 4, 4)
            self.linear1 = L.Linear(None, 1280)
            self.linear2 = L.Linear(1280, 1280)
            self.linear3 = L.Linear(1280, 1)
        
    def __call__(self, x):
        h1 = self.conv1(x)
        h2 = self.normalize1(h1)
        h3 = F.relu(h2)
        h4 = F.max_pooling_2d(h3, 4, 2)
        h5 = F.max_pooling_2d(F.relu(self.conv2(h4)), 4, 2)
        h6 = F.max_pooling_2d(F.relu(self.conv3(h5)), 4, 2)
        l1 = self.linear1(h6)
        l2 = self.linear2(l1)
        return self.linear3(l2)


class TradingModel1D(ModelProcedure):

    def __init__(self):
        super(TradingModel1D, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution1D(None, 64, ksize=4, stride=4)
            self.conv2 = L.Convolution1D(64, 512, ksize=4, stride=4)
            self.conv3 = L.Convolution1D(512, 1920, 4, 4)
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
        l1 = self.linear1(h6)
        l2 = self.linear2(l1)
        return self.linear3(l2)


    








