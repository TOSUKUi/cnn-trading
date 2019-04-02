import chainer
import chainer.functions as F
import chainer.links as L


class TraidingModel(chainer.Chain):
    
    def __init__(sefl, n_in, n_hidden1, n_hidden2, n_out):
        super(TraidingModel, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 64, ksize=4, strides=4)
            self.conv2 = L.Convolution2D(64, 512, ksize=4, strides=4)
            self.conv3 = L.Convolution2D(512, 1920, 4, 4)
            self.normalize1 = L.BatchNormalization()

self.conv1
            
            
            




def unko:
    pass

