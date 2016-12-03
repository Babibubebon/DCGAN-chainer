import math

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable

class Generator(chainer.Chain):
    def __init__(self, n_in):
        super(Generator, self).__init__(
            l0z = L.Linear(n_in, 4 * 4 * 512),
            dc1 = L.Deconvolution2D(512, 256, 4, stride=2, pad=1),
            dc2 = L.Deconvolution2D(256, 128, 4, stride=2, pad=1),
            dc3 = L.Deconvolution2D(128, 64, 4, stride=2, pad=1),
            dc4 = L.Deconvolution2D(64, 3, 4, stride=2, pad=1),
            bn0l = L.BatchNormalization(4 * 4 * 512),
            bn0 = L.BatchNormalization(512),
            bn1 = L.BatchNormalization(256),
            bn2 = L.BatchNormalization(128),
            bn3 = L.BatchNormalization(64),
        )

    def __call__(self, z, test=False):
        h = F.reshape(F.relu(self.bn0l(self.l0z(z), test=test)), (z.data.shape[0], 512, 4, 4))
        h = F.relu(self.bn1(self.dc1(h), test=test))
        h = F.relu(self.bn2(self.dc2(h), test=test))
        h = F.relu(self.bn3(self.dc3(h), test=test))
        x = F.tanh(self.dc4(h))
        return x


class Discriminator(chainer.Chain):
    def __init__(self):
        super(Discriminator, self).__init__(
            c0 = L.Convolution2D(3, 64, 4, stride=2, pad=1),
            c1 = L.Convolution2D(64, 128, 4, stride=2, pad=1),
            bn1 = L.BatchNormalization(128),
            c2 = L.Convolution2D(128, 256, 4, stride=2, pad=1),
            bn2 = L.BatchNormalization(256),
            c3 = L.Convolution2D(256, 512, 4, stride=2, pad=1),
            bn3 = L.BatchNormalization(512),
            fc4 = L.Linear(4 * 4 * 512, 1)
        )

    def __call__(self, x, test=False):
        h = F.leaky_relu(self.c0(x))
        h = F.leaky_relu(self.bn1(self.c1(h), test=test))
        h = F.leaky_relu(self.bn2(self.c2(h), test=test))
        h = F.leaky_relu(self.bn3(self.c3(h), test=test))
        l = self.fc4(h)
        return l
