import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable

class Generator(chainer.Chain):
    def __init__(self, n_in):
        initW = chainer.initializers.Normal(0.02)
        super(Generator, self).__init__(
            l0 = L.Linear(n_in, 4 * 4 * 512),
            dc1 = L.Deconvolution2D(512, 256, 4, stride=2, pad=1, initialW=initW),
            bn0 = L.BatchNormalization(4 * 4 * 512),
            dc2 = L.Deconvolution2D(256, 128, 4, stride=2, pad=1, initialW=initW),
            bn1 = L.BatchNormalization(256),
            dc3 = L.Deconvolution2D(128, 64, 4, stride=2, pad=1, initialW=initW),
            bn2 = L.BatchNormalization(128),
            dc4 = L.Deconvolution2D(64, 3, 4, stride=2, pad=1, initialW=initW),
            bn3 = L.BatchNormalization(64),
        )

    def __call__(self, z, test=False):
        h = F.reshape(F.relu(self.bn0(self.l0(z), test=test)), (z.data.shape[0], 512, 4, 4))
        h = F.relu(self.bn1(self.dc1(h), test=test))
        h = F.relu(self.bn2(self.dc2(h), test=test))
        h = F.relu(self.bn3(self.dc3(h), test=test))
        x = F.tanh(self.dc4(h))
        return x

    def get_loss_func(self, D):
        G = self
        def lf(z):
            p_g = D(G(z))
            loss = F.sigmoid_cross_entropy(
                p_g, Variable(self.xp.ones((z.data.shape[0], 1), dtype=self.xp.int32)))

            chainer.report({ 'loss': chainer.cuda.to_cpu(loss.data) }, self)
            return loss
        return lf


class Discriminator(chainer.Chain):
    def __init__(self):
        initW = chainer.initializers.Normal(0.02)
        super(Discriminator, self).__init__(
            c0 = L.Convolution2D(3, 64, 4, stride=2, pad=1, initialW=initW),
            c1 = L.Convolution2D(64, 128, 4, stride=2, pad=1, initialW=initW),
            bn1 = L.BatchNormalization(128),
            c2 = L.Convolution2D(128, 256, 4, stride=2, pad=1, initialW=initW),
            bn2 = L.BatchNormalization(256),
            c3 = L.Convolution2D(256, 512, 4, stride=2, pad=1, initialW=initW),
            bn3 = L.BatchNormalization(512),
            fc4 = L.Linear(4 * 4 * 512, 1, initialW=initW)
        )

    def __call__(self, x, test=False):
        h = F.leaky_relu(self.c0(x))
        h = F.leaky_relu(self.bn1(self.c1(h), test=test))
        h = F.leaky_relu(self.bn2(self.c2(h), test=test))
        h = F.leaky_relu(self.bn3(self.c3(h), test=test))
        l = self.fc4(h)
        return l

    def get_loss_func(self, G):
        D = self
        def lf(x, z):
            p_real = D(x)
            p_g = D(G(z))
            loss = F.sigmoid_cross_entropy(
                p_real, Variable(self.xp.ones((x.data.shape[0], 1), dtype=self.xp.int32)))
            loss += F.sigmoid_cross_entropy(
                p_g,    Variable(self.xp.zeros((z.data.shape[0], 1), dtype=self.xp.int32)))
            
            chainer.report({ 'loss': chainer.cuda.to_cpu(loss.data) }, self)
            return loss
        return lf
