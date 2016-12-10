import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable

class Generator(chainer.Chain):
    def __init__(self, ngf, ch_in, ch_out=3, size=64):
        assert size % 16 == 0, "size must be multiples of 16"

        self.ngf = ngf
        self.fsize = size >> 4
        initW = chainer.initializers.Normal(0.02)
        super(Generator, self).__init__(
            l0  = L.Linear(ch_in, self.fsize ** 2 * ngf),
            bn0 = L.BatchNormalization(self.fsize ** 2 * ngf),
            dc1 = L.Deconvolution2D(None, ngf // 2, 4, stride=2, pad=1, initialW=initW),
            bn1 = L.BatchNormalization(ngf // 2),
            dc2 = L.Deconvolution2D(None, ngf // 4, 4, stride=2, pad=1, initialW=initW),
            bn2 = L.BatchNormalization(ngf // 4),
            dc3 = L.Deconvolution2D(None, ngf // 8, 4, stride=2, pad=1, initialW=initW),
            bn3 = L.BatchNormalization(ngf // 8),
            dc4 = L.Deconvolution2D(None,   ch_out, 4, stride=2, pad=1, initialW=initW),
        )

    def __call__(self, z, test=False):
        h = F.reshape(
            F.relu(self.bn0(self.l0(z), test=test)),
            (z.data.shape[0], self.ngf, self.fsize, self.fsize))
        h = F.relu(self.bn1(self.dc1(h), test=test))
        h = F.relu(self.bn2(self.dc2(h), test=test))
        h = F.relu(self.bn3(self.dc3(h), test=test))
        x = F.tanh(self.dc4(h))
        return x

    def get_loss_func(self, D):
        G = self
        def lf(z):
            p_g = D(G(z), sigmoid=False)
            loss = F.bernoulli_nll(
                Variable(self.xp.ones((z.data.shape[0], 1), dtype=self.xp.float32)), p_g)

            chainer.report({ 'loss': chainer.cuda.to_cpu(loss.data) }, self)
            return loss
        return lf


class Discriminator(chainer.Chain):
    def __init__(self, ndf):
        initW = chainer.initializers.Normal(0.02)
        super(Discriminator, self).__init__(
            c0  = L.Convolution2D(None,     ndf, 4, stride=2, pad=1, initialW=initW),
            c1  = L.Convolution2D(None, ndf * 2, 4, stride=2, pad=1, initialW=initW),
            bn1 = L.BatchNormalization(ndf * 2),
            c2  = L.Convolution2D(None, ndf * 4, 4, stride=2, pad=1, initialW=initW),
            bn2 = L.BatchNormalization(ndf * 4),
            c3  = L.Convolution2D(None, ndf * 8, 4, stride=2, pad=1, initialW=initW),
            bn3 = L.BatchNormalization(ndf * 8),
            l4  = L.Linear(None, 1, initialW=initW)
        )

    def __call__(self, x, sigmoid=True, test=False):
        h = F.leaky_relu(self.c0(x))
        h = F.leaky_relu(self.bn1(self.c1(h), test=test))
        h = F.leaky_relu(self.bn2(self.c2(h), test=test))
        h = F.leaky_relu(self.bn3(self.c3(h), test=test))
        l = self.l4(h)
        if sigmoid:
            return F.sigmoid(l)
        else:
            return l

    def get_loss_func(self, G):
        D = self
        def lf(x, z):
            p_real = D(x, sigmoid=False)
            z.volatile = 'on'
            x_g = G(z)
            x_g.volatile = 'off'
            p_g = D(x_g, sigmoid=False)
            loss = F.bernoulli_nll(
                Variable(self.xp.ones((x.data.shape[0], 1), dtype=self.xp.float32)), p_real)
            loss += F.bernoulli_nll(
                Variable(self.xp.zeros((z.data.shape[0], 1), dtype=self.xp.float32)), p_g)
            
            chainer.report({ 'loss': chainer.cuda.to_cpu(loss.data) }, self)
            return loss
        return lf
