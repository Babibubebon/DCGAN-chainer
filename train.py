#!/usr/bin/env python
import argparse
import os
import sys
import numpy as np
from PIL import Image

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable
from chainer import serializers

from net import Generator, Discriminator

nz = 100          # # of dim for Z

class Dataset(chainer.datasets.ImageDataset):
    def get_example(self, i):
        path = os.path.join(self._root, self._paths[i])
        f = Image.open(path).convert('RGB')
        image = np.asarray(f, dtype=np.float32).transpose(2, 0, 1)

        rnd = np.random.randint(2)
        if rnd == 1:
            image = image[:,:,::-1]

        image = (image - 128.0) / 128.0
        return image


def main():
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--initmodel', '-m', default='',
                        help='Initialize the model from given file')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the optimization from snapshot')
    parser.add_argument('image_dir', default='images', help='Directory of training data')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    # check paths
    if not os.path.exists(args.image_dir):
        sys.exit('image_dir does not exist.')
    try:
        os.mkdir(args.out)
    except:
        pass

    # Set up a neural network to train
    G = Generator(nz)
    D = Discriminator()

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        G.to_gpu()
        D.to_gpu()
    xp = np if args.gpu < 0 else chainer.cuda.cupy

    # Setup an optimizer
    G_optimizer = chainer.optimizers.Adam(alpha=2e-4, beta1=0.5)
    D_optimizer = chainer.optimizers.Adam(alpha=2e-4, beta1=0.5)
    G_optimizer.use_cleargrads()
    D_optimizer.use_cleargrads()
    G_optimizer.setup(G)
    D_optimizer.setup(D)
    G_optimizer.add_hook(chainer.optimizer.WeightDecay(1e-5))
    D_optimizer.add_hook(chainer.optimizer.WeightDecay(1e-5))

    # Init/Resume
    if args.initmodel:
        print('Load model from', args.initmodel)
        serializers.load_npz(args.initmodel, G)
    if args.resume:
        print('Load optimizer state from', args.resume)
        serializers.load_npz(args.resume, G_optimizer)

    # Load dataset
    files = os.listdir(args.image_dir)
    dataset = Dataset(files, args.image_dir)

    for epoch in range(1, args.epoch + 1):
        print('epoch', epoch)
        dataset_iter = chainer.iterators.SerialIterator(dataset, args.batchsize, repeat=False)

        sum_G_loss = 0
        sum_D_loss = 0

        for data in dataset_iter:
            print(dataset_iter.epoch_detail)
            # train generator
            z = Variable(xp.random.uniform(-1, 1, (args.batchsize, nz)).astype(np.float32))
            x = G(z)
            p_g = D(x)

            G_loss = F.sigmoid_cross_entropy(p_g, Variable(xp.ones((args.batchsize, 1), dtype=np.int32)))
            G.cleargrads()
            G_loss.backward()
            G_optimizer.update()

            # train discriminator
            if args.gpu >= 0:
                p_real = D(Variable(chainer.cuda.to_gpu(data)))
            else:
                p_real = D(Variable(np.array(data)))

            D_loss = F.sigmoid_cross_entropy(p_real, Variable(xp.ones((len(p_real.data), 1), dtype=np.int32)))
            D_loss += F.sigmoid_cross_entropy(p_g, Variable(xp.zeros((len(p_g.data), 1), dtype=np.int32)))
            D.cleargrads()
            D_loss.backward()
            D_optimizer.update()

            sum_G_loss += G_loss.data
            sum_D_loss += G_loss.data

            print('generator loss     : {}'.format(G_loss.data))
            print('discriminator loss : {}'.format(D_loss.data))

            # output image
            z = Variable(xp.random.uniform(-1, 1, (1, nz)).astype(np.float32))
            x = G(z, test=True)
            tmp = chainer.cuda.to_cpu(x.data[0])
            func = np.vectorize(lambda x: np.float32(-1 if x < -1 else (1 if x > 1 else x)))
            tmp = (func(tmp) + 1) * 128
            img = tmp.astype(np.uint8).transpose(1, 2, 0)
            Image.fromarray(img).save("out.png")

            serializers.save_npz(os.path.join(args.out, 'generator_{}.model'.format(epoch)), G)
            serializers.save_npz(os.path.join(args.out, 'discriminator_{}.model'.format(epoch)), D)

if __name__ == '__main__':
    main()
