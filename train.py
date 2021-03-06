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
from chainer import training
from chainer.training import extensions

from net import Generator, Discriminator
from generate import ext_output_samples

# network
nz  = 100 # of dim for Z
ngf = 512 # of gen filters in first conv layer
ndf = 64  # of discrim filters in first conv layer
nc  = 3   # image channels
size = 64 # size of output image
# optimizer
learning_rate = 0.001
beta1 = 0.5
lr_decay = {
    "rate": 0.85,
    "target": 0.0001,
    "trigger": (1000, 'iteration')
}
weight_decay = 1e-5
gradient_clipping = 100

class Dataset(chainer.datasets.ImageDataset):
    def get_example(self, i):
        path = os.path.join(self._root, self._paths[i])
        f = Image.open(path).convert('RGB')
        return self.preprocess(f)

    def preprocess(self, image):
        cimg = np.asarray(image, dtype=np.float32).transpose(2, 0, 1)
        rnd = np.random.randint(2)
        if rnd == 1:
            # flip
            cimg = cimg[:,:,::-1]
        return (cimg - 128) / 128

class DCGANUpdater(chainer.training.StandardUpdater):
    def update_core(self):
        x_batch = self.converter(self._iterators['main'].next(), self.device)
        z_batch = self.converter(np.random.uniform(-1, 1, (len(x_batch), nz)).astype(np.float32), self.device)

        G_optimizer = self._optimizers['generator']
        D_optimizer = self._optimizers['discriminator']

        G_loss_func = G_optimizer.target.get_loss_func(D_optimizer.target)
        D_loss_func = D_optimizer.target.get_loss_func(G_optimizer.target)

        G_optimizer.update(G_loss_func, Variable(z_batch))
        D_optimizer.update(D_loss_func, Variable(x_batch), Variable(z_batch))


def main():
    parser = argparse.ArgumentParser(description='DCGAN with chainer')
    parser.add_argument('--batchsize', '-b', type=int, default=128,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=1000,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--initmodel', '-m', default='', nargs=2,
                        help='Initialize the model from given file')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the optimization from snapshot')
    parser.add_argument('image_dir', default='images', help='Directory of training data')
    parser.add_argument('--test', action='store_true', default=False)
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))

    # check paths
    if not os.path.exists(args.image_dir):
        sys.exit('image_dir does not exist.')

    # Set up a neural network to train
    G = Generator(ngf, nz, nc, size)
    D = Discriminator(ndf)

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        G.to_gpu()
        D.to_gpu()
    xp = np if args.gpu < 0 else chainer.cuda.cupy

    # Setup an optimizer
    G_optimizer = chainer.optimizers.Adam(alpha=learning_rate, beta1=beta1)
    D_optimizer = chainer.optimizers.Adam(alpha=learning_rate, beta1=beta1)
    G_optimizer.use_cleargrads()
    D_optimizer.use_cleargrads()
    G_optimizer.setup(G)
    D_optimizer.setup(D)
    if weight_decay:
        G_optimizer.add_hook(chainer.optimizer.WeightDecay(weight_decay))
        D_optimizer.add_hook(chainer.optimizer.WeightDecay(weight_decay))
    if gradient_clipping:
        G_optimizer.add_hook(chainer.optimizer.GradientClipping(gradient_clipping))
        D_optimizer.add_hook(chainer.optimizer.GradientClipping(gradient_clipping))

    # Init models
    if args.initmodel:
        print('Load model from', args.initmodel)
        serializers.load_npz(args.initmodel[0], G)
        serializers.load_npz(args.initmodel[1], D)

    # Load dataset
    files = os.listdir(args.image_dir)
    dataset = Dataset(files, args.image_dir)
    dataset_iter = chainer.iterators.MultiprocessIterator(dataset, args.batchsize)
    print('# samples: {}'.format(len(dataset)))

    # Set up a trainer
    optimizers = {'generator': G_optimizer, 'discriminator': D_optimizer}
    updater = DCGANUpdater(dataset_iter, optimizers, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    if lr_decay:
        trainer.extend(extensions.ExponentialShift(
            'alpha', rate=lr_decay["rate"], target=lr_decay["target"], optimizer=G_optimizer),
            trigger=lr_decay["trigger"])
        trainer.extend(extensions.ExponentialShift(
            'alpha', rate=lr_decay["rate"], target=lr_decay["target"], optimizer=D_optimizer),
            trigger=lr_decay["trigger"])

    log_interval = (100, 'iteration') if args.test else (1, 'epoch')
    snapshot_interval = (1000, 'iteration') if args.test else (1, 'epoch')
    suffix = '_{0}_{{.updater.{0}}}'.format(log_interval[1])

    trainer.extend(extensions.snapshot(
        filename='snapshot' + suffix), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        G, 'gen' + suffix), trigger=log_interval)
    trainer.extend(extensions.snapshot_object(
        D, 'dis' + suffix), trigger=log_interval)
    trainer.extend(ext_output_samples(
        10, 'samples' + suffix, seed=0), trigger=log_interval)

    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'generator/loss', 'discriminator/loss', 'elapsed_time']), trigger=log_interval)
    trainer.extend(extensions.ProgressBar(update_interval=20))

    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)

    # Run the training
    trainer.run()


if __name__ == '__main__':
    main()
