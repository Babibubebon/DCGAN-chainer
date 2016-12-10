#!/usr/bin/env python
import argparse
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import chainer
from chainer import Variable
from chainer import serializers

from net import Generator, Discriminator

nz = 100
ngf = 512
ndf = 64
nc = 3
size = 64

def generate_image(G, z):
    x = G(z, test=True)
    x = chainer.cuda.to_cpu(x.data)
    x = np.clip((x + 1) * 128, 0, 255)
    return [Image.fromarray(img.astype(np.uint8).transpose(1, 2, 0)) for img in x]  

def output_samples(G, table_size, out_name, seed=None, show=False):
    np.random.seed(seed)
    z = Variable(np.random.uniform(-1, 1, (table_size**2, nz)).astype(np.float32))
    if G.xp != np:
        z.to_gpu()
    images = generate_image(G, z)

    fig = plt.figure(figsize=(size/10, size/10), dpi=100)
    for i, img in enumerate(images):
        ax = plt.subplot(table_size, table_size, i + 1)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        plt.axis('off')
        plt.imshow(img)
    fig.subplots_adjust(wspace=0, hspace=0)
    fig.tight_layout(pad=0)
    fig.savefig(out_name)
    if show:
        plt.show()
    plt.close()

def ext_output_samples(table_size, out_name='samples_epoch_{.updater.epoch}.png', seed=0, trigger=(1, 'epoch')):
    @chainer.training.make_extension(trigger=trigger)
    def func(trainer):
        G = trainer.updater.get_optimizer('generator').target
        output_samples(G, table_size, os.path.join(trainer.out, out_name.format(trainer)), seed)
    return func

def main():
    parser = argparse.ArgumentParser(description='DCGAN with chainer')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--size', '-n', default=10,
                        help='size of output image table')
    parser.add_argument('gen_model',
                        help='Initialize generator from given file')
    args = parser.parse_args()

    try:
        os.mkdir(args.out)
    except:
        pass

    G = Generator(ngf, nz, nc, size)

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        G.to_gpu()

    print('Load model from', args.gen_model)
    serializers.load_npz(args.gen_model, G)

    output_samples(G, args.size, os.path.join(args.out, 'table.png'), show=True)

if __name__ == "__main__":
    main()
