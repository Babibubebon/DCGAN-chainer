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
size = 96

def generate_image(G, z):
    if G.xp != np:
        z.to_gpu()
    x = G(z, test=True)
    x = chainer.cuda.to_cpu(x.data)
    x = np.clip((x + 1) * 128, 0, 255)
    return [Image.fromarray(img.astype(np.uint8).transpose(1, 2, 0)) for img in x]  

def plot(images, table_size=10, out=None, show=False):
    if type(table_size) == int:
        table_w, table_h = table_size, table_size
    else:
        table_w, table_h = table_size
    img_w, img_h = images[0].size

    fig = plt.figure(figsize=(img_w*table_w/100, img_h*table_h/100), dpi=100)
    for i, img in enumerate(images):
        ax = plt.subplot(table_w, table_h, i + 1)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        plt.axis('off')
        plt.imshow(img)
    fig.subplots_adjust(wspace=0, hspace=0)
    fig.tight_layout(pad=0)
    if out:
        fig.savefig(out)
    if show:
        plt.show()
    plt.close()


def random_image(G, n, random=np.random):
    z = Variable(random.uniform(-1, 1, (n, nz)).astype(np.float32))
    images = generate_image(G, z)
    return images

def similar_image(G, n, scale=0.3, random=np.random):
    base = random.uniform(-1, 1, (nz)).astype(np.float32)
    diff = random.normal(0, scale, (n, nz)).astype(np.float32)
    z = Variable(np.array([base + v for v in diff]).clip(-1, 1))
    images = generate_image(G, z)
    return images

def interp_image (G, n, random=np.random):
    w, h = (n, n) if type(n) == int else n
    a = random.uniform(-1, 1, (h, nz)).astype(np.float32)
    b = random.uniform(-1, 1, (h, nz)).astype(np.float32)
    step = np.arange(w) / w
    z = Variable(np.array([v + t*(w - v) for (v, w) in zip(a, b) for t in step]))
    images = generate_image(G, z)
    return images

def mean_image(G, n, random=np.random):
    z = random.uniform(-1, 1, (n, nz)).astype(np.float32)
    mean = np.array([z.mean(0)])
    images = generate_image(G, Variable(z))
    mean_image = generate_image(G, Variable(mean))
    return images, mean_image


def ext_output_samples(
        table_size, out_name='samples_epoch_{.updater.epoch}.png', seed=0, trigger=(1, 'epoch')
    ):
    @chainer.training.make_extension(trigger=trigger)
    def func(trainer):
        G = trainer.updater.get_optimizer('generator').target
        state = random_state = np.random.RandomState(seed)
        images = random_image(G, table_size**2, state)
        plot(images, table_size, os.path.join(trainer.out, out_name.format(trainer)), show=False)
    return func


def main():
    parser = argparse.ArgumentParser(description='DCGAN with chainer')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--size', '-n', type=int, default=10,
                        help='size of output image table')
    parser.add_argument('--times', '-n2', type=int, default=1,
                        help='times of generate image')
    parser.add_argument('--seed', '-s', type=int, default=None,
                        help='random seed')
    parser.add_argument('--type', '-t', default='random',
                        choices=['random', 'interp', 'similar', 'mean'],
                        help='generate type')
    parser.add_argument('--quiet', '-q', action='store_true', default=False)
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

    np.random.seed(args.seed)

    for i in range(args.times):
        if args.type == 'random':
            images = random_image(G, args.size**2)
        if args.type == 'similar':
            images = similar_image(G, args.size**2)
        if args.type == 'interp':
            images = interp_image(G, args.size)
        if args.type == 'mean':
            images, mean = mean_image(G, args.size**2)
            plot(mean, 1, os.path.join(args.out, 'mean_{}.png'.format(i)), show=not args.quiet)

        plot(images,
            args.size,
            os.path.join(args.out, '{}_{}.png'.format(args.type, i)),
            show=not args.quiet)

if __name__ == "__main__":
    main()
