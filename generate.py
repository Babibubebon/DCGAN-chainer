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

def generate_image(G, vec):
    x = G(vec, test=True)
    x = chainer.cuda.to_cpu(x.data)
    x = (np.clip(x, -1, 1) + 1) * 128
    return [Image.fromarray(img.astype(np.uint8).transpose(1, 2, 0)) for img in x]  

def main():
    parser = argparse.ArgumentParser(description='DCGAN with chainer')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='out_image',
                        help='Directory to output the result')
    parser.add_argument('gen_model',
                        help='Initialize generator from given file')
    parser.add_argument('dis_model',
                        help='Initialize discriminator from given file')
    args = parser.parse_args()

    try:
        os.mkdir(args.out)
    except:
        pass

    G = Generator(nz)
    D = Discriminator()

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        G.to_gpu()
        D.to_gpu()
    xp = np if args.gpu < 0 else chainer.cuda.cupy

    print('Load model from', args.gen_model, args.dis_model)
    serializers.load_npz(args.gen_model, G)
    serializers.load_npz(args.dis_model, D)

    size = 10
    z = Variable(xp.random.uniform(-1, 1, (size**2, nz)).astype(np.float32))
    images = generate_image(G, z)

    fig = plt.figure(figsize=(6.4, 6.4))
    for i, img in enumerate(images):
        ax = plt.subplot(size, size, i + 1)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        plt.axis('off')
        plt.imshow(img)
    fig.subplots_adjust(wspace=0, hspace=0)
    fig.tight_layout(pad=0)

    plt.savefig(os.path.join(args.out, 'table.png'))
    plt.show()

if __name__ == "__main__":
    main()
