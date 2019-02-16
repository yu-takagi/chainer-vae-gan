import sys
import os
from encdec import util, VaeGan
from chainer import cuda, Variable
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

def main(args):
    if args.gpu is not None:
        cuda.get_device(args.gpu).use()

    # load model
    vaegan = VaeGan.load(args.model_dir + "/epoch" + str(args.epoch))
    xs_gen = vaegan.generate(args.z_dim,args.n_sample)
    xs_gen = np.reshape(xs_gen.data, (args.n_sample, 3, 64, 64)).transpose(0,2,3,1)
    print >> sys.stderr, 'Plotting...'
    plot(xs_gen,args.epoch,args.model_dir,args.save_dir)

def plot(x,epoch,model_name,save_dir):
    width = x.shape[0]
    fig, axis = plt.subplots(1, width, sharex=True, sharey=True)
    for i, image in enumerate(x):
        ax = axis[i]
        ax.imshow(image)
        ax.axis('off')
    plt.show()
    plt.savefig('./{}/{}_epoch{}_gen.png'.format(save_dir,model_name,epoch))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate Celeba from trained model')

    parser.add_argument('--model_dir', type=str, default="", help='trained model directory')    
    parser.add_argument('--save_dir', type=str, default="img", help='save directory')
    parser.add_argument('--epoch', type=int, default=0, help='number of epoch trained')
    parser.add_argument('--n-sample', type=int, default=11, help='number of samples to generate images')
    parser.add_argument('--z-dim', type=int, default=2048, help='number of hidden units')
    parser.add_argument('--gpu', type=int, default=None, help='GPU ID (default: use CPU)')

    main(parser.parse_args())