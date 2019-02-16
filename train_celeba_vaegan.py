#!/usr/bin/env python
from encdec import util, train
from encdec.nn.vaegan import VaeGan
from encdec.nn.celeba_vaegan import Encoder, Decoder, Discriminator
import os
import logging
from chainer import cuda, Variable
import chainer.functions

def main(args):
    if args.gpu is not None:
        cuda.get_device(args.gpu).use()

    try:
        os.makedirs(args.model)
    except:
        pass

    # set up logger
    logger = logging.getLogger()
    logging.basicConfig(level=logging.INFO)
    log_path = os.path.join(args.model, 'log')
    file_handler = logging.FileHandler(log_path)
    fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    # load data
    logger.info('Loading data...')
    dataset = util.load_celeba(args.data_home,args.data_size)
    assert len(dataset) > 0
    print len(dataset)

    # save hyperparameters
    with open(os.path.join(args.model, 'params'), 'w') as f:
        for k, v in vars(args).items():
            print >> f, '{}\t{}'.format(k, v)

    # create batches
    active = getattr(chainer.functions, args.active)
    encoder = Encoder(args.z)
    decoder = Decoder(args.z)
    discriminator = Discriminator(args.z)
    vaegan = VaeGan(encoder, decoder, discriminator)

    vaegan.save_model_def(args.model)

    train.train(vaegan, dataset, args.batch, args.optim, dest_dir=args.model, 
                max_epoch=args.epoch, gpu=args.gpu, save_every=args.save_every,
                alpha_init=args.alpha_init, alpha_delta=args.alpha_delta,
                n_train=args.n_train, gamma=args.gamma)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train VAEGAN for Celeba')

    parser.add_argument('model', help='destination of model')

    # NN architecture
    parser.add_argument('--z', type=int, default=2048, help='dimension of hidden variable')
    parser.add_argument('--hidden', nargs='+', type=int, default=[512, 512], help='size of hidden layers of recognition/generation models')
    parser.add_argument('--active', default='relu', help='activation function between hidden layers')

    # training options
    parser.add_argument('--train-size', type=int, default=60000, help='number of training samples')
    parser.add_argument('--test-size', type=int, default=10000, help='number of test samples')
    parser.add_argument('--batch', type=int, default=64, help='batch size')
    parser.add_argument('--epoch', type=int, default=50, help='number of epochs to train')
    parser.add_argument('--n-train', type=int, default=50, help='number of epochs to train in each epoch')
    parser.add_argument('--optim', nargs='+', default=['RMSprop','0.0003'], help='optimization method supported by chainer (optional arguments can be omitted)')
    parser.add_argument('--gpu', type=int, default=None, help='GPU ID (default: use CPU)')
    parser.add_argument('--save-every', type=int, default=10, help='save model every this number of epochs')
    parser.add_argument('--data-home', type=str, default='../../dat/celeba/', help='data home directory')
    parser.add_argument('--data-size', type=str, default='new', help='data home directory')

    parser.add_argument('--alpha-init', type=float, default=1., help='initial value of weight of KL loss')
    parser.add_argument('--alpha-delta', type=float, default=0.001, help='delta value of weight of KL loss')
    parser.add_argument('--gamma', type=float, default=0.1, help='initial value of weight of KL loss')

    main(parser.parse_args())


