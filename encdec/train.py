import logging
import time
from collections import OrderedDict
import pickle
from chainer import cuda, Variable
import numpy as np
import chainer
from PIL import Image
from io import StringIO
from io import BytesIO
import util
import chainer.functions as F

def train(model, dataset, batchsize, optimizer, dest_dir, max_epoch=None, gpu=None, save_every=1,
              alpha_init=1., alpha_delta=0., n_train=50, gamma=0.1):
    """Common training procedure.

    :param model: model to train
    :param batches: training data
    :param optimizer: chainer optimizer
    :param dest_dir: destination directory
    :param max_epoch: maximum number of epochs to train (None to train indefinitely)
    :param gpu: ID of GPU (None to use CPU)
    :param save_every: save every this number of epochs (first epoch and last epoch are always saved)
    :param get_status: function that takes batch and returns list of tuples of (name, value)
    :param alpha_init: initial value of alpha
    :param alpha_delta: change of alpha at every batch
    """
    if gpu is not None:
        # set up GPU
        cuda.get_device(gpu).use()
        model.to_gpu(gpu)

    logger = logging.getLogger()
    n_dataset = len(dataset)

    # set up optimizer
    opt_enc = util.list2optimizer(optimizer)
    opt_dec = util.list2optimizer(optimizer)
    opt_dis = util.list2optimizer(optimizer)    
    opt_enc.setup(model.encoder)
    opt_dec.setup(model.decoder)
    opt_dis.setup(model.discriminator)

    # training loop
    epoch = 0
    alpha = alpha_init
    test_losses = []
    xp = np if gpu is None else cuda.cupy

    while True:
        if max_epoch is not None and epoch >= max_epoch:
            # terminate training
            break

        for i in range(n_train):
            x_data = np.zeros((batchsize, 3, 64, 64), dtype=np.float32)           
            count = 0
            for j in range(batchsize):
                try:
                    rnd = np.random.randint(n_dataset)
                    img = np.asarray(Image.open(BytesIO(dataset[rnd])))
                    flag = np.random.randint(2)
                    if flag == 0:
                        img_n = img
                    else:
                        img_n = util.noise_filter(img)
                    img_n_r = util.img_resize(img_n,bbox=(40, 218-30, 15, 178-15),rescale_size=64).astype(np.float32).transpose(2,0,1)
                    x_data[j,:,:,:] = img_n_r
                    count += 1
                except:
                    pass
            print count
            assert count > 0

            # copy data to GPU
            if gpu is not None:
                x_data = cuda.to_gpu(x_data)

            # create variable
            xs = Variable(x_data)

            # set new alpha
            alpha += alpha_delta
            alpha = min(alpha, 1.)
            alpha = max(alpha, 0.)

            time_start = time.time()

            # encoder
            zs, kl = model.encoder(xs)

            # decoder
            x_tilda = model.decoder(zs)

            # generate fake data
            zeros = xp.zeros((batchsize, zs.data.shape[1])).astype(np.float32)
            z_fake = F.gaussian(Variable(zeros), Variable(zeros))
            x_fake = model.decoder(z_fake)

            # discriminator
            l_dis, l_gan = model.discriminator(xs, x_tilda, x_fake, gpu)

            # update network
            model.cleargrads()

            loss_enc = kl + l_dis
            loss_dec = gamma * l_dis - l_gan
            loss_dis = l_gan

            model.encoder.cleargrads()
            loss_enc.backward()
            opt_enc.update()

            model.decoder.cleargrads()
            loss_dec.backward()
            opt_dec.update()

            model.discriminator.cleargrads()
            loss_dis.backward()
            opt_dis.update()

            # report training status
            time_end = time.time()
            time_delta = time_end - time_start

            status = OrderedDict()
            status['epoch'] = epoch
            status['batch'] = i
            status['prog'] = '{:.1%}'.format(float(i+1) / n_train)
            status['time'] = int(time_delta * 1000)     # time in msec
            status['alpha'] = alpha
            status['loss_enc'] = '{:.4}'.format(float(loss_enc.data))
            status['loss_dec'] = '{:.4}'.format(float(loss_dec.data))
            status['loss_dis'] = '{:.4}'.format(float(loss_dis.data))
            logger.info(_status_str(status))
        if epoch % save_every == 0 or (max_epoch is not None and epoch == max_epoch - 1):
            model.save(dest_dir, epoch)

        epoch += 1


def _status_str(status):
    lst = []
    for k, v in status.items():
        lst.append(k + ':')
        lst.append(str(v))
    return '\t'.join(lst)
