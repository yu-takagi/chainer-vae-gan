from encdec_vaegan.nn import vaegan
import chainer.functions as F
import chainer.links as L
from chainer import Chain, ChainList, Variable, cuda
import math
import numpy as np

class Encoder(vaegan.Encoder):

    def __init__(self, z_dim):
        super(Encoder, self).__init__(
            c0 = L.Convolution2D(None, 64, 5, stride=2, pad=2, wscale=0.02*math.sqrt(4*4*3)),
            c1 = L.Convolution2D(None, 128, 5, stride=2, pad=2, wscale=0.02*math.sqrt(4*4*64)),
            c2 = L.Convolution2D(None, 256, 5, stride=2, pad=2, wscale=0.02*math.sqrt(4*4*128)),

            b_c0 = L.BatchNormalization(64),
            b_c1 = L.BatchNormalization(128),
            b_c2 = L.BatchNormalization(256),

            l_mu = L.Linear(None, z_dim),
            l_ln_var = L.Linear(None, z_dim),
            b_mu     = L.BatchNormalization(z_dim),
            b_ln_var = L.BatchNormalization(z_dim),
        )

    def __call__(self, xs, gpu=None, test=False):
        h = F.relu(self.b_c0(self.c0(xs),test=test))
        h = F.relu(self.b_c1(self.c1(h),test=test))
        h = F.relu(self.b_c2(self.c2(h),test=test))
        h = F.reshape(h,(h.data.shape[0],h.data.shape[1]*h.data.shape[2]*h.data.shape[3]))
        mu      = self.b_mu(self.l_mu(h), test=test)
        ln_var  = self.b_ln_var(self.l_ln_var(h),test=test)
        z, kl, _ = _infer_z(mu, ln_var)

        return z, kl

class Decoder(vaegan.Decoder):

    def __init__(self, z_dim):
        super(Decoder, self).__init__(
            l = L.Linear(z_dim, 8*8*256, wscale=0.02*math.sqrt(z_dim)),
            dc1 = L.Deconvolution2D(256, 256, 5, stride=2, pad=2, wscale=0.02*math.sqrt(4*4*256),outsize=(16,16)),
            dc2 = L.Deconvolution2D(256, 128, 5, stride=2, pad=2, wscale=0.02*math.sqrt(4*4*256),outsize=(32,32)),            
            dc3 = L.Deconvolution2D(128, 32, 5, stride=2, pad=2, wscale=0.02*math.sqrt(4*4*128),outsize=(64,64)),            
            dc4 = L.Convolution2D(None, 3, 5, stride=1, pad=2, wscale=0.02*math.sqrt(4*4*32)),

            b_l = L.BatchNormalization(8*8*256),
            b_c1 = L.BatchNormalization(256),
            b_c2 = L.BatchNormalization(128),
            b_c3 = L.BatchNormalization(32),
        )

    def __call__(self, z, gpu=None, test=False):
        batch_size = z.data.shape[0]
        h = F.reshape(F.relu(self.b_l(self.l(z),test=test)), (z.data.shape[0], 256, 8, 8))
        h = F.relu(self.b_c1(self.dc1(h),test=test))
        h = F.relu(self.b_c2(self.dc2(h),test=test))
        h = F.relu(self.b_c3(self.dc3(h),test=test))
        x_rec = F.sigmoid(self.dc4(h))
        return x_rec

class Discriminator(vaegan.Discriminator):
    def __init__(self, z_dim):
        super(Discriminator, self).__init__(
            c0 = L.Convolution2D(None, 32, 5, stride=1, pad=2, wscale=0.02*math.sqrt(4*4*3)),
            c1 = L.Convolution2D(None, 128, 5, stride=2, pad=2, wscale=0.02*math.sqrt(4*4*64)),
            c2 = L.Convolution2D(None, 256, 5, stride=2, pad=2, wscale=0.02*math.sqrt(4*4*128)),
            c3 = L.Convolution2D(None, 256, 5, stride=2, pad=2, wscale=0.02*math.sqrt(4*4*256)),            
            l0 = L.Linear(None, 512, wscale=0.02*math.sqrt(6*6*256)),
            l1 = L.Linear(None, 2, wscale=0.02*math.sqrt(512)),

            b_c1 = L.BatchNormalization(128),
            b_c2 = L.BatchNormalization(256),
            b_c3 = L.BatchNormalization(256),
            b_l0 = L.BatchNormalization(512),
        )

    def __call__(self, xs, x_tilda, x_fake, gpu=None, test=False):
        xp = np if gpu is None else cuda.cupy
        batch_size = xs.data.shape[0]

        # real
        h = F.relu(self.c0(xs))
        h = F.relu(self.b_c1(self.c1(h),test=test))
        h = F.relu(self.b_c2(self.c2(h),test=test))
        h_xs = F.relu(self.b_c3(self.c3(h),test=test))
        h_xs = F.reshape(h_xs,(h_xs.data.shape[0],h_xs.data.shape[1]*h_xs.data.shape[2]*h_xs.data.shape[3]))
        h = F.relu(self.b_l0(self.l0(h_xs), test=test))
        y_xs = self.l1(h)

        # tilda
        h = F.relu(self.c0(x_tilda))
        h = F.relu(self.b_c1(self.c1(h),test=test))
        h = F.relu(self.b_c2(self.c2(h),test=test))
        h_x_tilda = F.relu(self.b_c3(self.c3(h),test=test))
        h_x_tilda = F.reshape(h_x_tilda,(h_x_tilda.data.shape[0],h_x_tilda.data.shape[1]*h_x_tilda.data.shape[2]*h_x_tilda.data.shape[3]))
        h = F.relu(self.b_l0(self.l0(h_x_tilda), test=test))
        y_x_tilda = self.l1(h)

        # fake
        h = F.relu(self.c0(x_fake))
        h = F.relu(self.b_c1(self.c1(h),test=test))
        h = F.relu(self.b_c2(self.c2(h),test=test))
        h = F.relu(self.b_c3(self.c3(h),test=test))
        h = F.reshape(h,(h.data.shape[0],h.data.shape[1]*h.data.shape[2]*h.data.shape[3]))
        h = F.relu(self.b_l0(self.l0(h), test=test))
        y_fake = self.l1(h)

        # calulate similarity loss
        zeros = Variable(xp.zeros((batch_size, h_xs.data.shape[1]), dtype=np.float32))
        l_dis = F.gaussian_nll(h_xs, h_x_tilda, zeros)

        # calculate discrimation loss
        l_gan = F.softmax_cross_entropy(y_xs, Variable(xp.ones(batch_size, dtype=np.int32)))
        l_gan += F.softmax_cross_entropy(y_x_tilda, Variable(xp.zeros(batch_size, dtype=np.int32)))
        l_gan += F.softmax_cross_entropy(y_fake, Variable(xp.zeros(batch_size, dtype=np.int32)))

        print "P(y_xs)      :", F.sum(F.softmax(y_xs),axis=0).data/batch_size
        print "P(y_x_tilda) :", F.sum(F.softmax(y_x_tilda),axis=0).data/batch_size
        print "P(y_fake)    : ", F.sum(F.softmax(y_fake),axis=0).data/batch_size

        return l_dis, l_gan

class _Mlp(Chain):

    def __init__(self, in_dim, hidden_dims, active):
        super(_Mlp, self).__init__()
        self.active = active

        ds = [in_dim] + hidden_dims
        ls = ChainList()
        bns = ChainList()
        for d_in, d_out in zip(ds, ds[1:]):
            l = L.Linear(d_in, d_out)
            bn = L.BatchNormalization(d_out)
            ls.add_link(l)
            bns.add_link(bn)
        self.add_link('ls', ls)
        self.add_link('bns', bns)

    def __call__(self, x, test=False):
        h = x
        for (l,bn) in zip(self.ls,self.bns):
            h = self.active(bn(l(h),test=test))
        return h

def _infer_z(mu, ln_var):
    batch_size = mu.data.shape[0]
    var = F.exp(ln_var)
    z = F.gaussian(mu, ln_var)
    kl = -F.sum(1 + ln_var - mu ** 2 - var) / 2
    kl_all = -F.sum(1 + ln_var - mu ** 2 - var,axis=1) / 2
    kl /= batch_size
    return z, kl, kl_all
