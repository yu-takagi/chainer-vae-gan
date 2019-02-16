# VAE-GAN (Variational AutoEncoder-Generative Adversarial Network) implemented with Chainer
Implemenation of [Autoencoding beyond pixels using a learned similarity metric](https://arxiv.org/abs/1512.09300v2) (known as VAE-GAN).

MIT license. Contributions welcome.

## Requirements
python 2.x, chainer 4.3.1, numpy, matplotlib
Download celeba dataset before running

## examples
   
   Train:
    
    $ python train_celeb_vaegan.py savedirectoryname --gpu 0 --data-home /your/path/to/CelebA/ --data-size all --epoch 1000 --save-every 100 --gamma 0.5
