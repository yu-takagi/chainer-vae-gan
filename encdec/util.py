import numpy as np
from chainer import cuda, Variable
import chainer.optimizers as O
from collections import defaultdict
import pickle
import os
from skimage import transform
import random

def list2optimizer(lst):
    """Create chainer optimizer object from list of strings, such as ['SGD', '0.01']"""
    optim_name = lst[0]
    optim_args = map(float, lst[1:])
    optimizer = getattr(O, optim_name)(*optim_args)
    return optimizer

def load_celeba(data_home='../../dat/celeba/',data_size='test'):
    file_dir = data_home + 'img_' + data_size
    fs = os.listdir(file_dir)
    dataset = []
    for fn in fs:
        try:
            f = open('%s/%s'%(file_dir,fn), 'rb')
            img_bin = f.read()
            dataset.append(img_bin)
            f.close()
        except:
            f.close()
    print len(dataset)
    return dataset

def get_deterministic_idxs(size):
    import fractions
    prime = 10007
    assert fractions.gcd(size, prime) == 1
    return np.arange(size) * prime % size

def img_resize(img, bbox, rescale_size):
    img = img[bbox[0]:bbox[1], bbox[2]:bbox[3]]
    img = transform.resize(img, (rescale_size, rescale_size, 3), order=3)
    return img

def noise_filter(img):
    # Apply some noise filters to the given image.
    # # The current filters we have implemented are:

    # def gaussian_blur(img):
    #     if np.random.randint(2) == 0:
    #         return img
    #     else:
    #         ksize = random.randrange(1, 3 + 1, 2)
    #         dst_image = cv2.GaussianBlur(src=img, ksize=(ksize, ksize), sigmaX=0)
    #         return dst_image

    def brightness(img):
        diff = random.uniform(-0.1, 0.1)
        dst_image = np.clip(img + (255.0 * diff), 0, 255).astype(np.uint8)
        return dst_image

    def contrast(img):
        diff = random.uniform(-0.1, 0.1)
        dst_image = img.astype(np.float64)
        dst_image -= 127.5
        dst_image *= (1 + diff)
        dst_image += 127.5
        dst_image = np.clip(dst_image, 0, 255).astype(np.uint8)
        return dst_image

    def white_noise(img):
        sigma = random.uniform(0.0, 0.02)
        dst_image = np.clip(
            img + (255.0 * sigma) * np.random.randn(*img.shape), 0, 255)
        dst_image = dst_image.astype(np.uint8)
        return dst_image

    def rotate(imgs):
        k = np.random.randint(4)
        return np.rot90(img, k)

    def mirror(img):
        flag = np.random.randint(2)
        if flag == 0:
            return img
        else:
            return img[:,::-1]

    # imgs = [white_noise(contrast(brightness(gaussian_blur(x)))) for x in xs]
    img = white_noise(contrast(brightness(img)))
    return mirror(img)