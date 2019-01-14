"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import torch.utils.data as data
import os.path
import numpy as np
import torch

from skimage import transform
from skimage import io

try:
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError

def default_loader(path):

    return io.imread(path)

def default_flist_reader(flist):

    """
    flist format: impath label\nimpath label\n ...(same to caffe's filelist)
    """
    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            impath = line.strip()
            imlist.append(impath)

    return imlist

class ImageFilelist(data.Dataset):

    def __init__(self, root, flist, transform=None,
                 flist_reader=default_flist_reader, loader=default_loader):

        self.root = root
        self.imlist = flist_reader(flist)
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):

        impath = self.imlist[index]
        img = self.loader(os.path.join(self.root, impath))
        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):

        return len(self.imlist)

class ImageLabelFilelist(data.Dataset):

    def __init__(self, root, flist, transform=None,
                 flist_reader=default_flist_reader, loader=default_loader):

        self.root = root
        self.imlist = flist_reader(os.path.join(self.root, flist))
        self.transform = transform
        self.loader = loader
        self.classes = sorted(list(set([path.split('/')[0] for path in self.imlist])))
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        self.imgs = [(impath, self.class_to_idx[impath.split('/')[0]]) for impath in self.imlist]

    def __getitem__(self, index):

        impath, label = self.imgs[index]
        img = self.loader(os.path.join(self.root, impath))
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):

        return len(self.imgs)

###############################################################################
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code so that it also loads images from the current
# directory as well as the subdirectories
###############################################################################

import torch.utils.data as data

import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):

    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir, fold, dataset_letter):

    items = []

    img_dir = os.path.join(dir, fold + dataset_letter)
    msk_dir = os.path.join(dir, 'label_' + fold + dataset_letter)

    assert os.path.isdir(img_dir), '%s is not a valid directory' % dir
    assert os.path.isdir(msk_dir), '%s is not a valid directory' % dir

    files = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))]

    for f in files:

        img_path = os.path.join(img_dir, f)
        msk_path = os.path.join(msk_dir, f)

        items.append({'img': img_path, 'msk': msk_path, 'file': f})

    return items

def norm(img):

    img = img.astype(np.float32)

    mn = img.min()
    mx = img.max()

    return (img - mn) / (mx - mn)

class ImageFolder(data.Dataset):

    def __init__(self, root, sample, fold='train', dataset_letter='A', loader=default_loader, trim_bool=0, return_path=False, random_transform=False, channels=1):

        imgs = sorted(make_dataset(root, fold, dataset_letter))
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.loader = loader
        self.sample = sample
        self.trim_bool = trim_bool
        self.return_path = return_path
        self.random_transform = random_transform
        self.channels = channels

        np.random.seed(12345)

        perm = np.random.permutation(len(imgs))
        self.has_label = np.zeros((len(imgs)), np.int)
        self.has_label[perm[0:int(self.sample * len(imgs))]] = 1

        print('Sample limits for ' + fold + dataset_letter + ': [0:' + str(int(self.sample * len(imgs))) + ']')
        print(self.sample)

        print('Sample images...')
        for i in range(int(self.sample * len(imgs))):
            print(self.imgs[perm[i]])

    ############################################################################################################################
    # Trim function adapted from: https://codereview.stackexchange.com/questions/132914/crop-black-border-of-image-using-numpy #
    ############################################################################################################################
    def trim(self, img, msk):

        tolerance = 0.05 * float(img.max())

        # Mask of non-black pixels (assuming image has a single channel).
        bin = img > tolerance

        # Coordinates of non-black pixels.
        coords = np.argwhere(bin)

        # Bounding box of non-black pixels.
        x0, y0 = coords.min(axis=0)
        x1, y1 = coords.max(axis=0) + 1   # slices are exclusive at the top

        # Get the contents of the bounding box.
        img_crop = img[x0:x1, y0:y1]
        msk_crop = msk[x0:x1, y0:y1]

        return img_crop, msk_crop

    # Data augmentation.
    def transform(self, img, msk, negate=True, max_angle=8, low=0.1, high=0.9, shear=0.0, fliplr=False, flipud=False):

        # Random color inversion.
        if negate:
            if np.random.uniform() > 0.5:
                if self.channels == 1:
                    img = (img.max() - img)
                else:
                    for i in range(img.shape[2]):
                        img[:,:,i] = (img[:,:,i].max() - img[:,:,i])

        # Random Flipping.
        if fliplr:
            if np.random.uniform() > 0.5:
                img = np.fliplr(img)
                msk = np.fliplr(msk)

        if flipud:
            if np.random.uniform() > 0.5:
                img = np.flipud(img)
                msk = np.flipud(msk)


        # Random Rotation.
        if max_angle != 0.0:
            angle = np.random.uniform() * max_angle
            if np.random.uniform() > 0.5:
                angle = angle * -1.0

            img = transform.rotate(img, angle, resize=False)
            msk = transform.rotate(msk, angle, resize=False)

        # Random Shear.
        if shear != 0.0:
            rand_shear = np.random.uniform(low=0, high=shear)
            affine = transform.AffineTransform(shear=rand_shear)

            img = transform.warp(img, inverse_map=affine)
            msk = transform.warp(msk, inverse_map=affine)

        # Crop.
        if low != 0.0 or high != 1.0:
            beg_crop = np.random.uniform(low=0, high=low, size=2)
            end_crop = np.random.uniform(low=high, high=1.0, size=2)

            s0 = img.shape[0]
            s1 = img.shape[1]

            img = img[int(beg_crop[0] * s0):int(end_crop[0] * s0), int(beg_crop[1] * s1):int(end_crop[1] * s1)]
            msk = msk[int(beg_crop[0] * s0):int(end_crop[0] * s0), int(beg_crop[1] * s1):int(end_crop[1] * s1)]

        return img, msk

    def __getitem__(self, index):

        item = self.imgs[index]

        img_path = item['img']
        msk_path = item['msk']
        file_name = item['file']

        img = self.loader(img_path)
        msk = self.loader(msk_path)

        if self.channels == 1:
            if len(img.shape) > 2:
                img = img[:,:,0]
        if len(msk.shape) > 2:
            msk = msk[:,:,0]

        img = transform.resize(img, msk.shape, preserve_range=True)

        resize_to = (256, 256)

        use_msk = False

        if self.trim_bool != 0:
            img, msk = self.trim(img, msk)

        if self.random_transform == 3:
            img, msk = self.transform(img, msk, negate=False, max_angle=90, low=0.2, high=0.8, shear=0.05, fliplr=True, flipud=True)
        elif self.random_transform == 2:
            img, msk = self.transform(img, msk)
        elif self.random_transform == 1:
            img, msk = self.transform(img, msk, negate=False, max_angle=2, low=0.05, high=0.95)

        msk = transform.resize(msk, resize_to, preserve_range=True)
        msk[msk <= (msk.max() / 2)] = 0
        msk[msk >  (msk.max() / 2)] = 1
        msk = msk.astype(np.int)
        msk = torch.from_numpy(msk)

        if self.sample != -1:
            if self.has_label[index] != 0:
                use_msk = True
            else:
                use_msk = False
        else:
            use_msk = True

        if not use_msk:
            msk[:,:,] = 0

        img = transform.resize(img, resize_to, preserve_range=True).astype(np.float32)
        if self.channels == 1:
            img = (img - img.mean()) / (img.std() + 1e-10)
            img = np.expand_dims(img, 0)
        else:
            tmp = np.zeros((img.shape[2], img.shape[0], img.shape[1]), dtype=np.float32)
            for i in range(img.shape[2]):
                tmp[i, :, :] = (img[:, :, i] - img[:, :, i].mean()) / (img[:, :, i].std() + 1e-10)
            img = tmp

        img = torch.from_numpy(img)

        if self.return_path:
            return img, msk, use_msk, file_name
        else:
            return img, msk, use_msk

    def __len__(self):

        return len(self.imgs)
