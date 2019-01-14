"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from torch.utils.serialization import load_lua
from torch.utils.data import DataLoader
from networks import Vgg16
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torchvision import transforms
from data import ImageFilelist, ImageFolder
import torch
import os
import math
import torchvision.utils as vutils
import yaml
import numpy as np
import torch.nn.init as init

# Methods
# get_all_data_loaders      : Primary data loader interface (load trainA, trainB, ..., testA, testB, ...).
# get_data_loader_list      : List-based data loader.
# get_data_loader_folder    : Folder-based data loader.
# get_config                : Load yaml file.
# eformat                   :
# write_2images             : Save output image.
# prepare_sub_folder        : Create checkpoints and images folders for saving outputs.
# write_one_row_html        : Write one row of the html file for output images.
# write_html                : Create the html file.
# write_loss
# slerp
# get_slerp_interp
# get_model_list
# load_vgg16
# vgg_preprocess
# get_scheduler
# weights_init
# jaccard                   : Computing jaccard for two inputs.
# norm                      : Normalizes images to the interval [0, 1] for output.

def get_all_data_loaders(conf, n_datasets, samples, augmentation, trim):

    batch_size = conf['batch_size']
    num_workers = conf['num_workers']

    dataset_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']

    train_loader_list = list()
    test_loader_list = list()

    for i in range(n_datasets):

        train_loader = get_data_loader_folder(os.path.join(conf['data_root']), 'train', dataset_letters[i], batch_size, True, trim, num_workers, sample=samples[i], random_transform=augmentation[i], channels=conf['input_dim'])
        test_loader = get_data_loader_folder(os.path.join(conf['data_root']), 'test', dataset_letters[i], 1, True, trim, num_workers, sample=1.0, return_path=True, random_transform=0, channels=conf['input_dim'])

        train_loader_list.append(train_loader)
        test_loader_list.append(test_loader)

    return train_loader_list, test_loader_list

def get_data_loader_folder(input_folder, fold, dataset_letter, batch_size, shuffle, trim, num_workers=4, sample=-1, return_path=False, random_transform=False, channels=1):

    dataset = ImageFolder(input_folder, sample=sample, fold=fold, dataset_letter=dataset_letter, trim_bool=trim, return_path=return_path, random_transform=random_transform, channels=channels)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True, num_workers=num_workers)
    return loader

def get_config(config):

    with open(config, 'r') as stream:
        return yaml.load(stream)

def eformat(f, prec):

    s = "%.*e"%(prec, f)
    mantissa, exp = s.split('e')
    # Add 1 to digits as 1 is taken by sign +/-.
    return "%se%d"%(mantissa, int(exp))

def __write_images(image_outputs, display_image_num, file_name):

    image_outputs = [images.expand(-1, 3, -1, -1) for images in image_outputs] # Expand gray-scale images to 3 channels.
    image_tensor = torch.cat([images[:display_image_num] for images in image_outputs], 0)
    image_grid = vutils.make_grid(image_tensor.data, nrow=display_image_num, padding=0, normalize=True)
    vutils.save_image(image_grid, file_name, nrow=1)

def write_2images(image_outputs, display_image_num, image_directory, postfix):

    n = len(image_outputs)
    __write_images(image_outputs[0:n//2], display_image_num, '%s/gen_a2b_%s.jpg' % (image_directory, postfix))
    __write_images(image_outputs[n//2:n], display_image_num, '%s/gen_b2a_%s.jpg' % (image_directory, postfix))

def prepare_sub_folder(output_directory):

    image_directory = os.path.join(output_directory, 'images')
    if not os.path.exists(image_directory):
        print("Creating directory: {}".format(image_directory))
        os.makedirs(image_directory)

    if not os.path.exists(os.path.join(image_directory, 'originals')):
        print("Creating directory: {}".format(os.path.join(image_directory, 'originals')))
        os.makedirs(os.path.join(image_directory, 'originals'))

    if not os.path.exists(os.path.join(image_directory, 'labels')):
        print("Creating directory: {}".format(os.path.join(image_directory, 'labels')))
        os.makedirs(os.path.join(image_directory, 'labels'))

    if not os.path.exists(os.path.join(image_directory, 'predictions')):
        print("Creating directory: {}".format(os.path.join(image_directory, 'predictions')))
        os.makedirs(os.path.join(image_directory, 'predictions'))

    checkpoint_directory = os.path.join(output_directory, 'checkpoints')
    if not os.path.exists(checkpoint_directory):
        print("Creating directory: {}".format(checkpoint_directory))
        os.makedirs(checkpoint_directory)

    return checkpoint_directory, image_directory

def write_one_row_html(html_file, iterations, img_filename, all_size):

    html_file.write("<h3>iteration [%d] (%s)</h3>" % (iterations,img_filename.split('/')[-1]))
    html_file.write("""
        <p><a href="%s">
          <img src="%s" style="width:%dpx">
        </a><br>
        <p>
        """ % (img_filename, img_filename, all_size))
    return


def write_html(filename, iterations, image_save_iterations, image_directory, all_size=1536):
    html_file = open(filename, "w")
    html_file.write('''
    <!DOCTYPE html>
    <html>
    <head>
      <title>Experiment name = %s</title>
      <meta http-equiv="refresh" content="30">
    </head>
    <body>
    ''' % os.path.basename(filename))
    html_file.write("<h3>current</h3>")
    write_one_row_html(html_file, iterations, '%s/gen_a2b_train_current.jpg' % (image_directory), all_size)
    write_one_row_html(html_file, iterations, '%s/gen_b2a_train_current.jpg' % (image_directory), all_size)
    for j in range(iterations, image_save_iterations-1, -1):
        if j % image_save_iterations == 0:
            write_one_row_html(html_file, j, '%s/gen_a2b_test_%08d.jpg' % (image_directory, j), all_size)
            write_one_row_html(html_file, j, '%s/gen_b2a_test_%08d.jpg' % (image_directory, j), all_size)
            write_one_row_html(html_file, j, '%s/gen_a2b_train_%08d.jpg' % (image_directory, j), all_size)
            write_one_row_html(html_file, j, '%s/gen_b2a_train_%08d.jpg' % (image_directory, j), all_size)
    html_file.write("</body></html>")
    html_file.close()


def write_loss(iterations, trainer, train_writer):
    members = [attr for attr in dir(trainer) \
               if not callable(getattr(trainer, attr)) and not attr.startswith("__") and ('loss' in attr or 'grad' in attr or 'nwd' in attr)]
    for m in members:
        train_writer.add_scalar(m, getattr(trainer, m), iterations + 1)


def slerp(val, low, high):
    """
    original: Animating Rotation with Quaternion Curves, Ken Shoemake
    https://arxiv.org/abs/1609.04468
    Code: https://github.com/soumith/dcgan.torch/issues/14, Tom White
    """
    omega = np.arccos(np.dot(low / np.linalg.norm(low), high / np.linalg.norm(high)))
    so = np.sin(omega)
    return np.sin((1.0 - val) * omega) / so * low + np.sin(val * omega) / so * high

def get_slerp_interp(nb_latents, nb_interp, z_dim):

    """
    modified from: PyTorch inference for "Progressive Growing of GANs" with CelebA snapshot
    https://github.com/ptrblck/prog_gans_pytorch_inference
    """

    latent_interps = np.empty(shape=(0, z_dim), dtype=np.float32)
    for _ in range(nb_latents):
        low = np.random.randn(z_dim)
        high = np.random.randn(z_dim)  # low + np.random.randn(512) * 0.7.
        interp_vals = np.linspace(0, 1, num=nb_interp)
        latent_interp = np.array([slerp(v, low, high) for v in interp_vals],
                                 dtype=np.float32)
        latent_interps = np.vstack((latent_interps, latent_interp))

    return latent_interps[:, :, np.newaxis, np.newaxis]

# Get model list for resume.
def get_model_list(dirname, key):

    if os.path.exists(dirname) is False:
        return None
    gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
                  os.path.isfile(os.path.join(dirname, f)) and key in f and ".pt" in f]
    if gen_models is None:
        return None
    gen_models.sort()
    last_model_name = gen_models[-1]
    return last_model_name

def load_vgg16(model_dir):

    """ Use the model from https://github.com/abhiskk/fast-neural-style/blob/master/neural_style/utils.py """
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    if not os.path.exists(os.path.join(model_dir, 'vgg16.weight')):
        if not os.path.exists(os.path.join(model_dir, 'vgg16.t7')):
            os.system('wget https://www.dropbox.com/s/76l3rt4kyi3s8x7/vgg16.t7?dl=1 -O ' + os.path.join(model_dir, 'vgg16.t7'))
        vgglua = load_lua(os.path.join(model_dir, 'vgg16.t7'))
        vgg = Vgg16()
        for (src, dst) in zip(vgglua.parameters()[0], vgg.parameters()):
            dst.data[:] = src
        torch.save(vgg.state_dict(), os.path.join(model_dir, 'vgg16.weight'))
    vgg = Vgg16()
    vgg.load_state_dict(torch.load(os.path.join(model_dir, 'vgg16.weight')))
    return vgg

def vgg_preprocess(batch):

    tensortype = type(batch.data)
    (r, g, b) = torch.chunk(batch, 3, dim = 1)
    batch = torch.cat((b, g, r), dim = 1) # Convert RGB to BGR.
    batch = (batch + 1) * 255 * 0.5 # [-1, 1] -> [0, 255].
    mean = tensortype(batch.data.size())
    mean[:, 0, :, :] = 103.939
    mean[:, 1, :, :] = 116.779
    mean[:, 2, :, :] = 123.680
    batch = batch.sub(Variable(mean)) # Subtract mean.
    return batch

def get_scheduler(optimizer, hyperparameters, iterations=-1):

    if 'lr_policy' not in hyperparameters or hyperparameters['lr_policy'] == 'constant':
        scheduler = None # Constant scheduler.
    elif hyperparameters['lr_policy'] == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=hyperparameters['step_size'],
                                        gamma=hyperparameters['gamma'], last_epoch=iterations)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', hyperparameters['lr_policy'])
    return scheduler

# Weight initialization.
def weights_init(init_type='gaussian'):

    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            # print m.__class__.__name__
            if init_type == 'gaussian':
                init.normal(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant(m.bias.data, 0.0)

    return init_fun

# Image normalization.
def norm(arr, multichannel = False):

    arr = arr.astype(np.float)

    if not multichannel:

        mn = arr.min()
        mx = arr.max()

        arr = (arr - mn) / (mx - mn)

    else:

        tmp = np.zeros((arr.shape[1], arr.shape[2], arr.shape[0]), dtype=np.float32)
        for i in range(arr.shape[0]):

            mn = arr[i,:,:].min()
            mx = arr[i,:,:].max()
            tmp[:,:,i] = (arr[i,:,:] - mn) / (mx - mn)

        arr = tmp

    return arr


# Computing jaccard metric for two inputs.
def jaccard(input1, input2):

    input1 = input1.astype(np.bool)
    input2 = input2.astype(np.bool)

    smpInt = input1 & input2
    smpUni = input1 | input2

    cntInt = np.count_nonzero(smpInt)
    cntUni = np.count_nonzero(smpUni)

    if cntUni == 0:

        return 0.0

    else:

        return (float(cntInt) / float(cntUni))

class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=-1):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
        #print(inputs.shape)
        return self.nll_loss(F.log_softmax(inputs, dim = 1), targets)


