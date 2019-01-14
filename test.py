"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from utils import get_all_data_loaders, prepare_sub_folder, write_html, write_loss, get_config, write_2images, norm
import argparse
from torch.autograd import Variable
from trainer import MUNIT_Trainer, UNIT_Trainer
import torch.backends.cudnn as cudnn
import torch
try:
    from itertools import izip as zip
except ImportError: # Will be 3.x series.
    pass
import os
import sys
import math
import shutil
import numpy as np

from skimage import io

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/CXR_lungs_MUNIT_1.0.yaml', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='.', help="outputs path")
parser.add_argument('--load', type=int, default=400)
parser.add_argument('--snapshot_dir', type=str, default='.')
opts = parser.parse_args()

cudnn.benchmark = True

# Load experiment setting.
config = get_config(opts.config)
display_size = config['display_size']
config['vgg_model_path'] = opts.output_path

# Setup model and data loader.
if config['trainer'] == 'MUNIT':
    trainer = MUNIT_Trainer(config, resume_epoch=opts.load, snapshot_dir=opts.snapshot_dir)
elif config['trainer'] == 'UNIT':
    trainer = UNIT_Trainer(config, resume_epoch=opts.load, snapshot_dir=opts.snapshot_dir)
else:
    sys.exit("Only support MUNIT|UNIT.")
    os.exit()

trainer.cuda()

dataset_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
samples = list()
dataset_probs = list()
augmentation = list()
for i in range(config['n_datasets']):
    samples.append(config['sample_' + dataset_letters[i]])
    dataset_probs.append(config['prob_' + dataset_letters[i]])
    augmentation.append(config['transform_' + dataset_letters[i]])

_, test_loader_list = get_all_data_loaders(config, config['n_datasets'], samples, augmentation, config['trim'])

# Setup logger and output folders.
model_name = os.path.splitext(os.path.basename(opts.config))[0]
output_directory = os.path.join(opts.output_path + "/outputs", model_name)
checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml')) # Copy config file to output folder.

# Creating isomorphic directory.
if not os.path.exists(os.path.join(image_directory, 'isomorphic')):
    os.mkdir(os.path.join(image_directory, 'isomorphic'))

# Start test.
for i in range(config['n_datasets']):

    print('    Testing ' + dataset_letters[i] + '...')

    jacc_list = list()
    for it, data in enumerate(test_loader_list[i]):

        images = data[0]
        labels = data[1]
        use = data[2]
        path = data[3]

        images = Variable(images.cuda())

        labels = labels.to(dtype=torch.long)
        labels[labels > 0] = 1
        labels = Variable(labels.cuda(), requires_grad=False)

        jacc, pred, iso = trainer.sup_forward(images, labels, 0, config)
        jacc_list.append(jacc)

        images_path = os.path.join(image_directory, 'originals', path[0])
        labels_path = os.path.join(image_directory, 'labels', path[0])
        pred_path = os.path.join(image_directory, 'predictions', path[0])
        iso_path = os.path.join(image_directory, 'isomorphic', path[0] + '.npy')

        np_images = images.cpu().numpy().squeeze()
        np_labels = labels.cpu().numpy().squeeze()
        #np_iso = iso.detach().cpu().numpy().squeeze()

        io.imsave(images_path, norm(np_images, config['input_dim'] != 1))
        io.imsave(labels_path, norm(np_labels))
        io.imsave(pred_path, norm(pred))
        #np.save(iso_path, np_iso)

    jaccard = np.asarray(jacc_list)

    print('        Test ' + dataset_letters[i] + ' Jaccard epoch ' + str(opts.load) + ': ' + str(100 * jaccard.mean()) + ' +/- ' + str(100 * jaccard.std()))
