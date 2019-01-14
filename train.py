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
parser.add_argument('--config', type=str, default='configs/CXR_lungs', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='.', help="Outputs path.")
parser.add_argument('--resume', type=int, default=-1)
parser.add_argument('--snapshot_dir', type=str, default='.')
opts = parser.parse_args()

cudnn.benchmark = True

# Load experiment setting.
config = get_config(opts.config)
display_size = config['display_size']
config['vgg_model_path'] = opts.output_path

# Setup model and data loader.
if config['trainer'] == 'MUNIT':
    trainer = MUNIT_Trainer(config, resume_epoch=opts.resume, snapshot_dir=opts.snapshot_dir)
elif config['trainer'] == 'UNIT':
    trainer = UNIT_Trainer(config, resume_epoch=opts.resume, snapshot_dir=opts.snapshot_dir)
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

train_loader_list, test_loader_list = get_all_data_loaders(config, config['n_datasets'], samples, augmentation, config['trim'])

loader_sizes = list()

for l in train_loader_list:

    loader_sizes.append(len(l))

loader_sizes = np.asarray(loader_sizes)
n_batches = loader_sizes.min()

# Setup logger and output folders.
model_name = os.path.splitext(os.path.basename(opts.config))[0]
output_directory = os.path.join(opts.output_path + "/outputs", model_name)
checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml')) # Copy config file to output folder.

# Start training.
epochs = config['max_iter']

for ep in range(max(opts.resume, 0), epochs):

    print('Start of epoch ' + str(ep + 1) + '...')

    trainer.update_learning_rate()

    print('    Training...')
    for it, data in enumerate(zip(*train_loader_list)):

        images_list = list()
        labels_list = list()
        use_list = list()

        for i in range(config['n_datasets']):

            images = data[i][0]
            labels = data[i][1]
            use = data[i][2].to(dtype=torch.uint8)

            images_list.append(images)
            labels_list.append(labels)
            use_list.append(use)

        # Randomly selecting datasets.
        perm = np.random.choice(config['n_datasets'], 2, replace=False, p=dataset_probs)
        print('        Ep: ' + str(ep + 1) + ', it: ' + str(it + 1) + '/' + str(n_batches) + ', domain pair: ' + str(perm))

        index_1 = perm[0]
        index_2 = perm[1]

        images_1 = images_list[index_1]
        images_2 = images_list[index_2]

        labels_1 = labels_list[index_1]
        labels_2 = labels_list[index_2]

        use_1 = use_list[index_1]
        use_2 = use_list[index_2]

        images_1, images_2 = Variable(images_1.cuda()), Variable(images_2.cuda())

        # Main training code.
        if (ep + 1) <= int(0.75 * epochs):

            # If in Full Training mode.
            trainer.set_sup_trainable(True)
            trainer.set_gen_trainable(True)

            trainer.dis_update(images_1, images_2, index_1, index_2, config)
            trainer.gen_update(images_1, images_2, index_1, index_2, config)

        else:

            # If in Supervision Tuning mode.
            trainer.set_sup_trainable(True)
            trainer.set_gen_trainable(False)

        labels_1 = labels_1.to(dtype=torch.long)
        labels_1[labels_1 > 0] = 1
        labels_1 = Variable(labels_1.cuda(), requires_grad=False)

        labels_2 = labels_2.to(dtype=torch.long)
        labels_2[labels_2 > 0] = 1
        labels_2 = Variable(labels_2.cuda(), requires_grad=False)

        trainer.sup_update(images_1, images_2, labels_1, labels_2, index_1, index_2, use_1, use_2, config)

    if (ep + 1) % config['snapshot_save_iter'] == 0:

        trainer.save(checkpoint_directory, (ep + 1))

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

                np_images = images.cpu().numpy().squeeze()
                np_labels = labels.cpu().numpy().squeeze()

                io.imsave(images_path, norm(np_images, config['input_dim'] != 1))
                io.imsave(labels_path, norm(np_labels))
                io.imsave(pred_path, norm(pred))

            jaccard = np.asarray(jacc_list)

            print('        Test ' + dataset_letters[i] + ' Jaccard epoch ' + str(ep + 1) + ': ' + str(100 * jaccard.mean()) + ' +/- ' + str(100 * jaccard.std()))
