"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from networks import AdaINGen, MsImageDis, VAEGen, UNet
from utils import weights_init, get_model_list, vgg_preprocess, load_vgg16, get_scheduler, CrossEntropyLoss2d, norm, jaccard
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as functional
import os

import numpy as np

class MUNIT_Trainer(nn.Module):

    def __init__(self, hyperparameters, resume_epoch=-1, snapshot_dir=None):

        super(MUNIT_Trainer, self).__init__()

        lr = hyperparameters['lr']

        # Initiate the networks.
        self.gen = AdaINGen(hyperparameters['input_dim'] + hyperparameters['n_datasets'], hyperparameters['gen'], hyperparameters['n_datasets'])  # Auto-encoder for domain a.
        self.dis = MsImageDis(hyperparameters['input_dim'] + hyperparameters['n_datasets'], hyperparameters['dis'])  # Discriminator for domain a.

        self.instancenorm = nn.InstanceNorm2d(512, affine=False)
        self.style_dim = hyperparameters['gen']['style_dim']
        self.beta1 = hyperparameters['beta1']
        self.beta2 = hyperparameters['beta2']
        self.weight_decay = hyperparameters['weight_decay']

        # Initiating and loader pretrained UNet.
        self.sup = UNet(input_channels=hyperparameters['input_dim'], num_classes=2).cuda()

        # Fix the noise used in sampling.
        self.s_a = torch.randn(8, self.style_dim, 1, 1).cuda()
        self.s_b = torch.randn(8, self.style_dim, 1, 1).cuda()

        # Setup the optimizers.
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']

        dis_params = list(self.dis.parameters())
        gen_params = list(self.gen.parameters()) + list(self.sup.parameters())

        self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                                        lr=lr, betas=(self.beta1, self.beta2), weight_decay=hyperparameters['weight_decay'])
        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                        lr=lr, betas=(self.beta1, self.beta2), weight_decay=hyperparameters['weight_decay'])

        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)

        # Network weight initialization.
        self.apply(weights_init(hyperparameters['init']))
        self.dis.apply(weights_init('gaussian'))

        # Presetting one hot encoding vectors.
        self.one_hot_img = torch.zeros(hyperparameters['n_datasets'], hyperparameters['batch_size'], hyperparameters['n_datasets'], 256, 256).cuda()
        self.one_hot_c = torch.zeros(hyperparameters['n_datasets'], hyperparameters['batch_size'], hyperparameters['n_datasets'], 64, 64).cuda()

        for i in range(hyperparameters['n_datasets']):
            self.one_hot_img[i, :, i, :, :].fill_(1)
            self.one_hot_c[i, :, i, :, :].fill_(1)

        if resume_epoch != -1:

            self.resume(snapshot_dir, hyperparameters)

    def recon_criterion(self, input, target):

        return torch.mean(torch.abs(input - target))

    def semi_criterion(self, input, target):

        loss = CrossEntropyLoss2d(size_average=False).cuda()
        return loss(input, target)

    def forward(self, x_a, x_b):

        self.eval()

        x_a.volatile = True
        x_b.volatile = True

        s_a = Variable(self.s_a, volatile=True)
        s_b = Variable(self.s_b, volatile=True)

        one_hot_x_a = torch.cat([x_a, self.one_hot_img[d_index_a]], 1)
        one_hot_x_b = torch.cat([x_b, self.one_hot_img[d_index_b]], 1)
        c_a, s_a_fake = self.gen.encode(one_hot_x_a)
        c_b, s_b_fake = self.gen.encode(one_hot_x_b)

        one_hot_c_b = torch.cat([c_b, self.one_hot_c[d_index_a]],  1)
        one_hot_c_a = torch.cat([c_a, self.one_hot_c[d_index_b]],  1)
        x_ba = self.gen.decode(one_hot_c_b, s_a)
        x_ab = self.gen.decode(one_hot_c_a, s_b)

        self.train()

        return x_ab, x_ba

    def set_gen_trainable(self, train_bool):

        if train_bool:
            self.gen.train()
            for param in self.gen.parameters():
                param.requires_grad = True

        else:
            self.gen.eval()
            for param in self.gen.parameters():
                param.requires_grad = True

    def set_sup_trainable(self, train_bool):

        if train_bool:
            self.sup.train()
            for param in self.sup.parameters():
                param.requires_grad = True
        else:
            self.sup.eval()
            for param in self.sup.parameters():
                param.requires_grad = True

    def sup_update(self, x_a, x_b, y_a, y_b, d_index_a, d_index_b, use_a, use_b, hyperparameters):

        self.gen_opt.zero_grad()

        s_a = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_b = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())

        one_hot_x_a = torch.cat([x_a, self.one_hot_img[d_index_a]], 1)
        one_hot_x_b = torch.cat([x_b, self.one_hot_img[d_index_b]], 1)

        # Encode.
        c_a, s_a_prime = self.gen.encode(one_hot_x_a)
        c_b, s_b_prime = self.gen.encode(one_hot_x_b)

        # Decode (within domain).
        one_hot_c_a = torch.cat([c_a, self.one_hot_c[d_index_a]],  1)
        one_hot_c_b = torch.cat([c_b, self.one_hot_c[d_index_b]],  1)
        x_a_recon = self.gen.decode(one_hot_c_a, s_a_prime)
        x_b_recon = self.gen.decode(one_hot_c_b, s_b_prime)

        # Decode (cross domain).
        one_hot_c_ab = torch.cat([c_a, self.one_hot_c[d_index_b]],  1)
        one_hot_c_ba = torch.cat([c_b, self.one_hot_c[d_index_a]],  1)
        x_ba = self.gen.decode(one_hot_c_ba, s_a)
        x_ab = self.gen.decode(one_hot_c_ab, s_b)

        # Encode again.
        one_hot_x_ba = torch.cat([x_ba, self.one_hot_img[d_index_a]], 1)
        one_hot_x_ab = torch.cat([x_ab, self.one_hot_img[d_index_b]], 1)
        c_b_recon, s_a_recon = self.gen.encode(one_hot_x_ba)
        c_a_recon, s_b_recon = self.gen.encode(one_hot_x_ab)

        # Forwarding through supervised model.
        p_a = None
        p_b = None
        loss_semi_a = None
        loss_semi_b = None

        has_a_label = (c_a[use_a, :, :, :].size(0) != 0)
        if has_a_label:
            p_a = self.sup(c_a, use_a, True)
            p_a_recon = self.sup(c_a_recon, use_a, True)
            loss_semi_a = self.semi_criterion(p_a, y_a[use_a, :, :]) + \
                          self.semi_criterion(p_a_recon, y_a[use_a, :, :])

        has_b_label = (c_b[use_b, :, :, :].size(0) != 0)
        if has_b_label:
            p_b = self.sup(c_b, use_b, True)
            p_b_recon = self.sup(c_b, use_b, True)
            loss_semi_b = self.semi_criterion(p_b, y_b[use_b, :, :]) + \
                          self.semi_criterion(p_b_recon, y_b[use_b, :, :])

        self.loss_gen_total = None
        if loss_semi_a is not None and loss_semi_b is not None:
            self.loss_gen_total = loss_semi_a + loss_semi_b
        elif loss_semi_a is not None:
            self.loss_gen_total = loss_semi_a
        elif loss_semi_b is not None:
            self.loss_gen_total = loss_semi_b

        if self.loss_gen_total is not None:
            self.loss_gen_total.backward()
            self.gen_opt.step()

    def sup_forward(self, x, y, d_index, hyperparameters):

        self.sup.eval()

        # Encoding content image.
        one_hot_x = torch.cat([x, self.one_hot_img[d_index, 0].unsqueeze(0)], 1)
        content, _ = self.gen.encode(one_hot_x)

        # Forwarding on supervised model.
        y_pred = self.sup(content, only_prediction=True)

        # Computing metrics.
        pred = y_pred.data.max(1)[1].squeeze_(1).squeeze_(0).cpu().numpy()

        jacc = jaccard(pred, y.cpu().squeeze(0).numpy())

        return jacc, pred, content

    def gen_update(self, x_a, x_b, d_index_a, d_index_b, hyperparameters):

        self.gen_opt.zero_grad()

        s_a = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_b = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())

        # Encode.
        one_hot_x_a = torch.cat([x_a, self.one_hot_img[d_index_a]], 1)
        one_hot_x_b = torch.cat([x_b, self.one_hot_img[d_index_b]], 1)

        c_a, s_a_prime = self.gen.encode(one_hot_x_a)
        c_b, s_b_prime = self.gen.encode(one_hot_x_b)

        # Decode (within domain).
        one_hot_c_a = torch.cat([c_a, self.one_hot_c[d_index_a]],  1)
        one_hot_c_b = torch.cat([c_b, self.one_hot_c[d_index_b]],  1)
        x_a_recon = self.gen.decode(one_hot_c_a, s_a_prime)
        x_b_recon = self.gen.decode(one_hot_c_b, s_b_prime)

        # Decode (cross domain).
        one_hot_c_ab = torch.cat([c_a, self.one_hot_c[d_index_b]],  1)
        one_hot_c_ba = torch.cat([c_b, self.one_hot_c[d_index_a]],  1)
        x_ba = self.gen.decode(one_hot_c_ba, s_a)
        x_ab = self.gen.decode(one_hot_c_ab, s_b)

        # Encode again.
        one_hot_x_ba = torch.cat([x_ba, self.one_hot_img[d_index_a]], 1)
        one_hot_x_ab = torch.cat([x_ab, self.one_hot_img[d_index_b]], 1)
        c_b_recon, s_a_recon = self.gen.encode(one_hot_x_ba)
        c_a_recon, s_b_recon = self.gen.encode(one_hot_x_ab)

        # Decode again (if needed).
        one_hot_c_aba_recon = torch.cat([c_a_recon, self.one_hot_c[d_index_a]],  1)
        one_hot_c_bab_recon = torch.cat([c_b_recon, self.one_hot_c[d_index_b]],  1)
        x_aba = self.gen.decode(one_hot_c_aba_recon, s_a_prime)
        x_bab = self.gen.decode(one_hot_c_bab_recon, s_b_prime)

        # Reconstruction loss.
        self.loss_gen_recon_x_a = self.recon_criterion(x_a_recon, x_a)
        self.loss_gen_recon_x_b = self.recon_criterion(x_b_recon, x_b)
        self.loss_gen_recon_s_a = self.recon_criterion(s_a_recon, s_a)
        self.loss_gen_recon_s_b = self.recon_criterion(s_b_recon, s_b)
        self.loss_gen_recon_c_a = self.recon_criterion(c_a_recon, c_a)
        self.loss_gen_recon_c_b = self.recon_criterion(c_b_recon, c_b)

        self.loss_gen_cycrecon_x_a = self.recon_criterion(x_aba, x_a)
        self.loss_gen_cycrecon_x_b = self.recon_criterion(x_bab, x_b)

        # GAN loss.
        self.loss_gen_adv_a = self.dis.calc_gen_loss(one_hot_x_ba)
        self.loss_gen_adv_b = self.dis.calc_gen_loss(one_hot_x_ab)

        # Total loss.
        self.loss_gen_total = hyperparameters['gan_w'] * self.loss_gen_adv_a + \
                              hyperparameters['gan_w'] * self.loss_gen_adv_b + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_a + \
                              hyperparameters['recon_s_w'] * self.loss_gen_recon_s_a + \
                              hyperparameters['recon_c_w'] * self.loss_gen_recon_c_a + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_b + \
                              hyperparameters['recon_s_w'] * self.loss_gen_recon_s_b + \
                              hyperparameters['recon_c_w'] * self.loss_gen_recon_c_b + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_a + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_b

        self.loss_gen_total.backward()
        self.gen_opt.step()

    def compute_vgg_loss(self, vgg, img, target):

        img_vgg = vgg_preprocess(img)
        target_vgg = vgg_preprocess(target)
        img_fea = vgg(img_vgg)
        target_fea = vgg(target_vgg)
        return torch.mean((self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2)

    def sample(self, x_a, x_b):

        self.eval()
        x_a.volatile = True
        x_b.volatile = True
        s_a1 = Variable(self.s_a, volatile=True)
        s_b1 = Variable(self.s_b, volatile=True)
        s_a2 = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda(), volatile=True)
        s_b2 = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda(), volatile=True)
        x_a_recon, x_b_recon, x_ba1, x_ba2, x_ab1, x_ab2 = [], [], [], [], [], []
        for i in range(x_a.size(0)):

            one_hot_x_a = torch.cat([x_a[i].unsqueeze(0), self.one_hot_img_a[i].unsqueeze(0)], 1)
            one_hot_x_b = torch.cat([x_b[i].unsqueeze(0), self.one_hot_img_b[i].unsqueeze(0)], 1)

            c_a, s_a_fake = self.gen.encode(one_hot_x_a)
            c_b, s_b_fake = self.gen.encode(one_hot_x_b)
            x_a_recon.append(self.gen.decode(c_a, s_a_fake))
            x_b_recon.append(self.gen.decode(c_b, s_b_fake))
            x_ba1.append(self.gen.decode(c_b, s_a1[i].unsqueeze(0)))
            x_ba2.append(self.gen.decode(c_b, s_a2[i].unsqueeze(0)))
            x_ab1.append(self.gen.decode(c_a, s_b1[i].unsqueeze(0)))
            x_ab2.append(self.gen.decode(c_a, s_b2[i].unsqueeze(0)))

        x_a_recon, x_b_recon = torch.cat(x_a_recon), torch.cat(x_b_recon)
        x_ba1, x_ba2 = torch.cat(x_ba1), torch.cat(x_ba2)
        x_ab1, x_ab2 = torch.cat(x_ab1), torch.cat(x_ab2)
        self.train()
        return x_a, x_a_recon, x_ab1, x_ab2, x_b, x_b_recon, x_ba1, x_ba2

    def dis_update(self, x_a, x_b, d_index_a, d_index_b, hyperparameters):

        self.dis_opt.zero_grad()
        s_a = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_b = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())

        # Encode.
        one_hot_x_a = torch.cat([x_a, self.one_hot_img[d_index_a]], 1)
        one_hot_x_b = torch.cat([x_b, self.one_hot_img[d_index_b]], 1)

        c_a, _ = self.gen.encode(one_hot_x_a)
        c_b, _ = self.gen.encode(one_hot_x_b)

        one_hot_c_ba = torch.cat([c_b, self.one_hot_c[d_index_a]], 1)
        one_hot_c_ab = torch.cat([c_a, self.one_hot_c[d_index_b]], 1)

        # Decode (cross domain).
        x_ba = self.gen.decode(one_hot_c_ba, s_a)
        x_ab = self.gen.decode(one_hot_c_ab, s_b)

        # D loss.
        one_hot_x_ba = torch.cat([x_ba, self.one_hot_img[d_index_a]], 1)
        one_hot_x_ab = torch.cat([x_ab, self.one_hot_img[d_index_b]], 1)

        self.loss_dis_a = self.dis.calc_dis_loss(one_hot_x_ba, one_hot_x_a)
        self.loss_dis_b = self.dis.calc_dis_loss(one_hot_x_ab, one_hot_x_b)
        self.loss_dis_total = hyperparameters['gan_w'] * self.loss_dis_a + \
                              hyperparameters['gan_w'] * self.loss_dis_b
        self.loss_dis_total.backward()
        self.dis_opt.step()

    def update_learning_rate(self):

        if self.dis_scheduler is not None:
            self.dis_scheduler.step()
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()

    def resume(self, checkpoint_dir, hyperparameters):

        print("--> " + checkpoint_dir)

        # Load generator.
        last_model_name = get_model_list(checkpoint_dir, "gen")
        state_dict = torch.load(last_model_name)
        self.gen.load_state_dict(state_dict)
        epochs = int(last_model_name[-11:-3])

        # Load supervised model.
        last_model_name = get_model_list(checkpoint_dir, "sup")
        state_dict = torch.load(last_model_name)
        self.sup.load_state_dict(state_dict)

        # Load discriminator.
        last_model_name = get_model_list(checkpoint_dir, "dis")
        state_dict = torch.load(last_model_name)
        self.dis.load_state_dict(state_dict)

        # Load optimizers.
        last_model_name = get_model_list(checkpoint_dir, "opt")
        state_dict = torch.load(last_model_name)
        self.dis_opt.load_state_dict(state_dict['dis'])
        self.gen_opt.load_state_dict(state_dict['gen'])

        for state in self.dis_opt.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

        for state in self.gen_opt.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

        # Reinitilize schedulers.
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters, epochs)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters, epochs)

        print('Resume from epoch %d' % epochs)
        return epochs

    def save(self, snapshot_dir, epoch):

        # Save generators, discriminators, and optimizers.
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % epoch)
        dis_name = os.path.join(snapshot_dir, 'dis_%08d.pt' % epoch)
        sup_name = os.path.join(snapshot_dir, 'sup_%08d.pt' % epoch)
        opt_name = os.path.join(snapshot_dir, 'opt_%08d.pt' % epoch)

        torch.save(self.gen.state_dict(), gen_name)
        torch.save(self.dis.state_dict(), dis_name)
        torch.save(self.sup.state_dict(), sup_name)
        torch.save({'gen': self.gen_opt.state_dict(), 'dis': self.dis_opt.state_dict()}, opt_name)


class UNIT_Trainer(nn.Module):

    def __init__(self, hyperparameters, resume_epoch=-1, snapshot_dir=None):

        super(UNIT_Trainer, self).__init__()

        lr = hyperparameters['lr']

        # Initiate the networks.
        self.gen = VAEGen(hyperparameters['input_dim'] + hyperparameters['n_datasets'], hyperparameters['gen'], hyperparameters['n_datasets'])  # Auto-encoder for domain a.
        self.dis = MsImageDis(hyperparameters['input_dim'] + hyperparameters['n_datasets'], hyperparameters['dis'])  # Discriminator for domain a.

        self.instancenorm = nn.InstanceNorm2d(512, affine=False)

        self.sup = UNet(input_channels=hyperparameters['input_dim'], num_classes=2).cuda()

        # Setup the optimizers.
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']
        dis_params = list(self.dis.parameters())
        gen_params = list(self.gen.parameters()) + list(self.sup.parameters())
        self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)

        # Network weight initialization.
        self.apply(weights_init(hyperparameters['init']))
        self.dis.apply(weights_init('gaussian'))

        # Presetting one hot encoding vectors.
        self.one_hot_img = torch.zeros(hyperparameters['n_datasets'], hyperparameters['batch_size'], hyperparameters['n_datasets'], 256, 256).cuda()
        self.one_hot_h = torch.zeros(hyperparameters['n_datasets'], hyperparameters['batch_size'], hyperparameters['n_datasets'], 64, 64).cuda()

        for i in range(hyperparameters['n_datasets']):
            self.one_hot_img[i, :, i, :, :].fill_(1)
            self.one_hot_h[i, :, i, :, :].fill_(1)

        if resume_epoch != -1:

            self.resume(snapshot_dir, hyperparameters)

    def recon_criterion(self, input, target):

        return torch.mean(torch.abs(input - target))

    def semi_criterion(self, input, target):

        loss = CrossEntropyLoss2d(size_average=False).cuda()
        return loss(input, target)

    def forward(self, x_a, x_b):

        self.eval()
        x_a.volatile = True
        x_b.volatile = True
        h_a, _ = self.gen_a.encode(x_a)
        h_b, _ = self.gen_b.encode(x_b)
        x_ba = self.gen_a.decode(h_b)
        x_ab = self.gen_b.decode(h_a)
        self.train()
        return x_ab, x_ba

    def __compute_kl(self, mu):

        # def _compute_kl(self, mu, sd):
        # mu_2 = torch.pow(mu, 2)
        # sd_2 = torch.pow(sd, 2)
        # encoding_loss = (mu_2 + sd_2 - torch.log(sd_2)).sum() / mu_2.size(0)
        # return encoding_loss

        mu_2 = torch.pow(mu, 2)
        encoding_loss = torch.mean(mu_2)
        return encoding_loss

    def set_gen_trainable(self, train_bool):

        if train_bool:
            self.gen.train()
            for param in self.gen.parameters():
                param.requires_grad = True

        else:
            self.gen.eval()
            for param in self.gen.parameters():
                param.requires_grad = True

    def set_sup_trainable(self, train_bool):

        if train_bool:
            self.sup.train()
            for param in self.sup.parameters():
                param.requires_grad = True
        else:
            self.sup.eval()
            for param in self.sup.parameters():
                param.requires_grad = True

    def sup_update(self, x_a, x_b, y_a, y_b, d_index_a, d_index_b, use_a, use_b, hyperparameters):

        self.gen_opt.zero_grad()

        # Encode.
        one_hot_x_a = torch.cat([x_a, self.one_hot_img[d_index_a]], 1)
        one_hot_x_b = torch.cat([x_b, self.one_hot_img[d_index_b]], 1)
        h_a, n_a = self.gen.encode(one_hot_x_a)
        h_b, n_b = self.gen.encode(one_hot_x_b)

        # Decode (within domain).
        one_hot_h_a = torch.cat([h_a + n_a, self.one_hot_h[d_index_a]], 1)
        one_hot_h_b = torch.cat([h_b + n_b, self.one_hot_h[d_index_b]], 1)
        x_a_recon = self.gen.decode(one_hot_h_a)
        x_b_recon = self.gen.decode(one_hot_h_b)

        # Decode (cross domain).
        one_hot_h_ab = torch.cat([h_a + n_a, self.one_hot_h[d_index_b]], 1)
        one_hot_h_ba = torch.cat([h_b + n_b, self.one_hot_h[d_index_a]], 1)
        x_ba = self.gen.decode(one_hot_h_ba)
        x_ab = self.gen.decode(one_hot_h_ab)

        # Encode again.
        one_hot_x_ba = torch.cat([x_ba, self.one_hot_img[d_index_a]], 1)
        one_hot_x_ab = torch.cat([x_ab, self.one_hot_img[d_index_b]], 1)
        h_b_recon, n_b_recon = self.gen.encode(one_hot_x_ba)
        h_a_recon, n_a_recon = self.gen.encode(one_hot_x_ab)

        # Decode again (if needed).
        one_hot_h_a_recon = torch.cat([h_a_recon + n_a_recon, self.one_hot_h[d_index_a]], 1)
        one_hot_h_b_recon = torch.cat([h_b_recon + n_b_recon, self.one_hot_h[d_index_b]], 1)
        x_aba = self.gen.decode(one_hot_h_a_recon) if hyperparameters['recon_x_cyc_w'] > 0 else None
        x_bab = self.gen.decode(one_hot_h_b_recon) if hyperparameters['recon_x_cyc_w'] > 0 else None

        # Forwarding through supervised model.
        p_a = None
        p_b = None
        loss_semi_a = None
        loss_semi_b = None

        has_a_label = (h_a[use_a, :, :, :].size(0) != 0)
        if has_a_label:
            p_a = self.sup(h_a, use_a, True)
            p_a_recon = self.sup(h_a_recon, use_a, True)
            loss_semi_a = self.semi_criterion(p_a, y_a[use_a, :, :]) + \
                          self.semi_criterion(p_a_recon, y_a[use_a, :, :])

        has_b_label = (h_b[use_b, :, :, :].size(0) != 0)
        if has_b_label:
            p_b = self.sup(h_b, use_b, True)
            p_b_recon = self.sup(h_b, use_b, True)
            loss_semi_b = self.semi_criterion(p_b, y_b[use_b, :, :]) + \
                          self.semi_criterion(p_b_recon, y_b[use_b, :, :])

        self.loss_gen_total = None
        if loss_semi_a is not None and loss_semi_b is not None:
            self.loss_gen_total = loss_semi_a + loss_semi_b
        elif loss_semi_a is not None:
            self.loss_gen_total = loss_semi_a
        elif loss_semi_b is not None:
            self.loss_gen_total = loss_semi_b

        if self.loss_gen_total is not None:
            self.loss_gen_total.backward()
            self.gen_opt.step()

    def sup_forward(self, x, y, d_index, hyperparameters):

        self.sup.eval()

        # Encoding content image.
        one_hot_x = torch.cat([x, self.one_hot_img[d_index, 0].unsqueeze(0)], 1)
        hidden, _ = self.gen.encode(one_hot_x)

        # Forwarding on supervised model.
        y_pred = self.sup(hidden, only_prediction=True)

        # Computing metrics.
        pred = y_pred.data.max(1)[1].squeeze_(1).squeeze_(0).cpu().numpy()

        jacc = jaccard(pred, y.cpu().squeeze(0).numpy())

        return jacc, pred, hidden

    def gen_update(self, x_a, x_b, d_index_a, d_index_b, hyperparameters):

        self.gen_opt.zero_grad()

        # Encode.
        one_hot_x_a = torch.cat([x_a, self.one_hot_img[d_index_a]], 1)
        one_hot_x_b = torch.cat([x_b, self.one_hot_img[d_index_b]], 1)
        h_a, n_a = self.gen.encode(one_hot_x_a)
        h_b, n_b = self.gen.encode(one_hot_x_b)

        # Decode (within domain).
        one_hot_h_a = torch.cat([h_a + n_a, self.one_hot_h[d_index_a]], 1)
        one_hot_h_b = torch.cat([h_b + n_b, self.one_hot_h[d_index_b]], 1)
        x_a_recon = self.gen.decode(one_hot_h_a)
        x_b_recon = self.gen.decode(one_hot_h_b)

        # Decode (cross domain).
        one_hot_h_ab = torch.cat([h_a + n_a, self.one_hot_h[d_index_b]], 1)
        one_hot_h_ba = torch.cat([h_b + n_b, self.one_hot_h[d_index_a]], 1)
        x_ba = self.gen.decode(one_hot_h_ba)
        x_ab = self.gen.decode(one_hot_h_ab)

        # Encode again.
        one_hot_x_ba = torch.cat([x_ba, self.one_hot_img[d_index_a]], 1)
        one_hot_x_ab = torch.cat([x_ab, self.one_hot_img[d_index_b]], 1)
        h_b_recon, n_b_recon = self.gen.encode(one_hot_x_ba)
        h_a_recon, n_a_recon = self.gen.encode(one_hot_x_ab)

        # Decode again (if needed).
        one_hot_h_a_recon = torch.cat([h_a_recon + n_a_recon, self.one_hot_h[d_index_a]], 1)
        one_hot_h_b_recon = torch.cat([h_b_recon + n_b_recon, self.one_hot_h[d_index_b]], 1)
        x_aba = self.gen.decode(one_hot_h_a_recon) if hyperparameters['recon_x_cyc_w'] > 0 else None
        x_bab = self.gen.decode(one_hot_h_b_recon) if hyperparameters['recon_x_cyc_w'] > 0 else None

        # Reconstruction loss.
        self.loss_gen_recon_x_a = self.recon_criterion(x_a_recon, x_a)
        self.loss_gen_recon_x_b = self.recon_criterion(x_b_recon, x_b)
        self.loss_gen_recon_kl_a = self.__compute_kl(h_a)
        self.loss_gen_recon_kl_b = self.__compute_kl(h_b)
        self.loss_gen_cyc_x_a = self.recon_criterion(x_aba, x_a)
        self.loss_gen_cyc_x_b = self.recon_criterion(x_bab, x_b)
        self.loss_gen_recon_kl_cyc_aba = self.__compute_kl(h_a_recon)
        self.loss_gen_recon_kl_cyc_bab = self.__compute_kl(h_b_recon)

        # GAN loss.
        self.loss_gen_adv_a = self.dis.calc_gen_loss(one_hot_x_ba)
        self.loss_gen_adv_b = self.dis.calc_gen_loss(one_hot_x_ab)

        # Total loss.
        self.loss_gen_total = hyperparameters['gan_w'] * self.loss_gen_adv_a + \
                              hyperparameters['gan_w'] * self.loss_gen_adv_b + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_a + \
                              hyperparameters['recon_kl_w'] * self.loss_gen_recon_kl_a + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_b + \
                              hyperparameters['recon_kl_w'] * self.loss_gen_recon_kl_b + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cyc_x_a + \
                              hyperparameters['recon_kl_cyc_w'] * self.loss_gen_recon_kl_cyc_aba + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cyc_x_b + \
                              hyperparameters['recon_kl_cyc_w'] * self.loss_gen_recon_kl_cyc_bab

        self.loss_gen_total.backward()
        self.gen_opt.step()

    def sample(self, x_a, x_b):

        self.eval()
        x_a.volatile = True
        x_b.volatile = True
        x_a_recon, x_b_recon, x_ba, x_ab = [], [], [], []
        for i in range(x_a.size(0)):
            h_a, _ = self.gen_a.encode(x_a[i].unsqueeze(0))
            h_b, _ = self.gen_b.encode(x_b[i].unsqueeze(0))
            x_a_recon.append(self.gen_a.decode(h_a))
            x_b_recon.append(self.gen_b.decode(h_b))
            x_ba.append(self.gen_a.decode(h_b))
            x_ab.append(self.gen_b.decode(h_a))
        x_a_recon, x_b_recon = torch.cat(x_a_recon), torch.cat(x_b_recon)
        x_ba = torch.cat(x_ba)
        x_ab = torch.cat(x_ab)
        self.train()
        return x_a, x_a_recon, x_ab, x_b, x_b_recon, x_ba

    def dis_update(self, x_a, x_b, d_index_a, d_index_b, hyperparameters):

        self.dis_opt.zero_grad()

        # Encode.
        one_hot_x_a = torch.cat([x_a, self.one_hot_img[d_index_a]], 1)
        one_hot_x_b = torch.cat([x_b, self.one_hot_img[d_index_b]], 1)
        h_a, n_a = self.gen.encode(one_hot_x_a)
        h_b, n_b = self.gen.encode(one_hot_x_b)

        # Decode (cross domain).
        one_hot_h_ab = torch.cat([h_a + n_a, self.one_hot_h[d_index_b]], 1)
        one_hot_h_ba = torch.cat([h_b + n_b, self.one_hot_h[d_index_a]], 1)
        x_ba = self.gen.decode(one_hot_h_ba)
        x_ab = self.gen.decode(one_hot_h_ab)

        # D loss.
        one_hot_x_ba = torch.cat([x_ba, self.one_hot_img[d_index_a]], 1)
        one_hot_x_ab = torch.cat([x_ab, self.one_hot_img[d_index_b]], 1)
        self.loss_dis_a = self.dis.calc_dis_loss(one_hot_x_ba.detach(), one_hot_x_a)
        self.loss_dis_b = self.dis.calc_dis_loss(one_hot_x_ab.detach(), one_hot_x_b)
        self.loss_dis_total = hyperparameters['gan_w'] * self.loss_dis_a + \
                              hyperparameters['gan_w'] * self.loss_dis_b

        self.loss_dis_total.backward()
        self.dis_opt.step()

    def update_learning_rate(self):

        if self.dis_scheduler is not None:
            self.dis_scheduler.step()
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()

    def resume(self, checkpoint_dir, hyperparameters):

        # Load generators.
        last_model_name = get_model_list(checkpoint_dir, "gen")
        state_dict = torch.load(last_model_name)
        self.gen.load_state_dict(state_dict)
        epochs = int(last_model_name[-11:-3])

        # Load discriminators.
        last_model_name = get_model_list(checkpoint_dir, "dis")
        state_dict = torch.load(last_model_name)
        self.dis.load_state_dict(state_dict)

        # Load supervised model.
        last_model_name = get_model_list(checkpoint_dir, "sup")
        state_dict = torch.load(last_model_name)
        self.sup.load_state_dict(state_dict)

        # Load optimizers.
        last_model_name = get_model_list(checkpoint_dir, "opt")
        state_dict = torch.load(last_model_name)
        self.dis_opt.load_state_dict(state_dict['dis'])
        self.gen_opt.load_state_dict(state_dict['gen'])

        for state in self.dis_opt.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

        for state in self.gen_opt.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

        # Reinitilize schedulers.
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters, epochs)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters, epochs)

        print('Resume from iteration %d' % epochs)
        return epochs

    def save(self, snapshot_dir, epoch):

        # Save generators, discriminators, and optimizers.
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % epoch)
        dis_name = os.path.join(snapshot_dir, 'dis_%08d.pt' % epoch)
        sup_name = os.path.join(snapshot_dir, 'sup_%08d.pt' % epoch)
        opt_name = os.path.join(snapshot_dir, 'opt_%08d.pt' % epoch)

        torch.save(self.gen.state_dict(), gen_name)
        torch.save(self.dis.state_dict(), dis_name)
        torch.save(self.sup.state_dict(), sup_name)
        torch.save({'dis': self.dis_opt.state_dict(), 'gen': self.gen_opt.state_dict()}, opt_name)
