import torch
import apex
# from models.resnet_clr import ResNetSimCLR
from models.model import ModelCLR
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch.nn as nn
from loss.nt_xent import NTXentLoss, SogCLR_Loss, iSogCLR_Loss, iSogCLR_Plus_Loss, Group_iSogCLR_Loss
import numpy as np
import os
import shutil
import sys
from tqdm import tqdm
from transformers import AdamW
from transformers import AutoTokenizer
import logging
import copy

import torch.distributed as dist
from apex.parallel import DistributedDataParallel

logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys, os

apex_support = False
try:
    sys.path.append('./apex')
    from apex import amp

    apex_support = True
except:
    print("Please install apex for mixed precision training from: https://github.com/NVIDIA/apex")
    apex_support = False

torch.manual_seed(0)
import torch.distributed as dist

def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('./config.yaml', os.path.join(model_checkpoints_folder, 'config.yaml'))

class SimCLR(object):
    def __init__(self, dataset, config):
        self.config = config
        self.device = self._get_device()
        self.writer = SummaryWriter()
        self.dataset = dataset

        if config['criterion'] == 'nt_xent':
            self.criterion = NTXentLoss(self.device, config['batch_size'], **config['loss'])
        elif config['criterion'] == 'sogclr':
            self.criterion = SogCLR_Loss(self.device, config['batch_size'], **config['loss'])
        elif config['criterion'] == 'isogclr':
            self.criterion = iSogCLR_Loss(self.device, config['batch_size'], **config['loss'])
        elif config['criterion'] == 'isogclr_plus':
            self.criterion = iSogCLR_Plus_Loss(self.device, config['batch_size'], **config['loss'])
        elif config['criterion'] == 'group_isogclr':
            with open(config['loss']['taus_path'], "rb") as f:
                taus_dict = pickle.load(f)
            taus_I = taus_dict["image"]
            taus_T = taus_dict["text"]

            with open(config['loss']['group_info_path'], "rb") as f:
                group_info = pickle.load(f)
            group_info_I = group_info["image"]
            group_info_T = group_info["text"]

            self.criterion = Group_iSogCLR_Loss(self.device, config['batch_size'], **config['loss'],
                                                taus_I=taus_I, taus_T=taus_T, group_info_I=group_info_I, group_info_T=group_info_T)
        else:
            assert 0, 'criterion ' + config['criterion'] + ' is not implemented.'

        self.truncation = config['truncation']
        self.tokenizer = AutoTokenizer.from_pretrained(config['model']['bert_base_model'])#, do_lower_case=config['model_bert']['do_lower_case'])

    def _get_device(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Running on:", device)
        return device

    def train(self):
        #Dataloaders
        train_loader, valid_loader = self.dataset.get_data_loaders()

        #Model Resnet Initialize
        model = ModelCLR(**self.config["model"]).to(self.device)
        model = self._load_pre_trained_weights(model)

        if self.config['criterion'] == 'isogclr_plus':
            model_last = copy.deepcopy(model)

        optimizer = torch.optim.Adam(model.parameters(), 
                                        eval(self.config['learning_rate']), 
                                        weight_decay=eval(self.config['weight_decay']))

        #if self.config['criterion'] == 'isogclr':
        #    tempnet_optimizer = torch.optim.Adam([{"params":self.criterion.image_temp_gen.parameters()},{"params":self.criterion.text_temp_gen.parameters()}], 
        #                                            eval(self.config['tempnet_learning_rate']))

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                                T_max=self.config['epochs'], 
                                                                eta_min=0, 
                                                                last_epoch=-1)

        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)

        if apex_support and self.config['fp16_precision']:
            model, optimizer = amp.initialize(model, optimizer,
                                              opt_level='O2',
                                              keep_batchnorm_fp32=True)


        # model = nn.DataParallel(model, device_ids=[2, 3, 4, 5, 6], output_device=2)
        #Checkpoint folder

        # for name, param in model.named_parameters():
        #     print(f"Parameter: {name}, Device: {param.device}")

        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

        # save config file
        _save_config_file(model_checkpoints_folder)

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf

        print(f'Training...')

        for epoch_counter in range(self.config['epochs']):
            print(f'Epoch {epoch_counter}')
            for xis, xls, ids in tqdm(train_loader):

                optimizer.zero_grad()
                # optimizer_bert.zero_grad()

                if self.config['criterion'] == 'isogclr':
                    tempnet_optimizer.zero_grad()

                xls = self.tokenizer(list(xls), 
                                    return_tensors="pt", 
                                    padding=True, 
                                    truncation=self.truncation)


                xis = xis.to(self.device)
                xls = xls.to(self.device)
                ids = ids.to(self.device)

                zis, zls = model(xis, xls) # [N,C]

                # get the representations and the projections
                # zls = model_bert(xls)  # [N,C]
                # zls = xls
                # normalize projection feature vectors
                zis = zis.to(self.device)
                # print(zis.size())
                zls = zls.to(self.device)
                # print(zls.size())

                if self.config['criterion'] == 'isogclr':
                    loss, tau_img, tau_txt = self.criterion(zis, zls, ids)
                    self.writer.add_scalar('tau_img', tau_img.mean(), global_step=n_iter)
                    self.writer.add_scalar('tau_txt', tau_txt.mean(), global_step=n_iter)

                elif self.config['criterion'] == 'isogclr_plus':
                    zis_last, zls_last = model_last(xis, xls)
                    zis_last = zis_last.to(self.device)
                    zls_last = zls_last.to(self.device)

                    if epoch_counter == 0:
                        _, _, _, _, loss, tau_img, tau_txt, _ = self.criterion(zis, zls, ids)
                        loss.backward()
                        optimizer.step()

                    else:
                        last_g_I, last_g_T, last_grad_tau_image, last_grad_tau_text, last_loss, _, _, _ = self.criterion(zis_last, zls_last, ids, compute_last=True)
                        _, _, _, _, loss, tau_img, tau_txt, ma_dict = self.criterion(zis, zls, ids, last_g_I, last_g_T, last_grad_tau_image, last_grad_tau_text)

                        grad_last = torch.autograd.grad(last_loss, model_last.parameters())

                        model_last.load_state_dict(model.state_dict())
                        model_last.update_ma_estimators(ma_dict)

                        grad = torch.autograd.grad(loss, model.parameters())

                        # STORM
                        for param, g, g_last_round in zip(model.parameters(), grad, grad_last):
                            param.grad = g - 0.1 * g_last_round

                        optimizer.step()

                    self.writer.add_scalar('tau_img', tau_img.mean(), global_step=n_iter)
                    self.writer.add_scalar('tau_txt', tau_txt.mean(), global_step=n_iter)

                elif self.config['criterion'] == 'group_isogclr':
                    loss, p_I, p_T = self.criterion(zis, zls, ids)
                    #print("p_I:", p_I)
                    #print("p_T:", p_T)

                else:
                    loss = self.criterion(zis, zls, ids)

                # loss = self._step(model_res, model_bert, xis, xls, n_iter)

                if n_iter % self.config['log_every_n_steps'] == 0:
                    self.writer.add_scalar('train_loss', loss, global_step=n_iter)

                if not self.config['criterion'] == 'isogclr_plus':
                    if apex_support and self.config['fp16_precision']:
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()

                    optimizer.step()

                if self.config['criterion'] == 'isogclr':
                    tempnet_optimizer.step()

                # optimizer_bert.step()
                n_iter += 1
                
            # validate the model if requested
            """
            if epoch_counter % self.config['eval_every_n_epochs'] == 0:
                valid_loss = self._validate(model, valid_loader, n_iter)
                if valid_loss < best_valid_loss:
                    # save the model weights
                    best_valid_loss = valid_loss
                    torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))
                self.writer.add_scalar('validation_loss', valid_loss, global_step=valid_n_iter)
                valid_n_iter += 1
            """
            if epoch_counter % self.config['eval_every_n_epochs'] == 0:
                torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                scheduler.step()
            self.writer.add_scalar('cosine_lr_decay', scheduler.get_last_lr()[0], global_step=n_iter)

    def _load_pre_trained_weights(self, model):
        try:
            checkpoints_folder = os.path.join('./runs', self.config['fine_tune_from'], 'checkpoints')
            state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'))
            model.load_state_dict(state_dict)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        return model

    def _validate(self, model, valid_loader, n_iter):

        # validation steps
        with torch.no_grad():
            model.eval()
            # model_bert.eval()
            valid_loss = 0.0
            counter = 0
            print(f'Validation step')
            for xis, xls, ids in tqdm(valid_loader):

                xls = self.tokenizer(list(xls), return_tensors="pt", padding=True, truncation=self.truncation)

                xis = xis.to(self.device)
                xls = xls.to(self.device)

                # get the representations and the projections
                zis, zls = model(xis, xls)  # [N,C]
                print("ok")

                loss = self.criterion(zis, zls)

                valid_loss += loss.item()
                counter += 1
            valid_loss /= counter
        model.train()
        # model_bert.train()
        return valid_loss