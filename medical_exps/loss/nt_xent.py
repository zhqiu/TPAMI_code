import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class NTXentLoss(torch.nn.Module):

    def __init__(self, device, batch_size, temperature, use_cosine_similarity, alpha_weight):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.alpha_weight = alpha_weight
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def softXEnt(self, target, logits):
        """
        From the pytorch discussion Forum:
        https://discuss.pytorch.org/t/soft-cross-entropy-loss-tf-has-it-does-pytorch-have-it/69501 
        """
        logprobs = torch.nn.functional.log_softmax(logits, dim = 1)
        loss = -(target * logprobs).sum() / logits.shape[0]
        return loss

    def forward(self, zis, zjs, ids=None,
                    norm=True,
                    weights=1.0):
        temperature = self.temperature
        alpha = self.alpha_weight

        """
        Pytorch implementation of the loss  SimCRL function by googleresearch: https://github.com/google-research/simclr
        @article{chen2020simple,
                title={A Simple Framework for Contrastive Learning of Visual Representations},
                author={Chen, Ting and Kornblith, Simon and Norouzi, Mohammad and Hinton, Geoffrey},
                journal={arXiv preprint arXiv:2002.05709},
                year={2020}
                }
        @article{chen2020big,
                title={Big Self-Supervised Models are Strong Semi-Supervised Learners},
                author={Chen, Ting and Kornblith, Simon and Swersky, Kevin and Norouzi, Mohammad and Hinton, Geoffrey},
                journal={arXiv preprint arXiv:2006.10029},
                year={2020}
                }
        """

        LARGE_NUM = 1e9
        """Compute loss for model.
        Args:
        hidden: hidden vector (`Tensor`) of shape (2 * bsz, dim).
        hidden_norm: whether or not to use normalization on the hidden vector.
        temperature: a `floating` number for temperature scaling.
        tpu_context: context information for tpu.
        weights: a weighting number or vector.
        Returns:
        A loss scalar.
        The logits for contrastive prediction task.
        The labels for contrastive prediction task.
        """
        # Get (normalized) hidden1 and hidden2.
        if norm:
            zis = F.normalize(zis, p=2, dim=1)
            zjs = F.normalize(zjs, p=2, dim=1)
            
        hidden1, hidden2 = zis, zjs
        batch_size = hidden1.shape[0]

        hidden1_large = hidden1
        hidden2_large = hidden2
        labels = F.one_hot(torch.arange(start=0, end=batch_size, dtype=torch.int64), num_classes=batch_size).float()
        labels = labels.to(self.device)
        masks = F.one_hot(torch.arange(start=0, end=batch_size, dtype=torch.int64), num_classes=batch_size)
        
        """
        Different from Image-Image contrastive learning
        In the case of Image-Text contrastive learning we do not compute the similarity function between the Image-Image and Text-Text pairs  
        """
        # logits_aa = torch.matmul(hidden1, torch.transpose(hidden1_large,0, 1)) / temperature
        # logits_aa = logits_aa - masks * LARGE_NUM
        # logits_bb = torch.matmul(hidden2,  torch.transpose(hidden2_large,0, 1)) / temperature
        # logits_bb = logits_bb - masks * LARGE_NUM
        logits_ab = torch.matmul(hidden1, torch.transpose(hidden2_large,0, 1)) / temperature
        logits_ba = torch.matmul(hidden2, torch.transpose(hidden1_large,0, 1)) / temperature

        loss_a = self.softXEnt(labels, logits_ab)
        loss_b = self.softXEnt(labels, logits_ba)

        return alpha*loss_a + (1-alpha)*loss_b



class SogCLR_Loss(torch.nn.Module):

    def __init__(self, device, batch_size, temperature, use_cosine_similarity, alpha_weight, N=9000000, gamma=0.8):
        super(SogCLR_Loss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.alpha_weight = alpha_weight
        self.device = device
        self.mask_neg = (1.0 - torch.eye(batch_size)).to(device)
        
        self.gamma = gamma
        self.eps = 1e-14
        self.s_I = torch.zeros(N).cuda()
        self.s_T = torch.zeros(N).cuda()
        self.b_I = torch.zeros(N).cuda()
        self.b_T = torch.zeros(N).cuda()

    def forward(self, zis, zjs, ids, norm=True, weights=1.0):
        if norm:
            zis = F.normalize(zis, p=2, dim=1)
            zjs = F.normalize(zjs, p=2, dim=1)

        # compute the logits (similarity between each image-text pair)
        sim = torch.einsum('i d, j d -> i j', zis, zjs)
        diag_sim = torch.diagonal(sim)

        # E_I(x_i)*E_T(t) - E_I(x_i)*E_T(t_i)
        image_diffs = sim - diag_sim[:, None]
        # E_I(x)*E_T(t_i) - E_I(x_i)*E_T(t_i)
        text_diffs = sim - diag_sim[None, :]

        image_diffs_d_temps = (image_diffs / self.temperature).clone().detach_()
        text_diffs_d_temps = (text_diffs / self.temperature).clone().detach_()
        
        # update b
        old_b_I = self.b_I[ids]
        new_b_I = torch.max(image_diffs_d_temps, old_b_I[:, None].tile(1, self.batch_size))
        self.b_I[ids] = torch.max(new_b_I, dim=1)[0]

        old_b_T = self.b_T[ids]
        new_b_T = torch.max(text_diffs_d_temps, old_b_T[None, :].tile(self.batch_size, 1))
        self.b_T[ids] = torch.max(new_b_T, dim=0)[0]
        
        exp_image_diffs = torch.exp(image_diffs_d_temps - self.b_I[ids][:, None]) * self.mask_neg # -b to avoid exp operation overflow
        exp_text_diffs = torch.exp(text_diffs_d_temps - self.b_T[ids][None, :]) * self.mask_neg

        g_I = torch.sum(exp_image_diffs, dim=1, keepdim=True) / (self.batch_size-1)
        g_T = torch.sum(exp_text_diffs, dim=0, keepdim=True) / (self.batch_size-1)

        s_I = (1.0-self.gamma) * self.s_I[ids] * torch.exp(old_b_I - self.b_I[ids]) + self.gamma * g_I.squeeze()
        s_T = (1.0-self.gamma) * self.s_T[ids] * torch.exp(old_b_T - self.b_T[ids]) + self.gamma * g_T.squeeze()
        s_I = s_I.reshape(g_I.shape)
        s_T = s_T.reshape(g_T.shape)

        self.s_I[ids] = s_I.squeeze()
        self.s_T[ids] = s_T.squeeze()

        s_I = s_I.clamp(min=self.eps)
        s_T = s_T.clamp(min=self.eps)

        weights_image = exp_image_diffs / s_I
        weights_text = exp_text_diffs / s_T

        if torch.any(torch.isnan(weights_image)):
            assert 0, "weights_image has nan."
        if torch.any(torch.isnan(weights_text)):
            assert 0, "weights_text has nan."

        image_loss = torch.sum(weights_image * image_diffs, dim=1, keepdim=True) / (self.batch_size-1)
        text_loss = torch.sum(weights_text * text_diffs, dim=0, keepdim=True) / (self.batch_size-1)

        total_loss = self.alpha_weight * image_loss.mean() + (1.0-self.alpha_weight) * text_loss.mean()

        return total_loss




class iSogCLR_Loss(torch.nn.Module):

    def __init__(self, device, batch_size, alpha_weight, temperature, use_cosine_similarity, rho, feature_dim, 
                    N=9000000, gamma=0.8, beta_u=0.9, tau_init=0.07, eta=0.01):
        super(iSogCLR_Loss, self).__init__()
        self.batch_size = batch_size
        self.alpha_weight = alpha_weight
        self.device = device
        self.mask_neg = (1.0 - torch.eye(batch_size)).to(device)
        
        self.gamma = gamma
        self.eps = 1e-14
        self.s_I = torch.zeros(N).cuda()
        self.s_T = torch.zeros(N).cuda()
        self.b_I = torch.zeros(N).cuda()
        self.b_T = torch.zeros(N).cuda()

        self.rho = rho

        self.tau_init = tau_init
        self.tau_I = torch.ones(N).cuda() * self.tau_init
        self.tau_T = torch.ones(N).cuda() * self.tau_init
        self.u_I = torch.zeros(N).cuda()
        self.u_T = torch.zeros(N).cuda()
        self.grad_clip = 5.0
        self.beta_u = beta_u
        self.eta = eta
        self.tau_min = 1e-3
        self.tau_max = 1.0

    def forward(self, zis, zjs, ids, norm=True, weights=1.0):
        if norm:
            zis = F.normalize(zis, p=2, dim=1)
            zjs = F.normalize(zjs, p=2, dim=1)

        # compute the logits (similarity between each image-text pair)
        sim = torch.einsum('i d, j d -> i j', zis, zjs)
        diag_sim = torch.diagonal(sim)

        # E_I(x_i)*E_T(t) - E_I(x_i)*E_T(t_i)
        image_diffs = sim - diag_sim[:, None]
        # E_I(x)*E_T(t_i) - E_I(x_i)*E_T(t_i)
        text_diffs = sim - diag_sim[None, :]

        # learnable temperature for each image and text
        tau_image = self.tau_I[ids]
        tau_text = self.tau_T[ids]

        image_diffs_d_temps = (image_diffs / tau_image[:, None]).clone().detach_()
        text_diffs_d_temps = (text_diffs / tau_text[None, :]).clone().detach_()
        
        # update b
        old_b_I = self.b_I[ids]
        new_b_I = torch.max(image_diffs_d_temps, old_b_I[:, None].tile(1, self.batch_size))
        self.b_I[ids] = torch.max(new_b_I, dim=1)[0]

        old_b_T = self.b_T[ids]
        new_b_T = torch.max(text_diffs_d_temps, old_b_T[None, :].tile(self.batch_size, 1))
        self.b_T[ids] = torch.max(new_b_T, dim=0)[0]
        
        exp_image_diffs = torch.exp(image_diffs_d_temps - self.b_I[ids][:, None]) * self.mask_neg # -b to avoid exp operation overflow
        exp_text_diffs = torch.exp(text_diffs_d_temps - self.b_T[ids][None, :]) * self.mask_neg

        g_I = torch.sum(exp_image_diffs, dim=1, keepdim=True)
        g_T = torch.sum(exp_text_diffs, dim=0, keepdim=True)

        s_I = (1.0-self.gamma) * self.s_I[ids] * torch.exp(old_b_I - self.b_I[ids]) + self.gamma * g_I.squeeze()
        s_T = (1.0-self.gamma) * self.s_T[ids] * torch.exp(old_b_T - self.b_T[ids]) + self.gamma * g_T.squeeze()
        s_I = s_I.reshape(g_I.shape)
        s_T = s_T.reshape(g_T.shape)

        self.s_I[ids] = s_I.squeeze()
        self.s_T[ids] = s_T.squeeze()

        s_I = s_I.clamp(min=self.eps)
        s_T = s_T.clamp(min=self.eps)

        weights_image = exp_image_diffs / s_I
        weights_text = exp_text_diffs / s_T

        if torch.any(torch.isnan(weights_image)):
            assert 0, "weights_image has nan."
        if torch.any(torch.isnan(weights_text)):
            assert 0, "weights_text has nan."

        image_loss = torch.sum(weights_image * image_diffs, dim=1, keepdim=True)
        text_loss = torch.sum(weights_text * text_diffs, dim=0, keepdim=True)

        total_loss = self.alpha_weight * image_loss.mean() + (1.0-self.alpha_weight) * text_loss.mean()

        # gradient of tau for image and text
        grad_tau_image = torch.log(s_I) + self.b_I[ids][:, None] + self.rho - torch.sum(weights_image * image_diffs_d_temps, dim=1, keepdim=True) / (batch_size-1)
        grad_tau_text = torch.log(s_T) + self.b_T[ids][None, :] + self.rho - torch.sum(weights_text * text_diffs_d_temps, dim=0, keepdim=True) / (batch_size-1)
       
        self.u_I[ids] = (1.0-self.beta_u) * self.u_I[ids] + self.beta_u * grad_tau_image.squeeze().clamp_(min=-self.grad_clip, max=self.grad_clip)
        self.u_T[ids] = (1.0-self.beta_u) * self.u_T[ids] + self.beta_u * grad_tau_text.squeeze().clamp_(min=-self.grad_clip, max=self.grad_clip)

        self.tau_I[ids] = (tau_image - self.eta * self.u_I[ids]).clamp_(min=self.tau_min, max=self.tau_max)
        self.tau_T[ids] = (tau_text - self.eta * self.u_T[ids]).clamp_(min=self.tau_min, max=self.tau_max)

        avg_image_tau = tau_image.mean().item()
        avg_text_tau = tau_text.mean().item()

        return total_loss, avg_image_tau, avg_text_tau



class iSogCLR_Plus_Loss(torch.nn.Module):

    def __init__(self, device, batch_size, alpha_weight, temperature, use_cosine_similarity, rho, feature_dim, 
                    gamma_s, gamma_u, beta_s, beta_u, N=9000000, tau_init=0.07, eta=0.01):
        super(iSogCLR_Plus_Loss, self).__init__()
        self.batch_size = batch_size
        self.alpha_weight = alpha_weight
        self.device = device
        self.mask_neg = (1.0 - torch.eye(batch_size)).to(device)
        
        self.gamma_s = gamma_s
        self.gamma_u = gamma_u
        self.beta_s = beta_s
        self.beta_u = beta_u

        self.eps = 1e-14
        self.s_I = torch.zeros(N).cuda()
        self.s_T = torch.zeros(N).cuda()
        self.b_I = torch.zeros(N).cuda()
        self.b_T = torch.zeros(N).cuda()

        self.rho = rho

        self.tau_init = tau_init
        self.tau_I = torch.ones(N).cuda() * self.tau_init
        self.tau_T = torch.ones(N).cuda() * self.tau_init
        self.u_I = torch.zeros(N).cuda()
        self.u_T = torch.zeros(N).cuda()
        self.grad_clip = 5.0
        self.eta = eta

        self.tau_min = 1e-3
        self.tau_max = 1.0

    def update_ma_estimators(self, ma_update_dict):
        ids = ma_update_dict['ids']
        new_s_I, new_s_T = ma_update_dict['new_s_I'], ma_update_dict['new_s_T']
        new_u_I, new_u_T = ma_update_dict['new_u_I'], ma_update_dict['new_u_T']
        self.s_I[ids] = new_s_I.cuda()
        self.s_T[ids] = new_s_T.cuda()
        self.u_I[ids] = new_u_I.cuda()
        self.u_T[ids] = new_u_T.cuda()

    def forward(self, zis, zjs, ids, norm=True, weights=1.0,
                last_g_I=None, last_g_T=None, last_grad_tau_I=None, last_grad_tau_T=None, compute_last=False):
        if norm:
            zis = F.normalize(zis, p=2, dim=1)
            zjs = F.normalize(zjs, p=2, dim=1)

        # compute the logits (similarity between each image-text pair)
        sim = torch.einsum('i d, j d -> i j', zis, zjs)
        diag_sim = torch.diagonal(sim)

        # E_I(x_i)*E_T(t) - E_I(x_i)*E_T(t_i)
        image_diffs = sim - diag_sim[:, None]
        # E_I(x)*E_T(t_i) - E_I(x_i)*E_T(t_i)
        text_diffs = sim - diag_sim[None, :]

        # learnable temperature for each image and text
        tau_image = self.tau_I[ids]
        tau_text = self.tau_T[ids]

        image_diffs_d_temps = (image_diffs / tau_image[:, None]).clone().detach_()
        text_diffs_d_temps = (text_diffs / tau_text[None, :]).clone().detach_()
        
        # update b
        old_b_I = self.b_I[ids]
        new_b_I = torch.max(image_diffs_d_temps, old_b_I[:, None].tile(1, self.batch_size))
        self.b_I[ids] = torch.max(new_b_I, dim=1)[0]

        old_b_T = self.b_T[ids]
        new_b_T = torch.max(text_diffs_d_temps, old_b_T[None, :].tile(self.batch_size, 1))
        self.b_T[ids] = torch.max(new_b_T, dim=0)[0]
        
        exp_image_diffs = torch.exp(image_diffs_d_temps - self.b_I[ids][:, None]) * self.mask_neg # -b to avoid exp operation overflow
        exp_text_diffs = torch.exp(text_diffs_d_temps - self.b_T[ids][None, :]) * self.mask_neg

        g_I = torch.sum(exp_image_diffs, dim=1, keepdim=True)
        g_T = torch.sum(exp_text_diffs, dim=0, keepdim=True)

        if not compute_last:
            self.s_I[ids] = (1.0-self.gamma_s) * self.s_I[ids] * torch.exp(old_b_I - self.b_I[ids]) + self.gamma_s * g_I.squeeze()
            self.s_T[ids] = (1.0-self.gamma_s) * self.s_T[ids] * torch.exp(old_b_T - self.b_T[ids]) + self.gamma_s * g_T.squeeze()

            if last_g_I is not None and last_g_T is not None:
                self.s_I[ids] += self.beta_s * (g_I - last_g_I.cuda())
                self.s_T[ids] += self.beta_s * (g_T - last_g_T.cuda())

            s_I = self.s_I[ids].reshape(g_I.shape)
            s_T = self.s_T[ids].reshape(g_T.shape)

        else:
            self.s_I[ids] = (1.0-self.gamma_s) * self.s_I[ids] * torch.exp(old_b_I - self.b_I[ids]) + self.gamma_s * g_I.squeeze()
            self.s_T[ids] = (1.0-self.gamma_s) * self.s_T[ids] * torch.exp(old_b_T - self.b_T[ids]) + self.gamma_s * g_T.squeeze()

            s_I = self.s_I[ids].reshape(g_I.shape)
            s_T = self.s_T[ids].reshape(g_T.shape)

        s_I = s_I.clamp(min=self.eps)
        s_T = s_T.clamp(min=self.eps)

        weights_image = exp_image_diffs / s_I
        weights_text = exp_text_diffs / s_T

        if torch.any(torch.isnan(weights_image)):
            assert 0, "weights_image has nan."
        if torch.any(torch.isnan(weights_text)):
            assert 0, "weights_text has nan."

        image_loss = torch.sum(weights_image * image_diffs, dim=1, keepdim=True)
        text_loss = torch.sum(weights_text * text_diffs, dim=0, keepdim=True)

        total_loss = self.alpha_weight * image_loss.mean() + (1.0-self.alpha_weight) * text_loss.mean()

        # gradient of tau for image and text
        grad_tau_image = torch.log(s_I) + self.b_I[ids][:, None] + self.rho - torch.sum(weights_image * image_diffs_d_temps, dim=1, keepdim=True) / (batch_size-1)
        grad_tau_text = torch.log(s_T) + self.b_T[ids][None, :] + self.rho - torch.sum(weights_text * text_diffs_d_temps, dim=0, keepdim=True) / (batch_size-1)

        if not compute_last:
            self.u_I[ids] = (1.0-self.gamma_u) * self.u_I[ids] + self.gamma_u * grad_tau_image.squeeze().clamp_(min=-self.grad_clip, max=self.grad_clip)
            self.u_T[ids] = (1.0-self.gamma_u) * self.u_T[ids] + self.gamma_u * grad_tau_text.squeeze().clamp_(min=-self.grad_clip, max=self.grad_clip)

            if last_grad_tau_I is not None and last_grad_tau_T is not None:
                self.u_I[ids] += self.beta_u * (grad_tau_image - last_grad_tau_I.cuda())
                self.u_T[ids] += self.beta_u * (grad_tau_text - last_grad_tau_T.cuda())

        else:
            self.u_I[ids] = (1.0-self.gamma_u) * self.u_I[ids] + self.gamma_u * grad_tau_image.squeeze().clamp_(min=-self.grad_clip, max=self.grad_clip)
            self.u_T[ids] = (1.0-self.gamma_u) * self.u_T[ids] + self.gamma_u * grad_tau_text.squeeze().clamp_(min=-self.grad_clip, max=self.grad_clip)

        self.tau_I[ids] = (tau_image - self.eta * self.u_I[ids]).clamp_(min=self.tau_min, max=self.tau_max)
        self.tau_T[ids] = (tau_text - self.eta * self.u_T[ids]).clamp_(min=self.tau_min, max=self.tau_max)

        avg_image_tau = tau_image.mean().item()
        avg_text_tau = tau_text.mean().item()

        ma_update_dict = {'ids': ids,
                          'new_s_I': self.s_I[ids], 'new_s_T': self.s_T[ids],
                          'new_u_I': self.u_I[ids], 'new_u_T': self.u_T[ids]}

        return g_I, g_T, grad_tau_image, grad_tau_text, total_loss, avg_image_tau, avg_text_tau, ma_update_dict



class Group_iSogCLR_Loss(torch.nn.Module):

    def __init__(self, device, batch_size, alpha_weight, temperature, use_cosine_similarity, rho, feature_dim, 
                    taus_I, taus_T, group_info_I, group_info_T, lambada, num_groups,
                    N=9000000, gamma=0.8, eta_p=0.01):
        super(Group_iSogCLR_Loss, self).__init__()
        self.batch_size = batch_size
        self.alpha_weight = alpha_weight
        self.device = device
        self.mask_neg = (1.0 - torch.eye(batch_size)).to(device)

        self.taus_I = taus_I
        self.taus_T = taus_T
        self.group_info_I = group_info_I
        self.group_info_T = group_info_T
        self.lambada = lambada
        self.num_groups = num_groups
        
        self.gamma = gamma
        self.eps = 1e-14
        self.s_I = torch.zeros(N).cuda()
        self.s_T = torch.zeros(N).cuda()
        self.b_I = torch.zeros(N).cuda()
        self.b_T = torch.zeros(N).cuda()

        self.z_I = torch.zeros(num_groups).cuda()
        self.z_T = torch.zeros(num_groups).cuda()
        self.p_I = torch.ones(num_groups).cuda() / num_groups
        self.p_T = torch.ones(num_groups).cuda() / num_groups

        self.rho = rho
        self.grad_clip = 5.0
        self.eta_p = eta_p

    def index_to_groupid(self, index, group_info):
        group_ids = [group_info[ind.item()] for ind in index]
        group_ids_tensor = torch.tensor(group_ids, dtype=tensor.dtype).cuda()

        return group_ids_tensor

    def forward(self, zis, zjs, ids, norm=True, weights=1.0):
        if norm:
            zis = F.normalize(zis, p=2, dim=1)
            zjs = F.normalize(zjs, p=2, dim=1)

        # compute the logits (similarity between each image-text pair)
        sim = torch.einsum('i d, j d -> i j', zis, zjs)
        diag_sim = torch.diagonal(sim)

        # E_I(x_i)*E_T(t) - E_I(x_i)*E_T(t_i)
        image_diffs = sim - diag_sim[:, None]
        # E_I(x)*E_T(t_i) - E_I(x_i)*E_T(t_i)
        text_diffs = sim - diag_sim[None, :]

        # learnable temperature for each image and text
        tau_image = self.taus_I[ids]
        tau_text = self.taus_T[ids]

        image_diffs_d_temps = (image_diffs / tau_image[:, None]).clone().detach_()
        text_diffs_d_temps = (text_diffs / tau_text[None, :]).clone().detach_()
        
        # update b
        old_b_I = self.b_I[ids]
        new_b_I = torch.max(image_diffs_d_temps, old_b_I[:, None].tile(1, self.batch_size))
        self.b_I[ids] = torch.max(new_b_I, dim=1)[0]

        old_b_T = self.b_T[ids]
        new_b_T = torch.max(text_diffs_d_temps, old_b_T[None, :].tile(self.batch_size, 1))
        self.b_T[ids] = torch.max(new_b_T, dim=0)[0]
        
        exp_image_diffs = torch.exp(image_diffs_d_temps - self.b_I[ids][:, None]) * self.mask_neg # -b to avoid exp operation overflow
        exp_text_diffs = torch.exp(text_diffs_d_temps - self.b_T[ids][None, :]) * self.mask_neg

        g_I = torch.sum(exp_image_diffs, dim=1, keepdim=True)
        g_T = torch.sum(exp_text_diffs, dim=0, keepdim=True)

        s_I = (1.0-self.gamma) * self.s_I[ids] * torch.exp(old_b_I - self.b_I[ids]) + self.gamma * g_I.squeeze()
        s_T = (1.0-self.gamma) * self.s_T[ids] * torch.exp(old_b_T - self.b_T[ids]) + self.gamma * g_T.squeeze()
        s_I = s_I.reshape(g_I.shape)
        s_T = s_T.reshape(g_T.shape)

        self.s_I[ids] = s_I.squeeze()
        self.s_T[ids] = s_T.squeeze()

        s_I = s_I.clamp(min=self.eps)
        s_T = s_T.clamp(min=self.eps)

        weights_image = exp_image_diffs / s_I
        weights_text = exp_text_diffs / s_T

        if torch.any(torch.isnan(weights_image)):
            assert 0, "weights_image has nan."
        if torch.any(torch.isnan(weights_text)):
            assert 0, "weights_text has nan."

         # group weights for images and texts
        index_group_id_I = self.index_to_groupid(ids, self.group_info_I)
        index_group_id_T = self.index_to_groupid(ids, self.group_info_T)

        group_weight_I = (self.num_groups * self.p_I[index_group_id_I]).cuda().reshape(g_I.shape)
        group_weight_T = (self.num_groups * self.p_T[index_group_id_T]).cuda().reshape(g_T.shape)

        image_loss = torch.sum(weights_image * group_weight_I * image_diffs, dim=1, keepdim=True)
        text_loss = torch.sum(weights_text * group_weight_T * text_diffs, dim=0, keepdim=True)

        total_loss = self.alpha_weight * image_loss.mean() + (1.0-self.alpha_weight) * text_loss.mean()

        # update p
        F_I = self.taus_I[ids].cuda() * (torch.log(s_I) + self.b_I[ids] + self.rho)
        F_T = self.taus_T[ids].cuda() * (torch.log(s_T) + self.b_T[ids] + self.rho)

        index_group_id_mat_I = F.one_hot(index_group_id_I)
        group_counts_I = torch.sum(index_group_id_mat_I, dim=0)

        index_group_id_mat_T = F.one_hot(index_group_id_T)
        group_counts_T = torch.sum(index_group_id_mat_T, dim=0)

        grad_p_I = torch.sum(index_group_id_mat_I * F_I[None, :], dim=0) / group_counts_I
        grad_p_T = torch.sum(index_group_id_mat_T * F_T[None, :], dim=0) / group_counts_T

        self.z_I = (1-self.gamma) * self.z_I + self.gamma * grad_p_I
        self.z_T = (1-self.gamma) * self.z_T + self.gamma * grad_p_T

        grad_hp_I = -self.lambada * torch.log(self.p_I + self.eps) - self.lambada
        grad_hp_T = -self.lambada * torch.log(self.p_T + self.eps) - self.lambada

        new_p_I = self.p_I * torch.exp(2*self.eta_p * (self.z_I + grad_hp_I).clamp_(min=-self.grad_clip, max=self.grad_clip))
        new_p_T = self.p_T * torch.exp(2*self.eta_p * (self.z_T + grad_hp_T).clamp_(min=-self.grad_clip, max=self.grad_clip))

        self.p_I = new_p_I / new_p_I.sum()
        self.p_T = new_p_T / new_p_T.sum()
        
        return total_loss, self.p_I, self.p_T

