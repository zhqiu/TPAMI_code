"""
    implementation of other two-way contrastive losses
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


# https://github.com/Spijkervet/SimCLR/blob/master/simclr/modules/gather.py
class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out


class CLIP_Loss(nn.Module):

    def __init__(self, world_size=8, temperature=0.01, personalized_tau=False, image_tau=None, text_tau=None):
        super(CLIP_Loss, self).__init__()
        self.world_size = world_size
        self.temperature = temperature
        self.personalized_tau = personalized_tau # if true, then temperatures are learnable
        self.image_tau = image_tau
        self.text_tau = text_tau

    def forward(self, image_features, text_features, image_idx=None, text_idx=None):
        if self.world_size > 1:
            image_features = torch.cat(GatherLayer.apply(image_features), dim=0)
            text_features = torch.cat(GatherLayer.apply(text_features), dim=0)

        if self.personalized_tau:
            image_temp = self.image_tau[image_idx]
            text_temp = self.text_tau[text_idx]
            sim = torch.einsum('i d, j d -> i j', text_features, image_features)
            labels = torch.arange(image_features.shape[0], device=image_features.device)
            total_loss = (F.cross_entropy(sim / text_temp, labels) + F.cross_entropy(sim.t() / image_temp, labels)) / 2

        else:
            sim = torch.einsum('i d, j d -> i j', text_features, image_features) / self.temperature
            labels = torch.arange(image_features.shape[0], device=image_features.device)
            total_loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)) / 2

        return total_loss


class SogCLR_Loss(nn.Module):
    def __init__(self, N=8000000, gamma=0.1, temperature=0.07, world_size=8):
        """
        Inputs:
           N is number of samples in training set
        """
        super(SogCLR_Loss, self).__init__()
        self.world_size = world_size
        self.s_I = torch.zeros(N).cuda()
        self.s_T = torch.zeros(N).cuda()
        self.b_I = torch.zeros(N).cuda()
        self.b_T = torch.zeros(N).cuda()
        self.gamma = gamma
        self.temperature = temperature
        self.eps = 1e-10

    def forward(self, image_features, text_features, image_ids, text_ids):
        """
        Inputs:
            image_features, text_features is l2-normalized tensor
            image_features, text_features: [batch_size, emb_dim]
        """

        if self.world_size > 1:
            image_features = torch.cat(GatherLayer.apply(image_features), dim=0)
            text_features = torch.cat(GatherLayer.apply(text_features), dim=0)

        # compute the logits (similarity between each image-text pair)
        sim = torch.einsum('i d, j d -> i j', image_features, text_features)
        diag_sim = torch.diagonal(sim)

        batch_size = sim.shape[0]

        # E_I(x_i)*E_T(t) - E_I(x_i)*E_T(t_i)
        image_diffs = sim - diag_sim[:, None]
        # E_I(x)*E_T(t_i) - E_I(x_i)*E_T(t_i)
        text_diffs = sim - diag_sim[None, :]

        image_diffs_d_temps = (image_diffs / self.temperature).clone().detach_()
        text_diffs_d_temps = (text_diffs / self.temperature).clone().detach_()

        # update b
        old_b_I = self.b_I[image_ids]
        new_b_I = torch.max(image_diffs_d_temps, old_b_I[:, None].tile(1, batch_size))
        self.b_I[image_ids] = torch.max(new_b_I, dim=1)[0]

        old_b_T = self.b_T[text_ids]
        new_b_T = torch.max(text_diffs_d_temps, old_b_T[None, :].tile(batch_size, 1))
        self.b_T[text_ids] = torch.max(new_b_T, dim=0)[0]

        exp_image_diffs = torch.exp(image_diffs_d_temps - self.b_I[image_ids][:, None]) # -b to avoid exp operation overflow
        exp_text_diffs = torch.exp(text_diffs_d_temps - self.b_T[text_ids][None, :])

        g_I = torch.sum(exp_image_diffs, dim=1, keepdim=True) / (batch_size-1)
        g_T = torch.sum(exp_text_diffs, dim=0, keepdim=True) / (batch_size-1)

        self.s_I[image_ids] = (1.0-self.gamma) * self.s_I[image_ids] * torch.exp(old_b_I - self.b_I[image_ids]) + self.gamma * g_I.squeeze()
        self.s_T[text_ids] = (1.0-self.gamma) * self.s_T[text_ids] * torch.exp(old_b_T - self.b_T[text_ids]) + self.gamma * g_T.squeeze()
        
        s_I = self.s_I[image_ids].reshape(g_I.shape)
        s_T = self.s_T[text_ids].reshape(g_T.shape)

        weights_image = exp_image_diffs / (s_I + self.eps)
        weights_text = exp_text_diffs / (s_T + self.eps)

        if torch.any(torch.isnan(weights_image)):
            assert 0, "weights_image has nan."
        if torch.any(torch.isnan(weights_text)):
            assert 0, "weights_text has nan."

        image_loss = torch.sum(weights_image * image_diffs, dim=1, keepdim=True) / (batch_size-1)
        text_loss = torch.sum(weights_text * text_diffs, dim=0, keepdim=True) / (batch_size-1)

        total_loss = image_loss.mean() + text_loss.mean()

        return total_loss



class SogCLR_DRO_Loss(nn.Module):
    def __init__(self, N=2900000, gamma=0.8, tau_init=0.07, tau_min=5e-3, tau_max=1.0, rho_init=0.1, bsz=128,
                 eta_init=0.01, eta_min=1e-4, beta_u=0.9, eta_sched=None, eta_exp_gamma=0.8, world_size=8, eps=1e-10):
        
        #Inputs:
        #   N is number of samples in training set
        
        super(SogCLR_DRO_Loss, self).__init__()
        self.world_size = world_size
        self.gamma_I, self.gamma_T = gamma, gamma
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.tau_init = tau_init
        self.s_I = torch.zeros(N).cuda()
        self.s_T = torch.zeros(N).cuda()
        self.tau_I = torch.ones(N).cuda() * self.tau_init
        self.tau_T = torch.ones(N).cuda() * self.tau_init
        self.u_I = torch.zeros(N).cuda()
        self.u_T = torch.zeros(N).cuda()
        self.b_I = torch.zeros(N).cuda()
        self.b_T = torch.zeros(N).cuda()
        self.rho_I = rho_init
        self.rho_T = rho_init
        self.eps = eps
        self.eta_sched = eta_sched
        self.eta_exp_gamma = eta_exp_gamma  # multiplicative factor of learning rate decay for exponential eta_sched
        self.eta_init = eta_init
        self.eta_min = eta_min
        self.beta_u = beta_u
        self.batch_size = bsz
        self.grad_clip = 5.0
        self.mask_neg = (1.0 - torch.eye(bsz)).cuda()

    def forward(self, image_features, text_features, image_ids, text_ids, epoch, max_epoch):
        if self.world_size > 1:
            image_features = torch.cat(GatherLayer.apply(image_features), dim=0)
            text_features = torch.cat(GatherLayer.apply(text_features), dim=0)

        # compute the logits (similarity between each image-text pair)
        sim = torch.einsum('i d, j d -> i j', image_features, text_features)
        diag_sim = torch.diagonal(sim)

        batch_size = sim.shape[0]

        # learnable temperature for each image and text
        tau_image = self.tau_I[image_ids]
        tau_text = self.tau_T[text_ids]

        # E_I(x_i)*E_T(t) - E_I(x_i)*E_T(t_i)
        image_diffs = sim - diag_sim[:, None]
        # E_I(x)*E_T(t_i) - E_I(x_i)*E_T(t_i)
        text_diffs = sim - diag_sim[None, :]

        image_diffs_d_temps = (image_diffs / tau_image[:, None]).clone().detach_()
        text_diffs_d_temps = (text_diffs / tau_text[None, :]).clone().detach_()

        # update b
        old_b_I = self.b_I[image_ids]
        new_b_I = torch.max(image_diffs_d_temps, old_b_I[:, None].tile(1, batch_size))
        self.b_I[image_ids] = torch.max(new_b_I, dim=1)[0]

        old_b_T = self.b_T[text_ids]
        new_b_T = torch.max(text_diffs_d_temps, old_b_T[None, :].tile(batch_size, 1))
        self.b_T[text_ids] = torch.max(new_b_T, dim=0)[0]

        exp_image_diffs = torch.exp(image_diffs_d_temps - self.b_I[image_ids][:, None]) # -b to avoid exp operation overflow
        exp_text_diffs = torch.exp(text_diffs_d_temps - self.b_T[text_ids][None, :])

        g_I = torch.sum(exp_image_diffs, dim=1, keepdim=True) / (batch_size-1)
        g_T = torch.sum(exp_text_diffs, dim=0, keepdim=True) / (batch_size-1)

        self.s_I[image_ids] = (1.0-self.gamma_I) * self.s_I[image_ids] * torch.exp(old_b_I - self.b_I[image_ids]) + self.gamma_I * g_I.squeeze()
        self.s_T[text_ids] = (1.0-self.gamma_T) * self.s_T[text_ids] * torch.exp(old_b_T - self.b_T[text_ids]) + self.gamma_T * g_T.squeeze()
        
        s_I = self.s_I[image_ids].reshape(g_I.shape)
        s_T = self.s_T[text_ids].reshape(g_T.shape)

        weights_image = exp_image_diffs / (s_I + self.eps)
        weights_text = exp_text_diffs / (s_T + self.eps)

        if torch.any(torch.isnan(weights_image)):
            assert 0, "weights_image has nan."
        if torch.any(torch.isnan(weights_text)):
            assert 0, "weights_text has nan."

        image_loss = torch.sum(weights_image * image_diffs, dim=1, keepdim=True) / (batch_size-1)
        text_loss = torch.sum(weights_text * text_diffs, dim=0, keepdim=True) / (batch_size-1)

        total_loss = image_loss.mean() + text_loss.mean()

        # gradient of tau for image and text
        grad_tau_image = torch.log(s_I) + self.b_I[image_ids][:, None] + self.rho_I - torch.sum(weights_image * image_diffs_d_temps, dim=1, keepdim=True) / (batch_size-1)
        grad_tau_text = torch.log(s_T) + self.b_T[text_ids][None, :] + self.rho_T - torch.sum(weights_text * text_diffs_d_temps, dim=0, keepdim=True) / (batch_size-1)
       
        self.u_I[image_ids] = (1.0-self.beta_u) * self.u_I[image_ids] + self.beta_u * grad_tau_image.squeeze().clamp_(min=-self.grad_clip, max=self.grad_clip)
        self.u_T[text_ids] = (1.0-self.beta_u) * self.u_T[text_ids] + self.beta_u * grad_tau_text.squeeze().clamp_(min=-self.grad_clip, max=self.grad_clip)

        if self.eta_sched == 'cosine':
            eta_cur = self.eta_min + (self.eta_init - self.eta_min) * math.cos(math.pi * (epoch / max_epoch) / 2)
        elif self.eta_sched == 'exp':
            eta_cur = (self.eta_init - self.eta_min) * self.eta_exp_gamma ** (epoch-1) + self.eta_min
        elif self.eta_sched == 'const':
            eta_cur = self.eta_init
        else:
            assert 0, self.eta_sched + " is not supported."

        self.tau_I[image_ids] = (tau_image - eta_cur * self.u_I[image_ids]).clamp_(min=self.tau_min, max=self.tau_max)
        self.tau_T[text_ids] = (tau_text - eta_cur * self.u_T[text_ids]).clamp_(min=self.tau_min, max=self.tau_max)

        avg_image_tau = tau_image.mean().item()
        avg_text_tau = tau_text.mean().item()

        return total_loss, avg_image_tau, avg_text_tau, eta_cur, grad_tau_image.mean().item(), grad_tau_text.mean().item(), old_b_I.mean().item(), old_b_T.mean().item()



class iSogCLR_Plus_Loss(nn.Module):
    def __init__(self, gamma_s, gamma_u, beta_s, beta_u,
                 N=2900000, tau_init=0.07, tau_min=5e-3, tau_max=1.0, rho_init=0.1, bsz=128,
                 eta_init=0.01, eta_min=1e-4, eta_sched=None, eta_exp_gamma=0.8, world_size=8, eps=1e-10):
        
        #Inputs:
        #   N is number of samples in training set
        
        super(iSogCLR_Plus_Loss, self).__init__()
        self.world_size = world_size
        self.gamma_s = gamma_s
        self.gamma_u = gamma_u
        self.beta_s = beta_s
        self.beta_u = beta_u
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.tau_init = tau_init
        self.s_I = torch.zeros(N).cuda()
        self.s_T = torch.zeros(N).cuda()
        self.tau_I = torch.ones(N).cuda() * self.tau_init
        self.tau_T = torch.ones(N).cuda() * self.tau_init
        self.u_I = torch.zeros(N).cuda()
        self.u_T = torch.zeros(N).cuda()
        self.b_I = torch.zeros(N).cuda()
        self.b_T = torch.zeros(N).cuda()
        self.rho_I = rho_init
        self.rho_T = rho_init
        self.eps = eps
        self.eta_sched = eta_sched
        self.eta_exp_gamma = eta_exp_gamma  # multiplicative factor of learning rate decay for exponential eta_sched
        self.eta_init = eta_init
        self.eta_min = eta_min
        self.batch_size = bsz
        self.grad_clip = 5.0
        self.mask_neg = (1.0 - torch.eye(bsz)).cuda()

    def update_ma_estimators(self, ma_update_dict):
        image_ids, text_ids = ma_update_dict['image_ids'], ma_update_dict['text_ids']
        new_s_I, new_s_T = ma_update_dict['new_s_I'], ma_update_dict['new_s_T']
        new_u_I, new_u_T = ma_update_dict['new_u_I'], ma_update_dict['new_u_T']
        self.s_I[image_ids] = new_s_I.cuda()
        self.s_T[text_ids] = new_s_T.cuda()
        self.u_I[image_ids] = new_u_I.cuda()
        self.u_T[text_ids] = new_u_T.cuda()

    def forward(self, image_features, text_features, image_ids, text_ids, epoch, max_epoch,
                last_g_I=None, last_g_T=None, last_grad_tau_I=None, last_grad_tau_T=None, compute_last=False):

        if self.world_size > 1:
            image_features = torch.cat(GatherLayer.apply(image_features), dim=0)
            text_features = torch.cat(GatherLayer.apply(text_features), dim=0)

        # compute the logits (similarity between each image-text pair)
        sim = torch.einsum('i d, j d -> i j', image_features, text_features)
        diag_sim = torch.diagonal(sim)

        batch_size = sim.shape[0]

        # learnable temperature for each image and text
        tau_image = self.tau_I[image_ids]
        tau_text = self.tau_T[text_ids]

        # E_I(x_i)*E_T(t) - E_I(x_i)*E_T(t_i)
        image_diffs = sim - diag_sim[:, None]
        # E_I(x)*E_T(t_i) - E_I(x_i)*E_T(t_i)
        text_diffs = sim - diag_sim[None, :]

        image_diffs_d_temps = (image_diffs / tau_image[:, None]).clone().detach_()
        text_diffs_d_temps = (text_diffs / tau_text[None, :]).clone().detach_()

        # update b
        old_b_I = self.b_I[image_ids]
        new_b_I = torch.max(image_diffs_d_temps, old_b_I[:, None].tile(1, batch_size))
        self.b_I[image_ids] = torch.max(new_b_I, dim=1)[0]

        old_b_T = self.b_T[text_ids]
        new_b_T = torch.max(text_diffs_d_temps, old_b_T[None, :].tile(batch_size, 1))
        self.b_T[text_ids] = torch.max(new_b_T, dim=0)[0]

        exp_image_diffs = torch.exp(image_diffs_d_temps - self.b_I[image_ids][:, None]) # -b to avoid exp operation overflow
        exp_text_diffs = torch.exp(text_diffs_d_temps - self.b_T[text_ids][None, :])

        g_I = torch.sum(exp_image_diffs, dim=1, keepdim=True) / (batch_size-1)
        g_T = torch.sum(exp_text_diffs, dim=0, keepdim=True) / (batch_size-1)

        if not compute_last:
            self.s_I[image_ids] = (1.0-self.gamma_s) * self.s_I[image_ids] * torch.exp(old_b_I - self.b_I[image_ids]) + self.gamma_s * g_I.squeeze()
            self.s_T[text_ids] = (1.0-self.gamma_s) * self.s_T[text_ids] * torch.exp(old_b_T - self.b_T[text_ids]) + self.gamma_s * g_T.squeeze()
            
            if last_g_I is not None and last_g_T is not None:
                self.s_I[image_ids] += self.beta_s * (g_I - last_g_I.cuda())
                self.s_T[text_ids] += self.beta_s * (g_T - last_g_T.cuda())

            s_I = self.s_I[image_ids].reshape(g_I.shape)
            s_T = self.s_T[text_ids].reshape(g_T.shape)

        else:
            self.s_I[image_ids] = (1.0-self.gamma_s) * self.s_I[image_ids] * torch.exp(old_b_I - self.b_I[image_ids]) + self.gamma_s * g_I.squeeze()
            self.s_T[text_ids] = (1.0-self.gamma_s) * self.s_T[text_ids] * torch.exp(old_b_T - self.b_T[text_ids]) + self.gamma_s * g_T.squeeze()
            
            s_I = self.s_I[image_ids].reshape(g_I.shape)
            s_T = self.s_T[text_ids].reshape(g_T.shape)

        weights_image = exp_image_diffs / (s_I + self.eps)
        weights_text = exp_text_diffs / (s_T + self.eps)

        if torch.any(torch.isnan(weights_image)):
            assert 0, "weights_image has nan."
        if torch.any(torch.isnan(weights_text)):
            assert 0, "weights_text has nan."

        image_loss = torch.sum(weights_image * image_diffs, dim=1, keepdim=True) / (batch_size-1)
        text_loss = torch.sum(weights_text * text_diffs, dim=0, keepdim=True) / (batch_size-1)

        total_loss = image_loss.mean() + text_loss.mean()

        # gradient of tau for image and text
        grad_tau_image = torch.log(s_I) + self.b_I[image_ids][:, None] + self.rho_I - torch.sum(weights_image * image_diffs_d_temps, dim=1, keepdim=True) / (batch_size-1)
        grad_tau_text = torch.log(s_T) + self.b_T[text_ids][None, :] + self.rho_T - torch.sum(weights_text * text_diffs_d_temps, dim=0, keepdim=True) / (batch_size-1)
       
        if not compute_last:
            self.u_I[image_ids] = (1.0-self.gamma_u) * self.u_I[image_ids] + self.gamma_u * grad_tau_image.squeeze().clamp_(min=-self.grad_clip, max=self.grad_clip)
            self.u_T[text_ids] = (1.0-self.gamma_u) * self.u_T[text_ids] + self.gamma_u * grad_tau_text.squeeze().clamp_(min=-self.grad_clip, max=self.grad_clip)

            if last_grad_tau_I is not None and last_grad_tau_T is not None:
                self.u_I[image_ids] += self.beta_u * (grad_tau_image - last_grad_tau_I.cuda())
                self.u_T[text_ids] += self.beta_u * (grad_tau_text - last_grad_tau_T.cuda())

        else:
            self.u_I[image_ids] = (1.0-self.gamma_u) * self.u_I[image_ids] + self.gamma_u * grad_tau_image.squeeze().clamp_(min=-self.grad_clip, max=self.grad_clip)
            self.u_T[text_ids] = (1.0-self.gamma_u) * self.u_T[text_ids] + self.gamma_u * grad_tau_text.squeeze().clamp_(min=-self.grad_clip, max=self.grad_clip)

        if self.eta_sched == 'cosine':
            eta_cur = self.eta_min + (self.eta_init - self.eta_min) * math.cos(math.pi * (epoch / max_epoch) / 2)
        elif self.eta_sched == 'exp':
            eta_cur = (self.eta_init - self.eta_min) * self.eta_exp_gamma ** (epoch-1) + self.eta_min
        elif self.eta_sched == 'const':
            eta_cur = self.eta_init
        else:
            assert 0, self.eta_sched + " is not supported."

        self.tau_I[image_ids] = (tau_image - eta_cur * self.u_I[image_ids]).clamp_(min=self.tau_min, max=self.tau_max)
        self.tau_T[text_ids] = (tau_text - eta_cur * self.u_T[text_ids]).clamp_(min=self.tau_min, max=self.tau_max)

        avg_image_tau = tau_image.mean().item()
        avg_text_tau = tau_text.mean().item()

        ma_update_dict = {'image_ids': image_ids, 'text_ids': text_ids,
                          'new_s_I': self.s_I[image_ids], 'new_s_T': self.s_T[text_ids],
                          'new_u_I': self.u_I[image_ids], 'new_u_T': self.u_T[text_ids]}

        return g_I, g_T, grad_tau_image, grad_tau_text, total_loss, avg_image_tau, avg_text_tau, ma_update_dict



class Group_iSogCLR_Loss(nn.Module):
    def __init__(self, taus_I, taus_T, group_info_I, group_info_T, eta_p, lambada, num_groups,
                N=2900000, gamma=0.8, rho_init=0.1, bsz=128, world_size=8, eps=1e-10):
        
        #Inputs:
        #   N is number of samples in training set
        
        super(Group_iSogCLR_Loss, self).__init__()
        self.world_size = world_size
        self.gamma_I, self.gamma_T = gamma, gamma
        
        self.taus_I = taus_I
        self.taus_T = taus_T
        self.group_info_I = group_info_I
        self.group_info_T = group_info_T
        self.eta_p = eta_p
        self.lambada = lambada
        self.num_groups = num_groups

        self.s_I = torch.zeros(N).cuda()
        self.s_T = torch.zeros(N).cuda()
        self.z_I = torch.zeros(num_groups).cuda()
        self.z_T = torch.zeros(num_groups).cuda()
        self.b_I = torch.zeros(N).cuda()
        self.b_T = torch.zeros(N).cuda()
        self.p_I = torch.ones(num_groups).cuda() / num_groups
        self.p_T = torch.ones(num_groups).cuda() / num_groups

        self.rho_I = rho_init
        self.rho_T = rho_init
        self.eps = eps
        self.batch_size = bsz
        self.grad_clip = 5.0
        self.mask_neg = (1.0 - torch.eye(bsz)).cuda()

    def index_to_groupid(self, index, group_info):
        group_ids = [group_info[ind.item()] for ind in index]
        group_ids_tensor = torch.tensor(group_ids, dtype=tensor.dtype).cuda()

        return group_ids_tensor

    def forward(self, image_features, text_features, image_ids, text_ids, epoch, max_epoch):
        if self.world_size > 1:
            image_features = torch.cat(GatherLayer.apply(image_features), dim=0)
            text_features = torch.cat(GatherLayer.apply(text_features), dim=0)

        # compute the logits (similarity between each image-text pair)
        sim = torch.einsum('i d, j d -> i j', image_features, text_features)
        diag_sim = torch.diagonal(sim)

        batch_size = sim.shape[0]

        # learnable temperature for each image and text
        tau_image = self.taus_I[image_ids].cuda()
        tau_text = self.taus_T[text_ids].cuda()

        # E_I(x_i)*E_T(t) - E_I(x_i)*E_T(t_i)
        image_diffs = sim - diag_sim[:, None]
        # E_I(x)*E_T(t_i) - E_I(x_i)*E_T(t_i)
        text_diffs = sim - diag_sim[None, :]

        image_diffs_d_temps = (image_diffs / tau_image[:, None]).clone().detach_()
        text_diffs_d_temps = (text_diffs / tau_text[None, :]).clone().detach_()

        # update b
        old_b_I = self.b_I[image_ids]
        new_b_I = torch.max(image_diffs_d_temps, old_b_I[:, None].tile(1, batch_size))
        self.b_I[image_ids] = torch.max(new_b_I, dim=1)[0]

        old_b_T = self.b_T[text_ids]
        new_b_T = torch.max(text_diffs_d_temps, old_b_T[None, :].tile(batch_size, 1))
        self.b_T[text_ids] = torch.max(new_b_T, dim=0)[0]

        exp_image_diffs = torch.exp(image_diffs_d_temps - self.b_I[image_ids][:, None]) # -b to avoid exp operation overflow
        exp_text_diffs = torch.exp(text_diffs_d_temps - self.b_T[text_ids][None, :])

        g_I = torch.sum(exp_image_diffs, dim=1, keepdim=True) / (batch_size-1)
        g_T = torch.sum(exp_text_diffs, dim=0, keepdim=True) / (batch_size-1)

        self.s_I[image_ids] = (1.0-self.gamma_I) * self.s_I[image_ids] * torch.exp(old_b_I - self.b_I[image_ids]) + self.gamma_I * g_I.squeeze()
        self.s_T[text_ids] = (1.0-self.gamma_T) * self.s_T[text_ids] * torch.exp(old_b_T - self.b_T[text_ids]) + self.gamma_T * g_T.squeeze()
        
        s_I = self.s_I[image_ids].reshape(g_I.shape)
        s_T = self.s_T[text_ids].reshape(g_T.shape)

        weights_image = exp_image_diffs / (s_I + self.eps)
        weights_text = exp_text_diffs / (s_T + self.eps)

        if torch.any(torch.isnan(weights_image)):
            assert 0, "weights_image has nan."
        if torch.any(torch.isnan(weights_text)):
            assert 0, "weights_text has nan."

        # group weights for images and texts
        index_group_id_I = self.index_to_groupid(image_ids, self.group_info_I)
        index_group_id_T = self.index_to_groupid(text_ids, self.group_info_T)

        group_weight_I = (self.num_groups * self.p_I[index_group_id_I]).cuda().reshape(g_I.shape)
        group_weight_T = (self.num_groups * self.p_T[index_group_id_T]).cuda().reshape(g_T.shape)

        image_loss = torch.sum(weights_image * group_weight_I * image_diffs, dim=1, keepdim=True) / (batch_size-1)
        text_loss = torch.sum(weights_text * group_weight_T * text_diffs, dim=0, keepdim=True) / (batch_size-1)

        total_loss = image_loss.mean() + text_loss.mean()

        # update p
        F_I = self.taus_I[image_ids].cuda() * (torch.log(s_I) + self.b_I[image_ids] + self.rho_I)
        F_T = self.taus_T[text_ids].cuda() * (torch.log(s_T) + self.b_T[text_ids] + self.rho_T)

        index_group_id_mat_I = F.one_hot(index_group_id_I)
        group_counts_I = torch.sum(index_group_id_mat_I, dim=0)

        index_group_id_mat_T = F.one_hot(index_group_id_T)
        group_counts_T = torch.sum(index_group_id_mat_T, dim=0)

        grad_p_I = torch.sum(index_group_id_mat_I * F_I[None, :], dim=0) / group_counts_I
        grad_p_T = torch.sum(index_group_id_mat_T * F_T[None, :], dim=0) / group_counts_T

        self.z_I = (1-self.gamma_I) * self.z_I + self.gamma_I * grad_p_I
        self.z_T = (1-self.gamma_T) * self.z_T + self.gamma_T * grad_p_T

        grad_hp_I = -self.lambada * torch.log(self.p_I + self.eps) - self.lambada
        grad_hp_T = -self.lambada * torch.log(self.p_T + self.eps) - self.lambada

        new_p_I = self.p_I * torch.exp(2*self.eta_p * (self.z_I + grad_hp_I).clamp_(min=-self.grad_clip, max=self.grad_clip))
        new_p_T = self.p_T * torch.exp(2*self.eta_p * (self.z_T + grad_hp_T).clamp_(min=-self.grad_clip, max=self.grad_clip))

        self.p_I = new_p_I / new_p_I.sum()
        self.p_T = new_p_T / new_p_T.sum()


        return total_loss, self.p_I, self.p_T



"""
    https://github.com/goel-shashank/CyCLIP/blob/52d77af2a5f1a4bff01b4c371d6b98e2d0340137/src/train.py
"""
class CyCLIP_Loss(nn.Module):
    def __init__(self, world_size, temperature, cylambda_1=0.25 , cylambda_2=0.25):
        super(CyCLIP_Loss, self).__init__()

        self.world_size = world_size
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.cylambda_1 = cylambda_1
        self.cylambda_2 = cylambda_2


    def forward(self, image_features, text_features):
        if self.world_size > 1:
            image_features = torch.cat(GatherLayer.apply(image_features), dim=0)
            text_features = torch.cat(GatherLayer.apply(text_features), dim=0)

        batch_size = len(image_features)

        logits_text_per_image = (image_features @ text_features.t()) / self.temperature
        logits_image_per_text = logits_text_per_image.t()

        target = torch.arange(batch_size).long().cuda()

        # contrastive loss, the same as CLIP
        contrastive_loss = (self.criterion(logits_text_per_image, target) + self.criterion(logits_image_per_text, target)) / 2.0 

        # inmodal_cyclic_loss
        logits_image_per_image = (image_features @ image_features.t()) / self.temperature
        logits_text_per_text = (text_features @ text_features.t()) / self.temperature
        inmodal_cyclic_loss = (logits_image_per_image - logits_text_per_text).square().mean() * (self.temperature ** 2) * batch_size

        # crossmodal_cyclic_loss
        crossmodal_cyclic_loss = (logits_text_per_image - logits_image_per_text).square().mean() * (self.temperature ** 2) * batch_size

        loss = contrastive_loss + self.cylambda_1 * inmodal_cyclic_loss + self.cylambda_2 * crossmodal_cyclic_loss

        return loss


"""
    VICReg
    https://github.com/facebookresearch/vicreg/blob/main/main_vicreg.py
"""
def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
    

class VICReg_Loss(nn.Module):
    def __init__(self, world_size, dim_size, sim_coeff=25.0, std_coeff=25.0, cov_coeff=1.0):
        super(VICReg_Loss, self).__init__()

        self.world_size = world_size
        self.dim_size = dim_size
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff


    def forward(self, image_features, text_features):
        if self.world_size > 1:
            x = torch.cat(GatherLayer.apply(image_features), dim=0)
            y = torch.cat(GatherLayer.apply(text_features), dim=0)

        batch_size = len(x)

        repr_loss = F.mse_loss(x, y) # invariance term

        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2  # variance term

        cov_x = (x.T @ x) / (batch_size - 1)
        cov_y = (y.T @ y) / (batch_size - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(
            self.dim_size
        ) + off_diagonal(cov_y).pow_(2).sum().div(self.dim_size)  # covariance term

        loss = (
            self.sim_coeff * repr_loss
            + self.std_coeff * std_loss
            + self.cov_coeff * cov_loss
        )

        return loss


