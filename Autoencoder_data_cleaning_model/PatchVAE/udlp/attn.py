import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class Self_Attn(nn.Module):
    '''
    input: batch_size x feature_depth x feature_size x feature_size
    attn_score: batch_size x feature_size x feature_size
    output: batch_size x feature_depth x feature_size x feature_size
    '''

    def __init__(self, in_dim, skip=True, g_pool_in=False, g_pool_out=False, sparse=False):
        super(Self_Attn, self).__init__()
        self.in_dim = in_dim
        self.skip = skip
        self.g_pool_in = g_pool_in
        self.g_pool_out = g_pool_out
        self.sparse = sparse

        self.f_ = nn.Sequential(nn.Conv2d(in_dim, int(in_dim / 2), 1), nn.ReLU(), nn.Conv2d(int(in_dim/2), int(in_dim/2), 1))
        self.g_ = nn.Sequential(nn.Conv2d(in_dim, int(in_dim / 2), 1), nn.ReLU(), nn.Conv2d(int(in_dim/2), int(in_dim/2), 1))
        self.h_ = nn.Conv2d(in_dim, in_dim, 1)
        if self.sparse:
            self.sparsemax = SparseMax(0.1)

        if self.skip:
            self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.size()
        H_g, W_g = H, W

        f_x = self.f_(x).view(B, -1, H*W).permute(0, 2, 1)  # B x HW_f x C_out
        g_x = self.g_(x).view(B, -1, H_g*W_g)  # B x C_out x HW_g
        if self.g_pool_in:
            g_x = g_x.mean(-1, keepdim=True)  # B x C_out x 1
            H_g = W_g = 1
        h_x = self.h_(x).view(B, -1, H*W)  # B x C x HW_f

        attn_dist = torch.bmm(f_x, g_x)  # B x HW_f x HW_g
        if self.sparse:
            attn_soft = self.sparsemax(attn_dist, dim=1)
        else:
            attn_soft = F.softmax(attn_dist, dim=1)  # B x HW_f x HW_g
        self_attn_map = torch.bmm(h_x, attn_soft).view(B, -1, H_g, W_g)  # B x C x H_g x W_g

        if self.skip:
            self_attn_map = self.gamma * self_attn_map + x

        if self.g_pool_out:
            self_attn_map = self_attn_map.mean(3, keepdim=True).mean(2, keepdim=True)

        return self_attn_map, attn_soft.view(B, H, W, -1)

class Pixel_Attn(nn.Module):
    '''
    input: batch_size x feature_depth x feature_size x feature_size
    attn_score: batch_size x feature_size x feature_size
    output: batch_size x feature_depth x feature_size x feature_size
    '''

    def __init__(self, in_dim):
        super(Pixel_Attn, self).__init__()
        self.mask_w = nn.Conv2d(in_dim, 1, kernel_size=1, stride=1, padding=0)
        # self.act = nn.Softmax(dim=2)
        self.act = nn.Sigmoid()
        # self.act = nn.Tanh()
        # self.act = nn.ReLU()
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, feat):
        mask = self.mask_w(feat)
        B, C, _, _ = feat.size()
        m_mean = mask.view(B, 1, -1).mean(2, keepdim=True)
        m_std = mask.view(B, 1, -1).std(2, keepdim=True)
        mask = (mask - m_mean.unsqueeze(3)) / m_std.unsqueeze(3)
        mask = self.act(mask)
        feat = feat - self.gamma * feat * mask
        gap_feat = feat.view(B, C, -1).sum(2) / mask.view(B, 1, -1).sum(2)
        return gap_feat.view(B, C, 1, 1), mask

class Inv_Attn(nn.Module):
    '''
    input: batch_size x feature_depth x feature_size x feature_size
    attn_score: batch_size x feature_size x feature_size
    output: batch_size x feature_depth x feature_size x feature_size
    '''

    def __init__(self, in_dim, skip=True):
        super(Inv_Attn, self).__init__()
        self.in_dim = in_dim
        self.skip = skip

        self.f_ = nn.Conv2d(in_dim, int(in_dim / 2), 1)
        self.g_ = nn.Conv2d(in_dim, int(in_dim / 2), 1)
        self.h_ = nn.Conv2d(in_dim, in_dim, 1)

        if self.skip:
            self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, w):
        B, C, H, W = x.size()

        f_w = self.f_(w).view(B, -1, H*W).permute(0, 2, 1)  # B x HW_w x C_out
        g_x = self.g_(x).view(B, -1, H*W)  # B x C_out x HW_x
        h_x = self.h_(x).view(B, -1, H*W)  # B x C x HW_x

        attn_dist = torch.bmm(f_w, g_x)  # B x HW_w x HW_x
        attn_soft = F.softmax(attn_dist, dim=1).permute(0, 2, 1)  # B x HW_x x HW_w
        self_attn_map = torch.bmm(h_x, attn_soft).view(B, -1, H, W)  # B x C x H_w x W_w
        if self.skip:
            self_attn_map = self.gamma * self_attn_map + w

        return self_attn_map, attn_soft

class Inv_Attn_Sim(nn.Module):
    '''
    input: batch_size x feature_depth x feature_size x feature_size
    attn_score: batch_size x feature_size x feature_size
    output: batch_size x feature_depth x feature_size x feature_size
    '''

    def __init__(self, in_dim):
        super(Inv_Attn_Sim, self).__init__()
        self.in_dim = in_dim

        self.h_ = nn.Conv2d(in_dim, in_dim, 1)

    def forward(self, x, mask):
        mask = mask.squeeze().unsqueeze(1)

        h_x = self.h_(x * mask)  # B x C x H x W

        return h_x

class SparseMax(nn.Module):
    def __init__(self, alpha):
        super(SparseMax, self).__init__()
        self.alpha = alpha

    def forward(self, x, dim=1):
        x_max, x_min = x.max(dim, keepdim=True)[0], x.min(dim, keepdim=True)[0]
        x_range = x_max - x_min
        scale = 2 / (self.alpha**2 * x.size(dim) * x_range)
        x = x - x_min
        x = x * scale
        step_array = torch.arange(1, x.size(dim) + 1)
        if x.is_cuda:
            step_array = step_array.cuda()
        dim_expand = torch.ones(len(x.shape))
        dim_expand[dim] = x.size(dim)
        z = torch.sort(x, dim=dim, descending=True)[0]
        cum_z = torch.cumsum(z, dim=dim)
        cond = (1 + step_array.view(tuple(dim_expand))*z - cum_z).gt(0).float().detach()
        tau = ((z * cond).sum(dim) - 1)/cond.sum(dim)
        p = (x - tau.unsqueeze(dim)).clamp(min=0)
        print(p[0, :, 0].gt(0).sum())
        return p