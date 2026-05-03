# Module/F-PINN.py

import math
import torch
import torch.nn as nn


class Net(nn.Module):

    def __init__(self, node_num, output_num):
        super().__init__()

        # bookkeeping (Training.py需要)
        self.iter = 0
        self.iter_list = []

        self.loss_list = []
        self.loss_f_list = []
        self.loss_b_list = []
        self.loss_d_list = []
        self.loss_rgl_list = []
        self.para_ud_list = []

        self.rel_iter_list = []
        self.relEz_list = []
        self.relHx_list = []
        self.relHy_list = []

        self.node_num = node_num
        self.output_num = output_num

        # Fourier frequency
        freq = torch.linspace(1.0, 4.0, node_num).reshape(1, -1)

        self.register_buffer("freq_x", freq.clone())
        self.register_buffer("freq_y", freq.clone())
        self.register_buffer("freq_t", freq.clone())

        # scale
        self.scale_x = nn.Parameter(torch.tensor(0.3))
        self.scale_y = nn.Parameter(torch.tensor(0.3))
        self.scale_t = nn.Parameter(torch.tensor(0.3))

        # MLP
        self.net = nn.Sequential(
            nn.Linear(6 * node_num, node_num),
            nn.Tanh(),

            nn.Linear(node_num, node_num),
            nn.Tanh(),

            nn.Linear(node_num, node_num),
            nn.Tanh(),

            nn.Linear(node_num, output_num)
        )


    def fourier_embed(self, s, freq, scale):

        arg = scale * math.pi * s @ freq

        return torch.cat([
            torch.sin(arg),
            torch.cos(arg)
        ], dim=1)


    def forward(self, inp):

        x = inp[:,0:1]
        y = inp[:,1:2]
        t = inp[:,2:3]

        ex = self.fourier_embed(x, self.freq_x, self.scale_x)
        ey = self.fourier_embed(y, self.freq_y, self.scale_y)
        et = self.fourier_embed(t, self.freq_t, self.scale_t)

        feat = torch.cat([ex,ey,et],dim=1)

        return self.net(feat)