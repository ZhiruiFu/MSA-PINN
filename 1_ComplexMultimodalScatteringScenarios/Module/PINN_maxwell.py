# Module/PINN.py
# coding = utf-8
import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.iter = 0
        self.iter_list = []
        self.loss_list = []
        self.loss_f_list = []
        self.loss_b_list = []
        self.loss_d_list = []
        self.loss_rgl_list = []

        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))

        self.act = nn.Tanh()

        self.loss_teach_list = []
        self.loss_rgl_list = []
        self.loss_d_list = []
        self.para_ud_list = []

    def forward(self, x):
        # x: [N, 3] -> (x,y,t)
        for i in range(len(self.layers) - 1):
            x = self.act(self.layers[i](x))
        x = self.layers[-1](x)  # [N, 3] -> (Ez,Hx,Hy)
        return x