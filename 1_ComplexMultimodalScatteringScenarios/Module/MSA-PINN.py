# Module/MSA-PINN.py
import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, node_num, output_num):
        super().__init__()

        # ===== bookkeeping (Training.py expects these) =====
        self.iter = 0
        self.iter_list = []

        self.loss_list = []
        self.loss_f_list = []
        self.loss_b_list = []
        self.loss_d_list = []
        self.loss_rgl_list = []
        self.para_ud_list = []

        # Maxwell RelL2 logging
        self.rel_iter_list = []
        self.relEz_list = []
        self.relHx_list = []
        self.relHy_list = []

        self.node_num = node_num
        self.output_num = output_num

        # ---------------------------------
        # input projections: x, y, t
        # ---------------------------------
        self.fx1 = nn.Linear(1, node_num)
        self.fx2 = nn.Linear(1, node_num)

        self.fy1 = nn.Linear(1, node_num)
        self.fy2 = nn.Linear(1, node_num)

        self.ft1 = nn.Linear(1, node_num)
        self.ft2 = nn.Linear(1, node_num)

        # ---------------------------------
        # branch transforms (8 branches)
        # ---------------------------------
        self.g1 = nn.Linear(node_num, node_num)
        self.g2 = nn.Linear(node_num, node_num)
        self.g3 = nn.Linear(node_num, node_num)
        self.g4 = nn.Linear(node_num, node_num)
        self.g5 = nn.Linear(node_num, node_num)
        self.g6 = nn.Linear(node_num, node_num)
        self.g7 = nn.Linear(node_num, node_num)
        self.g8 = nn.Linear(node_num, node_num)

        # ---------------------------------
        # light branch attention
        # input: (x, y, t)
        # output: 8 branch weights
        # ---------------------------------
        self.attn = nn.Sequential(
            nn.Linear(3, 32),
            nn.Tanh(),
            nn.Linear(32, 8)
        )

        # ---------------------------------
        # third layer: reduce to 4 branches
        # ---------------------------------
        self.h1 = nn.Linear(node_num, node_num)
        self.h2 = nn.Linear(node_num, node_num)
        self.h3 = nn.Linear(node_num, node_num)
        self.h4 = nn.Linear(node_num, node_num)

        # ---------------------------------
        # output layer
        # ---------------------------------
        self.out = nn.Linear(4 * node_num, output_num)

        self.act = nn.Tanh()

    def forward(self, inp):
        """
        inp: [N, 3] -> columns are (x, y, t)
        out: [N, output_num] -> usually (Ez, Hx, Hy)
        """
        x = inp[:, 0:1]
        y = inp[:, 1:2]
        t = inp[:, 2:3]

        # ---------------------------------
        # first 4 combinations
        # ---------------------------------
        ax = self.fx1(x)
        by = self.fy1(y)
        ct = self.ft1(t)

        u1 = self.act(ax + by + ct)
        u2 = self.act(ax + by - ct)
        u3 = self.act(ax - by + ct)
        u4 = self.act(ax - by - ct)

        # ---------------------------------
        # second 4 combinations
        # ---------------------------------
        ax2 = self.fx2(x)
        by2 = self.fy2(y)
        ct2 = self.ft2(t)

        u5 = self.act(ax2 + by2 + ct2)
        u6 = self.act(ax2 + by2 - ct2)
        u7 = self.act(ax2 - by2 + ct2)
        u8 = self.act(ax2 - by2 - ct2)

        # ---------------------------------
        # independent branch transforms
        # ---------------------------------
        v1 = self.act(self.g1(u1))
        v2 = self.act(self.g2(u2))
        v3 = self.act(self.g3(u3))
        v4 = self.act(self.g4(u4))
        v5 = self.act(self.g5(u5))
        v6 = self.act(self.g6(u6))
        v7 = self.act(self.g7(u7))
        v8 = self.act(self.g8(u8))

        # ---------------------------------
        # branch attention
        # alpha: [N, 8], each row sums to 1
        # ---------------------------------
        alpha = torch.softmax(self.attn(inp), dim=1)

        v1 = alpha[:, 0:1] * v1
        v2 = alpha[:, 1:2] * v2
        v3 = alpha[:, 2:3] * v3
        v4 = alpha[:, 3:4] * v4
        v5 = alpha[:, 4:5] * v5
        v6 = alpha[:, 5:6] * v6
        v7 = alpha[:, 6:7] * v7
        v8 = alpha[:, 7:8] * v8

        # ---------------------------------
        w1 = self.act(self.h1(v1 + v2))
        w2 = self.act(self.h2(v3 + v4))
        w3 = self.act(self.h3(v5 - v6))
        w4 = self.act(self.h4(v7 - v8))

        feat = torch.cat([w1, w2, w3, w4], dim=1)
        out = self.out(feat)
        return out