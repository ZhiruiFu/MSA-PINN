# coding = utf-8
import numpy as np
import torch
import torch.optim as optim
import pandas as pd
import os
import importlib
import time
import Module.SingleVis as SingleVis
import Module.GroupVis as GroupVis

torch.manual_seed(1234)

if torch.cuda.is_available():
    device = torch.device('cuda')
    print("GPU is available")
else:
    device = torch.device('cpu')


class model():
    def __init__(self, ques_name, ini_num):

        self.ques_name = ques_name
        self.ini_num = ini_num

        self.ini_file_path = f'./Config/{ques_name}_{ini_num}.csv'

        data = pd.read_csv(
            self.ini_file_path,
            header=None,
            names=['key', 'value'],
            usecols=[0, 1]
        )

        self.model_ini_dict = {}
        for _, row in data.iterrows():
            key = row['key']
            value = row['value']

            if 'min' in key or 'max' in key:
                self.model_ini_dict[key] = float(value)
            elif 'num' in key or 'state' in key:
                self.model_ini_dict[key] = int(value)
            else:
                self.model_ini_dict[key] = str(value)

        self.pace_record_state = self.model_ini_dict['pace_record_state']
        self.node_num = self.model_ini_dict['node_num']

        self.coord_num = self.model_ini_dict['coord_num'] if 'coord_num' in self.model_ini_dict else self.model_ini_dict['input_num']
        self.output_num = self.model_ini_dict['output_num']

        self.learning_rate = float(self.model_ini_dict['learning_rate']) if 'learning_rate' in self.model_ini_dict else 1e-4

        self.model_ini_dict['model'] = self.model_ini_dict['model'].split(' ')

        self.x_min = self.model_ini_dict['x_min']
        self.x_max = self.model_ini_dict['x_max']
        self.y_min = self.model_ini_dict['y_min']
        self.y_max = self.model_ini_dict['y_max']
        self.z_min = self.model_ini_dict['z_min'] if 'z_min' in self.model_ini_dict else 0.0
        self.z_max = self.model_ini_dict['z_max'] if 'z_max' in self.model_ini_dict else 1.0

        self.input_num = self.coord_num

        self.hidden_layers_group = list(map(float, self.model_ini_dict['hidden_layers_group'].split(',')))
        self.layer = [self.input_num, self.output_num]
        self.layer[1:1] = list(map(lambda x: x * self.node_num, self.hidden_layers_group))
        self.layer = list(map(int, self.layer))

        self.grid_node_num = self.model_ini_dict['grid_node_num']
        self.regular_state = self.model_ini_dict['regularization_state']
        self.load_state = self.model_ini_dict['load_state']

        self.step_num = self.model_ini_dict['step_num'] if self.model_ini_dict['step_num'] < 10 else 1
        self.bun_node_num = self.model_ini_dict['bun_node_num']
        self.figure_node_num = self.model_ini_dict['figure_node_num']

        self.milestone = list(map(int, self.model_ini_dict['milestone'].split(','))) if 'milestone' in self.model_ini_dict else []
        self.gamma = float(self.model_ini_dict['gamma']) if 'gamma' in self.model_ini_dict else 0.5

        self.pace_record_gap = list(map(int, self.model_ini_dict['pace_record_gap'].split(','))) if 'pace_record_gap' in self.model_ini_dict else [100]
        self.pace_record_skip = list(map(int, self.model_ini_dict['pace_record_skip'].split(','))) if 'pace_record_skip' in self.model_ini_dict else [0]

        if int(self.model_ini_dict['step_num']) > 10000:
            self.train_steps = int(self.model_ini_dict['step_num'])
        elif int(self.model_ini_dict['step_num']) < 1000:
            self.train_steps = int(self.model_ini_dict['train_steps'])
        else:
            self.train_steps = 100000

        self.save_desti = f'./Results/{self.ques_name}_{str(self.ini_num)}/'

        # Maxwell-only 固定状态
        self.monitor_state = False
        self.distill_state = False
        self.load_study_state = False

    def mesh_init(self):
        self.x = np.linspace(self.x_min, self.x_max, self.grid_node_num).reshape([-1, 1])
        self.y = np.linspace(self.y_min, self.y_max, self.grid_node_num).reshape([-1, 1])
        self.z = np.linspace(self.z_min, self.z_max, self.grid_node_num).reshape([-1, 1])

        self.x, self.y, self.z = np.meshgrid(self.x, self.y, self.z)

        self.x = torch.tensor(self.x, requires_grad=True).float().to(device).reshape([-1, 1])
        self.y = torch.tensor(self.y, requires_grad=True).float().to(device).reshape([-1, 1])
        self.z = torch.tensor(self.z, requires_grad=True).float().to(device).reshape([-1, 1])

    def net_b(self):
        # === PEC plane-wall multi-angle scattering: exact boundary + exact initial condition ===

        t_b = torch.linspace(
            self.z_min, self.z_max, self.bun_node_num,
            requires_grad=True
        ).float().to(device).reshape([-1, 1])

        y_line = torch.linspace(
            self.y_min, self.y_max, self.bun_node_num,
            requires_grad=True
        ).float().to(device).reshape([-1, 1])

        x_line = torch.linspace(
            self.x_min, self.x_max, self.bun_node_num,
            requires_grad=True
        ).float().to(device).reshape([-1, 1])

        def eval_u(xv, yv, tv):
            return self.net(torch.cat([xv, yv, tv], dim=1))

        # -----------------------------
        # boundary 1: x = x_min (PEC wall)
        # Only enforce PEC tangential electric field Ez = 0
        # -----------------------------
        Nt = t_b.shape[0]
        Ny = y_line.shape[0]

        x0 = torch.full((Nt * Ny, 1), self.x_min, requires_grad=True).float().to(device)
        y_rep = y_line.repeat(Nt, 1)
        t_rep = t_b.repeat_interleave(Ny, dim=0)

        out_x0 = eval_u(x0, y_rep, t_rep)
        Ez_x0 = out_x0[:, 0:1]

        loss_bc_x0 = (Ez_x0 ** 2).mean()

        # -----------------------------
        # boundary 2: x = x_max
        # exact total field
        # -----------------------------
        x1 = torch.full((Nt * Ny, 1), self.x_max, requires_grad=True).float().to(device)
        out_x1 = eval_u(x1, y_rep, t_rep)

        Ez_x1_t, Hx_x1_t, Hy_x1_t = self.pec_multiangle_scattering_exact(x1, y_rep, t_rep)

        loss_bc_x1 = ((out_x1[:, 0:1] - Ez_x1_t) ** 2).mean() \
                   + ((out_x1[:, 1:2] - Hx_x1_t) ** 2).mean() \
                   + ((out_x1[:, 2:3] - Hy_x1_t) ** 2).mean()

        # -----------------------------
        # boundary 3: y = y_min
        # exact total field
        # -----------------------------
        Nx = x_line.shape[0]

        y0 = torch.full((Nt * Nx, 1), self.y_min, requires_grad=True).float().to(device)
        x_rep = x_line.repeat(Nt, 1)
        t_rep2 = t_b.repeat_interleave(Nx, dim=0)

        out_y0 = eval_u(x_rep, y0, t_rep2)
        Ez_y0_t, Hx_y0_t, Hy_y0_t = self.pec_multiangle_scattering_exact(x_rep, y0, t_rep2)

        loss_bc_y0 = ((out_y0[:, 0:1] - Ez_y0_t) ** 2).mean() \
                   + ((out_y0[:, 1:2] - Hx_y0_t) ** 2).mean() \
                   + ((out_y0[:, 2:3] - Hy_y0_t) ** 2).mean()

        # -----------------------------
        # boundary 4: y = y_max
        # exact total field
        # -----------------------------
        y1 = torch.full((Nt * Nx, 1), self.y_max, requires_grad=True).float().to(device)
        out_y1 = eval_u(x_rep, y1, t_rep2)
        Ez_y1_t, Hx_y1_t, Hy_y1_t = self.pec_multiangle_scattering_exact(x_rep, y1, t_rep2)

        loss_bc_y1 = ((out_y1[:, 0:1] - Ez_y1_t) ** 2).mean() \
                   + ((out_y1[:, 1:2] - Hx_y1_t) ** 2).mean() \
                   + ((out_y1[:, 2:3] - Hy_y1_t) ** 2).mean()

        loss_bc = loss_bc_x0 + loss_bc_x1 + loss_bc_y0 + loss_bc_y1

        # -----------------------------
        # initial condition at t = 0
        # exact total field
        # -----------------------------
        N0 = self.bun_node_num
        x0i = torch.linspace(self.x_min, self.x_max, N0, requires_grad=True).float().to(device).reshape([-1, 1])
        y0i = torch.linspace(self.y_min, self.y_max, N0, requires_grad=True).float().to(device).reshape([-1, 1])

        xx, yy = torch.meshgrid(x0i.squeeze(1), y0i.squeeze(1), indexing='ij')
        xx = xx.reshape([-1, 1])
        yy = yy.reshape([-1, 1])
        tt0 = torch.zeros_like(xx, requires_grad=True).float().to(device)

        out0 = self.net(torch.cat([xx, yy, tt0], dim=1))
        Ez0 = out0[:, 0:1]
        Hx0 = out0[:, 1:2]
        Hy0 = out0[:, 2:3]

        Ez0_t, Hx0_t, Hy0_t = self.pec_multiangle_scattering_exact(xx, yy, tt0)

        loss_ic = ((Ez0 - Ez0_t) ** 2).mean() \
                + ((Hx0 - Hx0_t) ** 2).mean() \
                + ((Hy0 - Hy0_t) ** 2).mean()

        loss_b = loss_bc + loss_ic
        return loss_b

    def net_f(self):
        inp = torch.cat([self.x, self.y, self.z], dim=1)
        u = self.net(inp).to(device)

        Ez, Hx, Hy = torch.split(u, 1, dim=1)

        Ez_x = torch.autograd.grad(Ez, self.x, grad_outputs=torch.ones_like(Ez), retain_graph=True, create_graph=True)[0]
        Ez_y = torch.autograd.grad(Ez, self.y, grad_outputs=torch.ones_like(Ez), retain_graph=True, create_graph=True)[0]
        Ez_t = torch.autograd.grad(Ez, self.z, grad_outputs=torch.ones_like(Ez), retain_graph=True, create_graph=True)[0]

        Hx_y = torch.autograd.grad(Hx, self.y, grad_outputs=torch.ones_like(Hx), retain_graph=True, create_graph=True)[0]
        Hx_t = torch.autograd.grad(Hx, self.z, grad_outputs=torch.ones_like(Hx), retain_graph=True, create_graph=True)[0]

        Hy_x = torch.autograd.grad(Hy, self.x, grad_outputs=torch.ones_like(Hy), retain_graph=True, create_graph=True)[0]
        Hy_t = torch.autograd.grad(Hy, self.z, grad_outputs=torch.ones_like(Hy), retain_graph=True, create_graph=True)[0]

        # Maxwell TE 方程
        r1 = Hx_t + Ez_y
        r2 = Hy_t - Ez_x
        r3 = Ez_t - (Hy_x - Hx_y)

        loss_f = torch.mean(r1 ** 2) + torch.mean(r2 ** 2) + torch.mean(r3 ** 2)
        return loss_f

    def net_rgl(self, mode='teacher', object='all', reg_type='l2', weight_rgl=1e-3):
        loss_rgl = torch.tensor(0.).to(device)

        parameters_rgl = self.net.named_parameters()

        if object == 'all':
            for name, param in parameters_rgl:
                if reg_type == 'l2':
                    loss_rgl += weight_rgl * torch.norm(param, p=2)
                elif reg_type == 'l1':
                    loss_rgl += weight_rgl * torch.norm(param, p=1)

        elif object == 'weight':
            for name, param in parameters_rgl:
                if 'weight' in name:
                    if reg_type == 'l2':
                        loss_rgl += weight_rgl * torch.norm(param, p=2)
                    elif reg_type == 'l1':
                        loss_rgl += weight_rgl * torch.norm(param, p=1)
                    elif reg_type == 'growl':
                        row_norms = torch.norm(param, p=2, dim=1)
                        sorted_row_norms, _ = torch.sort(row_norms, descending=True)
                        lambda_vals = torch.linspace(1, 0.1, steps=sorted_row_norms.size(0)).to(device)
                        lambda_vals = lambda_vals[:sorted_row_norms.size(0)]
                        loss_rgl += torch.sum(lambda_vals * sorted_row_norms)

        return loss_rgl

    def get_eval_times(self, num_points: int = 5):
        """
        根据配置中的时间范围自动生成评估/绘图时刻。
        默认取 5 个等间距时刻，例如 [0, 2] -> [0.0, 0.5, 1.0, 1.5, 2.0]
        """
        t_start = float(self.z_min)
        t_end = float(self.z_max)

        if num_points <= 1 or abs(t_end - t_start) < 1e-12:
            return [t_start]

        return list(np.linspace(t_start, t_end, num_points))

    def pec_multiangle_scattering_exact(self, x, y, t):
        """
        Multi-angle PEC plane-wall scattering exact total field.
        PEC wall is at x = x_min.

        The total field is a superposition of several oblique incident-reflected pairs:
            Ez_j = A_j [sin(theta_i_j) - sin(theta_r_j)]
            Hx_j = A_j (ky_j / omega_j) [sin(theta_i_j) - sin(theta_r_j)]
            Hy_j = A_j (kx_j / omega_j) [sin(theta_i_j) + sin(theta_r_j)]

        where
            theta_i_j = -kx_j * (x - x_min) + ky_j * (y - y_min) - omega_j * t + phi_j
            theta_r_j = +kx_j * (x - x_min) + ky_j * (y - y_min) - omega_j * t + phi_j

        Notes:
        1) Ez = 0 exactly at x = x_min, so the PEC boundary is satisfied.
        2) ky_j is chosen as 2*pi*n/Ly, so the field is exactly periodic in y and
           does not conflict with the top/bottom boundaries when exact traces are used.
        """
        Lx = float(self.x_max - self.x_min)
        Ly = float(self.y_max - self.y_min)

        if Lx <= 0 or Ly <= 0:
            raise ValueError("Domain lengths must be positive.")

        pi_val = torch.tensor(np.pi, dtype=x.dtype, device=x.device)

        xr = x - self.x_min
        yr = y - self.y_min

        Ez = torch.zeros_like(x)
        Hx = torch.zeros_like(x)
        Hy = torch.zeros_like(x)

        # -------------------------------------------------
        # Three oblique components
        # ky_j = 2*pi*n_j/Ly  -> exact periodicity in y
        # choose moderate amplitudes and distinct phases
        # -------------------------------------------------
        mode_list = [
            # amplitude, ky_mode, kx_scale, phase
            (1.00, 1.0, 1.20, 0.00),
            (0.55, 2.0, 1.70, 0.35 * np.pi),
            (0.30, 3.0, 2.10, -0.20 * np.pi),
        ]

        for A, ny_mode, kx_scale, phase in mode_list:
            ky = 2.0 * ny_mode * pi_val / Ly
            kx = kx_scale * pi_val / Lx
            omega = torch.sqrt(kx ** 2 + ky ** 2)
            phi = torch.tensor(float(phase), dtype=x.dtype, device=x.device)

            theta_i = -kx * xr + ky * yr - omega * t + phi
            theta_r =  kx * xr + ky * yr - omega * t + phi

            sin_i = torch.sin(theta_i)
            sin_r = torch.sin(theta_r)

            Ez = Ez + A * (sin_i - sin_r)
            Hx = Hx + A * (ky / omega) * (sin_i - sin_r)
            Hy = Hy + A * (kx / omega) * (sin_i + sin_r)

        return Ez, Hx, Hy

    def maxwell_exact_and_error(self, N=None, t0=0.5):
        """
        返回：
        mse_Ez, mse_Hx, mse_Hy,
        relL2_Ez, relL2_Hx, relL2_Hy,
        rmse_Ez, rmse_Hx, rmse_Hy,
        mse_total, relL2_total, rmse_total
        """
        if N is None:
            N = self.figure_node_num

        x = torch.linspace(self.x_min, self.x_max, N).reshape(-1, 1).to(device)
        y = torch.linspace(self.y_min, self.y_max, N).reshape(-1, 1).to(device)
        xx, yy = torch.meshgrid(x.squeeze(1), y.squeeze(1), indexing='ij')
        xx = xx.reshape(-1, 1)
        yy = yy.reshape(-1, 1)
        tt = torch.full_like(xx, float(t0)).to(device)

        self.net.eval()
        with torch.no_grad():
            out = self.net(torch.cat([xx, yy, tt], dim=1))
            Ez_p = out[:, 0:1]
            Hx_p = out[:, 1:2]
            Hy_p = out[:, 2:3]

            Ez_e, Hx_e, Hy_e = self.pec_multiangle_scattering_exact(xx, yy, tt)

        def mse(a, b):
            return torch.mean((a - b) ** 2).item()

        def rel_l2(a, b):
            rmse = torch.sqrt(torch.mean((a - b) ** 2))
            rms_true = torch.sqrt(torch.mean(b ** 2)) + 1e-12
            return (rmse / rms_true).item()

        def rmse(a, b):
            return torch.sqrt(torch.mean((a - b) ** 2)).item()

        mse_Ez = mse(Ez_p, Ez_e)
        mse_Hx = mse(Hx_p, Hx_e)
        mse_Hy = mse(Hy_p, Hy_e)

        relL2_Ez = rel_l2(Ez_p, Ez_e)
        relL2_Hx = rel_l2(Hx_p, Hx_e)
        relL2_Hy = rel_l2(Hy_p, Hy_e)

        rmse_Ez = rmse(Ez_p, Ez_e)
        rmse_Hx = rmse(Hx_p, Hx_e)
        rmse_Hy = rmse(Hy_p, Hy_e)

        pred_stack = torch.cat([Ez_p, Hx_p, Hy_p], dim=1)
        exact_stack = torch.cat([Ez_e, Hx_e, Hy_e], dim=1)

        mse_total = mse(pred_stack, exact_stack)
        relL2_total = rel_l2(pred_stack, exact_stack)
        rmse_total = rmse(pred_stack, exact_stack)

        self.net.train()
        return (
            mse_Ez, mse_Hx, mse_Hy,
            relL2_Ez, relL2_Hx, relL2_Hy,
            rmse_Ez, rmse_Hx, rmse_Hy,
            mse_total, relL2_total, rmse_total
        )

    def train_adam(self):
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=self.milestone,
            gamma=self.gamma
        )

        self.current_time = time.time()
        self.time_list = [0.]

        current_gap_teacher = self.pace_record_gap[0]

        for iter_group in range(self.step_num):
            for iter_inner in range(self.train_steps):

                self.optimizer.zero_grad()

                self.loss_f = self.net_f()
                self.loss_b = self.net_b()
                self.loss_rgl = self.net_rgl(object='all', reg_type='l2') if self.regular_state else torch.tensor(0.).to(device)

                self.loss = self.loss_f + self.loss_b
                if self.regular_state:
                    self.loss += self.loss_rgl

                self.loss.backward(retain_graph=True)
                self.optimizer.step()
                self.scheduler.step()

                self.net.iter += 1
                self.net.iter_list.append(self.net.iter)
                self.net.loss_list.append(self.loss.item())
                self.net.loss_f_list.append(self.loss_f.item())
                self.net.loss_b_list.append(self.loss_b.item())
                self.net.loss_d_list.append(0.0)
                self.net.loss_rgl_list.append(self.loss_rgl.item())

                # 新增 MSE 记录（不改原有 loss 定义）
                self.net.mse_list.append((self.loss_f + self.loss_b).item())
                self.net.mse_f_list.append(self.loss_f.item())
                self.net.mse_b_list.append(self.loss_b.item())

                if self.net.iter - 1 in self.pace_record_skip:
                    iter_index_teacher = self.pace_record_skip.index(self.net.iter - 1)
                    current_gap_teacher = self.pace_record_gap[iter_index_teacher]

                self.loss_dict = {
                    'Iter': self.net.iter,
                    'Loss': self.loss.item(),
                    'Loss_f': self.loss_f.item(),
                    'Loss_b': self.loss_b.item(),
                    'Loss_rgl': self.loss_rgl.item(),
                    'MSE': (self.loss_f + self.loss_b).item()
                }

                if self.net.iter % current_gap_teacher == 0:
                    total_iter = self.step_num * self.train_steps
                    loss_str = ', '.join([
                        f'{key}: {int(value) if key == "Iter" else value:.5e}'
                        for key, value in self.loss_dict.items()
                        if key != "Iter" and value != 0
                    ])
                    iter_str = f'Iter: {{{self.net.iter}/{total_iter}}}'
                    print(f'{iter_str}, {loss_str}')

                    # 多个时刻的表格化打印
                    t_list = self.get_eval_times(num_points=5)

                    self.net.err_iter_list.append(self.net.iter)

                    if len(self.net.error_table_rows) == 0:
                        pass

                    print("    [Maxwell Error Table]")
                    print(f"    Domain: x in [{self.x_min}, {self.x_max}], y in [{self.y_min}, {self.y_max}], t in [{self.z_min}, {self.z_max}]")
                    print(f"    Eval times: {[round(v, 4) for v in t_list]}")
                    print("    ----------------------------------------------------------------------------------------------------------------")
                    print("        t        MSE_Ez       MSE_Hx       MSE_Hy      RelL2_Ez    RelL2_Hx    RelL2_Hy    RMSE_Total")
                    print("    ----------------------------------------------------------------------------------------------------------------")

                    for t0 in t_list:
                        (
                            mse_Ez, mse_Hx, mse_Hy,
                            relL2_Ez, relL2_Hx, relL2_Hy,
                            rmse_Ez, rmse_Hx, rmse_Hy,
                            mse_total, relL2_total, rmse_total
                        ) = self.maxwell_exact_and_error(t0=t0)

                        print(
                            f"      {t0:>4.2f}   "
                            f"{mse_Ez:>10.3e}  {mse_Hx:>10.3e}  {mse_Hy:>10.3e}  "
                            f"{relL2_Ez:>10.3e}  {relL2_Hx:>10.3e}  {relL2_Hy:>10.3e}  "
                            f"{rmse_total:>10.3e}"
                        )

                        self.net.error_table_rows.append({
                            'iter': self.net.iter,
                            't': t0,
                            'mse_Ez': mse_Ez,
                            'mse_Hx': mse_Hx,
                            'mse_Hy': mse_Hy,
                            'relL2_Ez': relL2_Ez,
                            'relL2_Hx': relL2_Hx,
                            'relL2_Hy': relL2_Hy,
                            'rmse_Ez': rmse_Ez,
                            'rmse_Hx': rmse_Hx,
                            'rmse_Hy': rmse_Hy,
                            'mse_total': mse_total,
                            'relL2_total': relL2_total,
                            'rmse_total': rmse_total
                        })

                    print("    ----------------------------------------------------------------------------------------------")

                    if self.pace_record_state:
                        self.model_save(str(self.net.iter))

                    current_lr = self.optimizer.param_groups[0]['lr']
                    if current_lr != self.original_lr:
                        print(f"Learning rate changed from {self.original_lr:.6f} to {current_lr:.6f}")
                    self.original_lr = current_lr

                self.time_list[0] += time.time() - self.current_time
                self.current_time = time.time()

        print(f'\nTime occupied: {(self.time_list[0]):.5e} s.\n')

    def model_save(self, suffix: str = '', mode: str = 'teacher'):

        if not os.path.exists('./Results/'):
            os.mkdir('./Results/')

        if not os.path.exists(self.save_desti):
            os.mkdir(self.save_desti)

        if not os.path.exists(f'{self.save_desti}/Models/'):
            os.mkdir(f'{self.save_desti}/Models/')

        in_net = self.net
        suffix_mode = ''

        if suffix == '':
            torch.save(
                in_net.state_dict(),
                f"{self.save_desti}/Models/{self.ques_name}_{self.ini_num}_{in_net.__module__.split('.')[-1]}{suffix_mode}.pth"
            )
        elif self.pace_record_state:
            torch.save(
                in_net.state_dict(),
                f"{self.save_desti}/Models/{self.ques_name}_{self.ini_num}_{in_net.__module__.split('.')[-1]}{suffix_mode}_step_{suffix}.pth"
            )

        # 详细误差表
        if hasattr(self.net, 'error_table_rows') and len(self.net.error_table_rows) > 0:
            if not os.path.exists(self.save_desti + '/Error/'):
                os.mkdir(self.save_desti + '/Error/')

            df_error_table = pd.DataFrame(self.net.error_table_rows)
            df_error_table.to_csv(
                f"{self.save_desti}/Error/{self.ques_name}_{str(self.ini_num)}_error_table_{self.net.__module__.split('.')[-1]}.csv",
                index=False
            )

        self.control_paras = pd.read_csv(self.ini_file_path)
        self.control_paras.to_csv(f'{self.save_desti}{self.ques_name}_{self.ini_num}.csv', index=False)

        if suffix == '':
            self.time_save = pd.DataFrame({
                'Question': [self.ques_name],
                'Number': [self.ini_num],
                'Module': [in_net.__module__.split('.')[-1]],
                'Training Time': [self.time_list[0]]
            })
            file_path = self.save_desti + 'Clock time.csv'
            if not os.path.isfile(file_path):
                self.time_save.to_csv(file_path, mode='a', index=False)
            else:
                self.time_save.to_csv(file_path, mode='a', index=False, header=False)

        loss_data_dict = {
            'iter': self.net.iter_list,
            'loss': self.net.loss_list,
            'loss_f': self.net.loss_f_list,
            'loss_b': self.net.loss_b_list,
            'loss_d': self.net.loss_d_list,
            'loss_rgl': self.net.loss_rgl_list
        }

        loss_data_dict = {key: value for key, value in loss_data_dict.items() if value != 0}
        df_loss_data = pd.DataFrame(loss_data_dict)
        df_loss_data = df_loss_data.loc[:, (df_loss_data != 0).any(axis=0)]

        if not os.path.exists(self.save_desti + '/Loss/'):
            os.mkdir(self.save_desti + '/Loss/')

        df_loss_data.to_csv(
            f"{self.save_desti}/Loss/{self.ques_name}_{str(self.ini_num)}_loss_{self.net.__module__.split('.')[-1]}.csv",
            index=False
        )

        # MSE 表
        if hasattr(self.net, 'mse_list') and len(self.net.mse_list) > 0:
            if not os.path.exists(self.save_desti + '/MSE/'):
                os.mkdir(self.save_desti + '/MSE/')

            df_mse = pd.DataFrame({
                'iter': self.net.iter_list,
                'mse': self.net.mse_list,
                'mse_f': self.net.mse_f_list,
                'mse_b': self.net.mse_b_list
            })

            df_mse.to_csv(
                f"{self.save_desti}/MSE/{self.ques_name}_{str(self.ini_num)}_mse_{self.net.__module__.split('.')[-1]}.csv",
                index=False
            )

    def result_show(self):
        print('[DEBUG] Maxwell: drawing 2D slices (Ez/Hx/Hy)')

        self.net.eval()

        fig_dir = os.path.join(self.save_desti, 'Figure')
        os.makedirs(fig_dir, exist_ok=True)

        t_list = self.get_eval_times(num_points=5)
        N = self.figure_node_num

        print(f'[DEBUG] Figure domain: x in [{self.x_min}, {self.x_max}], y in [{self.y_min}, {self.y_max}], t in [{self.z_min}, {self.z_max}]')
        print(f'[DEBUG] Figure times: {[round(v, 4) for v in t_list]}')

        x = torch.linspace(self.x_min, self.x_max, N).reshape(-1, 1).to(device)
        y = torch.linspace(self.y_min, self.y_max, N).reshape(-1, 1).to(device)
        xx, yy = torch.meshgrid(x.squeeze(1), y.squeeze(1), indexing='ij')
        xx = xx.reshape(-1, 1)
        yy = yy.reshape(-1, 1)

        def save_field(field2d, name, t0):
            import matplotlib.pyplot as plt
            plt.figure()
            plt.imshow(
                field2d.T,
                origin='lower',
                extent=[self.x_min, self.x_max, self.y_min, self.y_max],
                aspect='equal'
            )
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title(f'{self.ques_name} {self.net.__module__.split(".")[-1]} {name} at t={t0:.2f}')
            plt.colorbar()
            save_path = os.path.join(
                fig_dir,
                f'{self.ques_name}_{self.net.__module__.split(".")[-1]}_{name}_t{t0:.2f}.png'
            )
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            plt.close()

        for t0 in t_list:
            tt = torch.full_like(xx, float(t0)).to(device)
            with torch.no_grad():
                out = self.net(torch.cat([xx, yy, tt], dim=1))
                Ez = out[:, 0].cpu().numpy().reshape(N, N)
                Hx = out[:, 1].cpu().numpy().reshape(N, N)
                Hy = out[:, 2].cpu().numpy().reshape(N, N)

            save_field(Ez, 'Ez', t0)
            save_field(Hx, 'Hx', t0)
            save_field(Hy, 'Hy', t0)

        print(f'[DEBUG] Saved Maxwell 2D slices to: {fig_dir}')
        self.net.train()

    def workflow(self):
        self.mesh_init()
        self.train_adam()
        self.model_save()
        self.result_show()

    def train(self):

        model_define_trigger = 0

        if len(self.model_ini_dict['model']) > 1:
            group = GroupVis.Vis(self.ques_name, self.ini_num, self.save_desti)

        for i in range(len(self.model_ini_dict['model'])):

            self.original_lr = self.learning_rate

            model_define_trigger = 1
            module = importlib.import_module(f"Module.{self.model_ini_dict['model'][i]}")
            NetClass = getattr(module, 'Net')

            model_name = self.model_ini_dict['model'][i]
            is_pinn = model_name.startswith('PINN')

            if is_pinn:
                self.net = NetClass(self.layer).float().to(device)
            else:
                self.net = NetClass(self.node_num, self.output_num).float().to(device)

            if not hasattr(self.net, 'iter'):
                self.net.iter = 0
            if not hasattr(self.net, 'iter_list'):
                self.net.iter_list = []
            if not hasattr(self.net, 'loss_list'):
                self.net.loss_list = []
            if not hasattr(self.net, 'loss_f_list'):
                self.net.loss_f_list = []
            if not hasattr(self.net, 'loss_b_list'):
                self.net.loss_b_list = []
            if not hasattr(self.net, 'loss_d_list'):
                self.net.loss_d_list = []
            if not hasattr(self.net, 'loss_rgl_list'):
                self.net.loss_rgl_list = []

            # 历史记录缓冲区
            self.net.mse_list = []
            self.net.mse_f_list = []
            self.net.mse_b_list = []

            self.net.err_iter_list = []
            self.net.error_table_rows = []

            if self.load_state:
                load_path = f"./Results/{self.ques_name}_{self.ini_num}/Models/{self.ques_name}_{self.ini_num}_{self.net.__module__.split('.')[-1]}.pth"
                self.net.load_state_dict(torch.load(load_path, map_location=device))

            print(f'\nRunning Model: {self.model_ini_dict["model"][i]}\n')

            self.workflow()

            if len(self.model_ini_dict['model']) > 1:
                group.loss_read(self.net.__module__.split('.')[-1])

        if len(self.model_ini_dict['model']) > 1:
            group.loss_vis()

        if model_define_trigger == 0:
            raise ValueError('The model name is incorrect. Please check again.')