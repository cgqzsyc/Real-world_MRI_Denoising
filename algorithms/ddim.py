# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved

import torch
import numpy as np

class DDIM:
    def __init__(self, model):
        self.model = model
        self.device = 'cuda'
        self.eta = 0.5
        self.sdedit = False
        self.cond_awd = False
        self.start_step = 1000
        self.end_step = 0
        self.num_steps = 50
        self.beta_start = 1e-4
        self.beta_end = 2e-2
        self.betas = np.linspace(self.beta_start, self.beta_end, self.start_step, dtype=np.float64)
        self.betas = torch.from_numpy(self.betas)
        self.betas = self.betas.to(self.device)
        self.betas = torch.cat([torch.zeros(1).to(self.device), self.betas], dim=0).cuda().float()
        self.alphas = (1 - self.betas).cumprod(dim=0).to(self.device).float()
        self.alpha_fast = 0.01

    def alpha(self, t):
        return self.alphas.index_select(0, t + 1)

    @torch.no_grad()
    def sample(self, x, y, labels):
        ts = list(range(self.end_step, self.start_step, (self.start_step - self.end_step) // self.num_steps))
        x = self.initialize(x, y)
        n = x.size(0)
        ss = [-1] + list(ts[:-1])  # q(x_s|x_t,x_0), ensure that 0<s<t<T
        xt_s = [x.cpu()]
        x0_s = []
        
        xt = x
        
        for ti, si in zip(reversed(ts), reversed(ss)):
            t = torch.ones(n).to(x.device).long() * ti
            s = torch.ones(n).to(x.device).long() * si
            alpha_t = self.alpha(t).view(-1, 1, 1, 1)
            alpha_s = self.alpha(s).view(-1, 1, 1, 1)
            c1 = ((1 - alpha_t / alpha_s) * (1 - alpha_s) / (1 - alpha_t)).sqrt() * self.eta
            c2 = ((1 - alpha_s) - c1 ** 2).sqrt()
            if self.cond_awd:
                scale = alpha_s.sqrt() / (alpha_s.sqrt() - c2 * alpha_t.sqrt() / (1 - alpha_t).sqrt())
                scale = scale.view(-1)[0].item()
            else:
                scale = 1.0
            x0_pred = self.model(torch.cat((xt, y), dim=1), y, t / self.start_step, labels)
            et = (xt - x0_pred * alpha_t.sqrt()) / (1 - alpha_t).sqrt()  # 相比源代码，此处et改为直接由正向公式求得
            # x_result = self.nn_model(torch.cat((x_t, c), dim=1), c, t / self.start_step, labels_model, None)
            xs = alpha_s.sqrt() * x0_pred + c1 * torch.randn_like(xt) + c2 * et
            xt_s.append(xs.cpu())
            x0_s.append(x0_pred.cpu())
            xt = (1 - self.alpha_fast) * xs + self.alpha_fast * y
            
        # return list(reversed(xt_s)), list(reversed(x0_s))
        return xt.detach()
    
    def initialize(self, x, y):
        ts = list(range(self.end_step, self.start_step, (self.start_step - self.end_step) // self.num_steps))
        if self.sdedit:
            n = x.size(0)
            ti = ts[-1]
            t = torch.ones(n).to(x.device).long() * ti
            alpha_t = self.alpha(t).view(-1, 1, 1, 1)
            return x * alpha_t.sqrt() + torch.randn_like(x) * (1 - alpha_t).sqrt()
        else:
            return torch.randn_like(x)
