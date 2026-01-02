import torch
import torch.nn.functional as F
import numpy as np

from algorithms import DDIM

class DPS(DDIM):
    def __init__(self, model):
        self.model = model
        # self.H = build_degredation_model(cfg) !
        # self.H_inv = ... !
        # self.cfg = cfg
        self.awd = True
        self.cond_awd = False
        self.mcg = False
        self.grad_term_weight = 0.1
        self.eta = 0.0
        self.original = True
        self.start_step = 1000
        self.end_step = 0
        self.num_steps = 200
        self.beta_start = 1e-4
        self.beta_end = 2e-2
        self.betas = np.linspace(self.beta_start, self.beta_end, self.start_step, dtype=np.float64)
        self.betas = torch.from_numpy(self.betas)
        self.betas = self.betas.to(self.device)
        self.betas = torch.cat([torch.zeros(1).to(self.device), self.betas], dim=0).cuda().float()
        self.alphas = (1 - self.betas).cumprod(dim=0).to(self.device).float()

    def alpha(self, t):
        return self.alphas.index_select(0, t + 1)

    def sample(self, x, y, labels):
        ts = list(range(self.end_step, self.start_step, (self.start_step - self.end_step) // self.num_steps))
        y_0 = y.copy()
        n = x.size(0)
        H = self.H

        x = self.initialize(x, y, ts, y_0=y_0)
        ss = [-1] + list(ts[:-1])
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
            xt = xt.clone().to('cuda').requires_grad_(True)

            if self.cond_awd:
                scale = alpha_s.sqrt() / (alpha_s.sqrt() - c2 * alpha_t.sqrt() / (1 - alpha_t).sqrt())
                scale = scale.view(-1)[0].item()
            else:
                scale = 1.0

            x0_pred = self.model(torch.cat((xt, y), dim=1), y, t / self.start_step, labels)
            et = (xt - x0_pred * alpha_t.sqrt()) / (1 - alpha_t).sqrt()
            mat_norm = ((y_0 - H(x0_pred)).reshape(n, -1) ** 2).sum(dim=1).sqrt().detach()
            mat = ((y_0 - H(x0_pred)).reshape(n, -1) ** 2).sum()

            grad_term = torch.autograd.grad(mat, xt, retain_graph=True)[0]

            if self.original:
                coeff = self.grad_term_weight / mat_norm.reshape(-1, 1, 1, 1)
            else:
                coeff = alpha_s.sqrt() * alpha_t.sqrt()  # - c2 * alpha_t.sqrt() / (1 - alpha_t).sqrt()

            grad_term = grad_term.detach()
            xs = alpha_s.sqrt() * x0_pred.detach() + c1 * torch.randn_like(xt) + c2 * et.detach() - grad_term * coeff
            xt_s.append(xs.detach().cpu())
            x0_s.append(x0_pred.detach().cpu())
            xt = xs

        # return list(reversed(xt_s)), list(reversed(x0_s))
        return xt.detach()

    def initialize(self, x, y):
        # y_0 = kwargs['y_0']
        # H = self.H
        ts = list(range(self.end_step, self.start_step, (self.start_step - self.end_step) // self.num_steps))
        deg = self.cfg.algo.deg
        n = x.size(0)
        x_0 = self.H_inv(y).view(*x.size()).detach()
        ti = ts[-1]
        t = torch.ones(n).to(x.device).long() * ti
        alpha_t = self.diffusion.alpha(t).view(-1, 1, 1, 1)
        return alpha_t.sqrt() * x_0 + (1 - alpha_t).sqrt() * torch.randn_like(x_0)
