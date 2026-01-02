import torch
import torch.nn as nn
import numpy as np
from torchmetrics.functional import structural_similarity_index_measure

def norm(x):
    mean = x.view(x.shape[0], 1, 1, -1).mean(-1, keepdim=True)
    std = x.view(x.shape[0], 1, 1, -1).std(-1, keepdim=True)
    x = (x - mean) / std
    return x

def normalized_eval(output, target):
    mt = target.mean(dim=(-2, -1), keepdim=True)
    st = target.std(dim=(-2, -1), keepdim=True)
    mo = output.mean(dim=(-2, -1), keepdim=True)
    so = output.std(dim=(-2, -1), keepdim=True)
    return (output - mo) / so * st + mt

class REDdiff(nn.Module):
    def __init__(self, nn_model, betas, n_T, device, drop_prob):
        super(REDdiff, self).__init__()
        self.device = device
        self.nn_model = nn_model.to(device)
        self.awd = True
        self.cond_awd = False
        self.grad_term_weight = 0.25  # denoising snr λ=0.25 !
        self.obs_weight = 1  # observe_weight !
        self.eta = 0  # η=0
        self.lr = 0.1  # 0.1 !
        self.denoise_term_weight = "linear"
        self.sigma_x0 = 0.0 # 0.0
        self.start_step = 1000
        self.end_step = 0
        self.num_steps = 200
        self.beta_start = 1e-4
        self.beta_end = 2e-2
        self.loss_mse = nn.MSELoss() # min ||ε_predict-ε_~N(0,1)||^2
        self.betas = np.linspace(self.beta_start, self.beta_end, self.start_step, dtype=np.float64)
        self.betas = torch.from_numpy(self.betas)
        self.betas = self.betas.to(self.device)
        self.betas = torch.cat([torch.zeros(1).to(self.device), self.betas], dim=0).cuda().float()
        self.alphas = (1 - self.betas).cumprod(dim=0).to(self.device).float()

    def forward(self, x_fast, x_GT, labels, prompt_embeds): # forward process of diffusion
        n = x_fast.shape[0]
        c = x_fast
        # this method is used in training, so samples t and noise randomly
        ti = torch.randint(1, self.start_step, (x_fast.shape[0],)).to(self.device)  # t ~ Uniform(0, n_T)
        t = torch.ones(n).to(self.device).long() * ti[0].item()  # ti[i] = si[i-1]
        alpha_t = self.alphas.index_select(0, t).view(-1, 1, 1, 1)  # !! nxcxhxw
        noise_epsilon = torch.randn_like(x_GT)
        x_t = alpha_t.sqrt() * x_GT + (1 - alpha_t).sqrt() * noise_epsilon  # x_t in pseudo code
        return self.loss_mse(x_GT, self.nn_model(torch.cat((x_t, c), dim=1), c, t / self.start_step, labels[:, :6], prompt_embeds, None))  # !

    def sample(self, x, x_GT, y, labels, H, F, ep=0):
        x = x.to(self.device)
        # x = torch.randn_like(x) # !
        H.eval()
        F.eval()
        y_0 = y.to(self.device)  # observation y
        H_y0 = H(y_0)
        F_y0 = normalized_eval(F(y_0), y_0)
        sigma_y = 0
        n = x.shape[0]  # batch size
        # H = self.H
        ts = list(range(self.end_step, self.start_step, (self.start_step - self.end_step) // self.num_steps))
        ss = [-1] + list(ts[:-1])

        # optimizer
        dtype = torch.FloatTensor
        x0_noised = x.to(self.device)

        # print(x.shape)
        # mu_ini = normalized_eval(F(x), x) # !!

        mu_ini = H(x)
        mu = torch.tensor(mu_ini.clone().detach(), requires_grad=True)

        optimizer = torch.optim.Adam([mu], lr=self.lr, betas=(0.9, 0.99), weight_decay=0.0)  # original: 0.999
        # optimizer = torch.optim.SGD([mu], lr=1e6, momentum=0.9)  #momentum=0.9

        # for ti, si in zip(reversed(ts), reversed(ss)):
        for ti in reversed(ts):
            if ti > 200: # !
                continue
            optimizer.param_groups[0]['lr'] = self.lr * (ti / self.start_step) # !
            t = torch.ones(n).to(self.device).long() * ti  # ti[i] = si[i-1] !
            alpha_t = self.alphas.index_select(0, t).view(-1, 1, 1, 1)  # alpha_t : batch_sizex1x1x1
            sigma_x0 = self.sigma_x0  # 0.0001

            noise_epsilon = torch.randn_like(mu)

            x0_pred = mu + sigma_x0 * noise_epsilon
            mu_plot = torch.tensor(mu.clone().detach(), requires_grad=False)
            # print(x0_pred)
            xt = alpha_t.sqrt() * x0_pred + (1 - alpha_t).sqrt() * noise_epsilon  # 伪代码中的x_t

            t_is = torch.tensor([ti / self.start_step]).to(self.device)
            t_is = t_is.repeat(1)
            # x0_hat = self.nn_model(xt.unsqueeze(1), x0_noised, t_is, labels[:, :3]).squeeze().unsqueeze(0).detach()
            # x0_hat = self.nn_model(xt, x0_noised, t_is, labels).squeeze().unsqueeze(0).detach()
            x0_hat = self.nn_model(torch.cat((xt, x0_noised), dim=1), x0_noised, t_is, labels[:, :6]).detach()
            et = (xt - x0_hat * alpha_t.sqrt()) / (1 - alpha_t).sqrt()
            et = et.detach()
            e_obs = H_y0 - normalized_eval(F(x0_pred), x) # !!
            # e_obs = F_y0 - x0_pred  # !!
            loss_obs = (e_obs ** 2).mean() / 2  # observation loss: ||y-f(x)||^2/2
            # noise_difference = (et - noise_epsilon).detach()
            noise_difference = (x0_pred - x0_hat).detach()
            loss_noise = torch.mul(noise_difference, x0_pred).mean()  # reg loss: ||ε(_t,tx)-ε||*μ

            # print(ti, loss_obs, loss_noise)

            snr_inv = (1 - alpha_t[0]).sqrt() / alpha_t[0].sqrt()

            w_t = self.grad_term_weight * snr_inv
            v_t = self.obs_weight
            # if ti <= 200:  # !
            #     w_t += 5 * (200 - ti) / 200 # !

            loss = w_t * loss_noise + v_t * loss_obs
            # print(ti, w_t, loss)

            # adam step
            optimizer.zero_grad()  # initialize
            loss.backward(retain_graph=True)
            optimizer.step()
            torch.cuda.empty_cache()

        mu_final = torch.tensor(mu.clone().detach(), requires_grad=False)
        return mu_final
