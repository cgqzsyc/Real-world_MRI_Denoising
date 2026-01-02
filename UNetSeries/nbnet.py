import torch
import torch.nn as nn

from pix2pix import pix2pixGAN
import torch.nn.functional as F


def conv3x3(in_chn, out_chn, bias=True):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=3, stride=1, padding=1, bias=bias)
    return layer


def conv_down(in_chn, out_chn, bias=False):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=4, stride=2, padding=1, bias=bias)
    return layer


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, downsample, relu_slope):
        super(UNetConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(relu_slope),
            nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(relu_slope))

        self.downsample = downsample
        if downsample:
            self.downsample = conv_down(out_size, out_size, bias=False)

        self.shortcut = nn.Conv2d(in_size, out_size, kernel_size=1, bias=True)

    def forward(self, x):
        out = self.block(x)
        sc = self.shortcut(x)
        out = out + sc
        if self.downsample:
            out_down = self.downsample(out)
            return out_down, out
        else:
            return out


class Subspace(nn.Module):
    def __init__(self, in_size, out_size):
        super(Subspace, self).__init__()
        self.blocks = nn.ModuleList()
        self.blocks.append(UNetConvBlock(in_size, out_size, False, 0.2))
        self.shortcut = nn.Conv2d(in_size, out_size, kernel_size=1, bias=True)

    def forward(self, x):
        sc = self.shortcut(x)
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
        return x + sc


class skip_blocks(nn.Module):
    def __init__(self, in_size, out_size, repeat_num=1):
        super(skip_blocks, self).__init__()
        self.blocks = nn.ModuleList()
        self.re_num = repeat_num
        mid_c = 128
        self.blocks.append(UNetConvBlock(in_size, mid_c, False, 0.2))
        for i in range(self.re_num - 2):
            self.blocks.append(UNetConvBlock(mid_c, mid_c, False, 0.2))
        self.blocks.append(UNetConvBlock(mid_c, out_size, False, 0.2))
        self.shortcut = nn.Conv2d(in_size, out_size, kernel_size=1, bias=True)

    def forward(self, x):
        sc = self.shortcut(x)
        for m in self.blocks:
            x = m(x)
        return x + sc


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope, subnet_repeat_num, subspace_dim=16):
        super(UNetUpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, bias=True)
        self.conv_block = UNetConvBlock(in_size, out_size, False, relu_slope)
        self.num_subspace = subspace_dim
        # print(self.num_subspace, subnet_repeat_num)
        self.subnet = Subspace(in_size, self.num_subspace)
        self.skip_m = skip_blocks(out_size, out_size, subnet_repeat_num)

    def forward(self, x, bridge):
        up = self.up(x)
        bridge = self.skip_m(bridge)
        out = torch.concat([up, bridge], 1)
        if self.subnet:
            b_, c_, h_, w_ = bridge.shape
            sub = self.subnet(out)
            V_t = sub.reshape(b_, self.num_subspace, h_ * w_)
            V_t_abs = torch.abs(V_t)
            V_t = V_t / (1e-6 + torch.sum(V_t_abs, dim=2, keepdim=True))
            V = V_t.transpose(2, 1)
            mat = torch.bmm(V_t, V)
            # print("mat:", mat.shape, V_t.shape, V.shape)
            mat_inv = torch.empty((0,)).to(torch.device("cuda"))
            for i in range(b_):
                mat_inv_i = torch.inverse(mat[i]).unsqueeze(0)
                mat_inv = torch.concat((mat_inv, mat_inv_i), dim=0)
            # mat_inv = torch.inverse(mat)
            project_mat = torch.bmm(mat_inv, V_t)
            # print("mat_inv:", project_mat.shape, mat_inv.shape, V_t.shape)
            bridge_ = bridge.reshape(b_, c_, h_ * w_)
            project_feature = torch.bmm(project_mat, bridge_.transpose(2, 1))
            bridge = torch.bmm(V, project_feature)
            # print("bridge:", bridge.shape)
            bridge = bridge.transpose(2, 1).reshape(b_, c_, h_, w_)
            # print("bridge:", bridge.shape)
            out = torch.concat([up, bridge], 1)
        out = self.conv_block(out)
        return out


class ConditionEmbedFC(nn.Module):  # 所有条件（position、image？）的嵌入：MLP+GeLU+MLP
    def __init__(self, input_dim, emb_dim):
        super(ConditionEmbedFC, self).__init__()
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class NBNet(nn.Module):
    def __init__(self, in_chn, wf=32, depth=5, relu_slope=0.2, subspace_dim=16):
        super(NBNet, self).__init__()
        self.depth = depth
        self.wf = wf
        self.down_path = nn.ModuleList()
        prev_channels = self.get_input_chn(in_chn)
        for i in range(depth):
            downsample = True if (i + 1) < depth else False
            self.down_path.append(UNetConvBlock(prev_channels, (2 ** i) * wf, downsample, relu_slope))
            prev_channels = (2 ** i) * wf

        # self.ema = EMAU(prev_channels, prev_channels//8)
        self.con_emb = ConditionEmbedFC(3, (2 ** (depth - 1)) * wf)

        self.up_path = nn.ModuleList()
        subnet_repeat_num = 1
        for i in reversed(range(depth - 1)):
            self.up_path.append(UNetUpBlock(prev_channels, (2 ** i) * wf, relu_slope, subnet_repeat_num, subspace_dim))
            prev_channels = (2 ** i) * wf
            subnet_repeat_num += 1

        self.last = conv3x3(prev_channels, in_chn, bias=True)
        # self._initialize()

    def forward(self, x1, labels):
        blocks = []
        for i, down in enumerate(self.down_path):
            # print(x1.shape)
            if (i + 1) < self.depth:
                x1, x1_up = down(x1)
                blocks.append(x1_up)
            else:
                x1 = down(x1)

        # print(x1.shape)
        # x1 = self.ema(x1)
        ### ! add conditions
        # print(labels.shape, labels[:, :4])
        # cemb = self.con_emb(labels[:, :4]).view(-1, (2 ** (self.depth - 1)) * self.wf, 1, 1)
        # ones_mat = torch.ones_like(x1)
        # cemb = cemb * ones_mat
        # x1 = x1 + cemb
        ###

        for i, up in enumerate(self.up_path):
            # print(x1.shape, blocks[-i-1].shape)
            x1 = up(x1, blocks[-i - 1])

        pred = self.last(x1)
        return pred

    def get_input_chn(self, in_chn):
        return in_chn

    def _initialize(self):
        gain = nn.init.calculate_gain('leaky_relu', 0.20)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # print("weight")
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    # print("bias")
                    nn.init.zeros_(m.bias)


class NBNetSeries_GAN(nn.Module):
    def __init__(self, config_d, lrate, in_chn=1, wf=32, depth=5, relu_slope=0.2, subspace_dim=16):
        super(NBNetSeries, self).__init__()
        self.device = torch.device("cuda")
        self.NBNet_finv = NBNet(in_chn, wf, depth, relu_slope, subspace_dim)
        self.NBNet_f = NBNet(in_chn, wf, depth, relu_slope, subspace_dim)
        self.pixelGAN_finv = pix2pixGAN(net_d_config=config_d)
        self.optimizer_g_finv = torch.optim.Adam(self.NBNet_finv.parameters(), lr=lrate)
        self.optimizer_d_finv = torch.optim.Adam(self.pixelGAN_finv.parameters(), lr=lrate)
        self.pixelGAN_f = pix2pixGAN(net_d_config=config_d)
        self.optimizer_g_f = torch.optim.Adam(self.NBNet_f.parameters(), lr=lrate)
        self.optimizer_d_f = torch.optim.Adam(self.pixelGAN_f.parameters(), lr=lrate)
        # self.l1_loss = nn.L1Loss(reduction='mean')

    def loss(self, x_d, x_a, x_GT, x, lambda_f):  # x_d(enoised), x_a(ddnoised)
        self.optimizer_g_finv.zero_grad()  # Generator与Discriminator分开更新，不能用同一个梯度优化器！
        self.optimizer_d_finv.zero_grad()
        self.optimizer_g_f.zero_grad()  # Generator与Discriminator分开更新，不能用同一个梯度优化器！
        self.optimizer_d_f.zero_grad()
        loss_g_finv, loss_l1_finv = self.pixelGAN_finv.loss_g(x_GT, x_d)
        loss_d_finv = self.pixelGAN_finv.loss_d(x_GT, x_d)
        loss_g_finv.backward(retain_graph=True)
        loss_d_finv.backward(retain_graph=True)
        loss_g_f, loss_l1_f = self.pixelGAN_f.loss_g(x, x_a)
        loss_d_f = self.pixelGAN_f.loss_d(x, x_a)
        for param in self.pixelGAN_finv.parameters():  # 此阶段，finv的discrimonator无需更新
            param.requires_grad_(False)
        loss_g_f.backward()
        loss_d_f.backward()
        self.optimizer_g_finv.step()  # Generator与Discriminator分开更新，不能用同一个梯度优化器！
        self.optimizer_d_finv.step()
        self.optimizer_g_f.step()  # Generator与Discriminator分开更新，不能用同一个梯度优化器！
        self.optimizer_d_f.step()
        return loss_g_finv, loss_d_finv, loss_g_f, loss_d_f, loss_l1_finv, loss_l1_f
        # loss_finv, loss_f = self.l1_loss(x_d, x_GT), self.l1_loss(x_a, x)
        # loss = loss_finv + lambda_f * loss_f
        # return loss, loss_finv, loss_f

    def forward(self, x, labels):
        x_noise_finv = self.NBNet_finv(x, labels)
        x_denoised_finv = x - x_noise_finv
        x_noise_f = self.NBNet_f(x_denoised_finv, labels)
        x_addnoised_f = x_denoised_finv + x_noise_f
        return x_denoised_finv, x_addnoised_f


class NBNetSeries(nn.Module):
    def __init__(self, in_chn, wf=32, depth=5, relu_slope=0.2, subspace_dim=16):
        super(NBNetSeries, self).__init__()
        self.NBNet_forward = NBNet(in_chn, wf, depth, relu_slope, subspace_dim)
        self.NBNet_backward = NBNet(in_chn, wf, depth, relu_slope, subspace_dim)

    def forward(self, x, labels):
        x_noise_f = self.NBNet_forward(x, labels)
        x_denoised = x - x_noise_f
        x_noise_b = self.NBNet_backward(x_denoised, labels)
        x_addnoised = x_denoised + x_noise_b
        return x_denoised, x_addnoised

    def denoise(self, x,labels):
        x_noise_f = self.NBNet_forward(x, labels)
        x_denoised = x - x_noise_f
        return x_denoised

    def add_noise(self, x, labels):
        x_noise_b = self.NBNet_backward(x, labels)
        x_addnoised = x + x_noise_b
        return x_addnoised