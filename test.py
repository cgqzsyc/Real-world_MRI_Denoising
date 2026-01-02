import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import yaml
from torch.utils.data import Dataset, DataLoader
from DicomLoader import MriValidConDataset
from algorithms import REDdiff, DDIM
from UNetSeries import SongUNet, NBNet
from PIL import Image
from torchmetrics.functional import structural_similarity_index_measure
from collections import deque  # added
from torch.nn.functional import cosine_similarity, l1_loss


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

def unet_forward(model, inputs, targets, device, normalize_output=False):
    outputs = model(inputs)
    if normalize_output:
        outputs = normalized_eval(outputs, targets)
    return outputs

def eval(model_name, config_path):
    map_organ = {'BRAIN': 1, 'HEAD': 1, 'KNEE': 2, 'CSPINE': 3, 'LSPINE': 4, 'SPINE': 5,
                 'TSPINE': 6, 'CAROTID': 7, 'SHOULDER': 8, 'UTERUS': 9, 'lp': 10,
                 'EMPTY': 0}
    cnt_organ, psnr_organ, ssim_organ = torch.zeros(11), torch.zeros(11), torch.zeros(11)
    global config
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')

    device = torch.device(config['device'])
    config_dataset = config['test_dataset']
    dataset_test = MriValidConDataset(**config['test_dataset']['args'])
    test_loader = DataLoader(dataset_test, batch_size=config_dataset['batch_size'], shuffle=False, num_workers=1)
    torch.cuda.empty_cache()

    model = SongUNet(**config['denoiser']['args'])
    config_reddiff = config['reddiff']['args']
    reddiff = REDdiff(nn_model=model, betas=config_reddiff['betas'], n_T=config_reddiff['n_T'],
                      device=device, drop_prob=config_reddiff['drop_prob'])
    model_path = 'xxx'
    ckpt = torch.load(model_path, map_location=device)
    ckpt = ckpt['model_state_dict']
    reddiff.load_state_dict(ckpt)
    reddiff = reddiff.to(device)

    model_h = NBNet(**config['baseline']['NBNet'])
    model_path = 'xxx'
    ckpt = torch.load(model_path, map_location=device)
    ckpt = ckpt['model_state_dict']
    model_h.load_state_dict(ckpt)
    model_h = model_h.to(device)

    model_f = NBNet(**config['baseline']['NBNet'])
    model_path = 'xxx'
    ckpt = torch.load(model_path, map_location=device)
    ckpt = ckpt['model_state_dict']
    model_f.load_state_dict(ckpt)
    model_f = model_f.to(device)

    loss_lpips = lpips.LPIPS(net='vgg')
    loss_lpips.to(device)
    psnr_tot, ssim_tot, lpips_tot, cnt_test = 0, 0, 0, 0

    with torch.no_grad():
        for xi, labels in test_loader:
            x = xi[:, 0, :, :]
            x_GT = xi[:, 1, :, :]
            x = x.unsqueeze(1).to(device).detach()
            x_GT = x_GT.unsqueeze(1).to(device).detach()
            labels = labels.to(device).detach()
            x_denoised = torch.empty((0,)).to(device)

            cnt_test += x.shape[0]
            mse = torch.mean((x_GT - x) ** 2, dim=(2, 3))
            psnr = 10.0 * torch.log10(1 / (mse + 1e-10))
            ssim = structural_similarity_index_measure(x_GT, x, reduction=None)
            lpips_cur = loss_lpips.forward(x_GT, x_denoised)
            lpips_tot += torch.sum(lpips_cur).item()
            print(cnt_test, "before:", torch.sum(psnr).item(), torch.sum(ssim).item())


            x_denoised = reddiff.sample(x, x_GT, x, labels, model_h, model_f, cnt_test)

            mse = torch.mean((x_GT - x_denoised) ** 2, dim=(2, 3))
            psnr = 10.0 * torch.log10(1 / (mse + 1e-10))
            ssim = structural_similarity_index_measure(x_GT, x_denoised, reduction=None)
            lpips_cur = loss_lpips.forward(x_GT, x_denoised)
            psnr_tot += torch.sum(psnr).item()
            ssim_tot += torch.sum(ssim).item()
            psnr_organ[int(labels[0][1].cpu())] += torch.sum(psnr).item()
            ssim_organ[int(labels[0][1].cpu())] += torch.sum(ssim).item()
            cnt_organ[int(labels[0][1].cpu())] += 1
            lpips_tot += torch.sum(lpips_cur).item()

            print(cnt_test, "after:", torch.sum(psnr).item(), torch.sum(ssim).item())

    psnr_tot /= cnt_test
    ssim_tot /= cnt_test
    lpips_tot /= cnt_test

    print(model_name, f"PSNR={psnr_tot:.4f}, SSIM={ssim_tot:.4f}, LPIPS={lpips_tot:.4f}")

    for organ_name, value in map_organ.items():
        print(organ_name, cnt_organ[value], psnr_organ[value] / cnt_organ[value], ssim_organ[value] / cnt_organ[value])

if __name__=="__main__":
    eval(model_name='FAST', config_path='./configs/uiirecon.yaml')