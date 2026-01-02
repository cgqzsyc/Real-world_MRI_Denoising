import os
import sys
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from DicomLoader import MriTrainConDataset
from UNetSeries import NBNetSeries
from loss import msssim_loss


def train(config_path):
    # Basic parameter Setting
    global config
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')

    device = torch.device(config['device'])
    torch.cuda.empty_cache()

    n_epoch = 100
    alpha = torch.tensor([0.3, 0.3, 0.2, 0.2])
    alpha_loss_min = 0.3
    alpha_loss_max = 0.8

    config_dataset = config['train_dataset']
    dataset_train = MriTrainConDataset(**config['train_dataset']['args'])
    train_loader = DataLoader(dataset_train, batch_size=config_dataset['batch_size'], shuffle=True, num_workers=1)
    torch.cuda.empty_cache()

    model = NBNetSeries(1)
    model = model.to(device)
    # model.load_state_dict(torch.load("./checkpoints/epoch1.pth", map_location=device))
    lrate = config['optimizer']['args']['lr']  # learning rate
    optim = torch.optim.Adam(model.parameters(), lr=lrate)
    l1_loss = nn.L1Loss(reduction='mean')
    ms_ssim = msssim_loss()
    epoch_start = config['epoch_start']
    epoch_max = config['epoch_max']
    epoch_val = config['epoch_val']

    for epoch in range(epoch_start, epoch_max + 1):
        cnt_training = 0
        print(f'epoch {epoch}')
        model.train()
        #  linear learning rate decay
        optim.param_groups[0]['lr'] = lrate * (1 - epoch / epoch_max)

        pbar = tqdm(train_loader)
        loss_ema, loss_l1_ema, loss_ssim_ema = 0, torch.zeros(4), torch.zeros(4)

        for xi, labels in pbar:
            optim.zero_grad()
            x = xi[:, 0, :, :]
            x_GT = xi[:, 1, :, :]
            x = x.unsqueeze(1).to(device)
            x_GT = x_GT.unsqueeze(1).to(device)
            labels = labels.to(device)
            x_ans = torch.zeros(4, x.shape[0], x.shape[1], x.shape[2], x.shape[3]).to(device)
            # x_denoised_1, x_addnoised_1, x_denoised_2, x_addnoised_2 = model(x, labels)
            x_ans[0], x_ans[1], x_ans[2], x_ans[3] = model(x, labels)
            loss_l1, loss_ssim, loss_l1_cur, loss_ssim_cur = None, None, None, None
            for i in range(4):
                if i == 0:
                    loss_l1_cur = l1_loss(x_ans[i], x_GT)
                    loss_ssim_cur = ms_ssim(x_ans[i], x_GT)
                elif i % 2 == 0:
                    loss_l1_cur = l1_loss(x_ans[i], x_GT)
                    loss_ssim_cur = ms_ssim(x_ans[i], x_GT)
                else:
                    loss_l1_cur = l1_loss(x_ans[i], x)
                    loss_ssim_cur = ms_ssim(x_ans[i], x)
                if loss_l1 == None:
                    loss_l1 = alpha[i] * loss_l1_cur
                    loss_ssim = alpha[i] * loss_ssim_cur
                else:
                    loss_l1 += alpha[i] * loss_l1_cur
                    loss_ssim += alpha[i] * loss_ssim_cur
                loss_l1_ema[i] += loss_l1_cur.item()
                loss_ssim_ema[i] += loss_ssim_cur.item()
            alpha_loss = alpha_loss_max - (epoch / epoch_max) * (alpha_loss_max - alpha_loss_min)
            loss = alpha_loss * loss_l1 + (1 - alpha_loss) * loss_ssim
            loss.backward()
            loss_ema += loss.item()
            cnt_training += 1
            loss_now = loss_ema / cnt_training
            pbar.set_description(f"loss: {loss_now:.6f}")
            # print(f"loss: {loss_ema:.6f}")
            optim.step()
        loss_ema = loss_ema / cnt_training
        for i in range(4):
            loss_l1_ema[i] /= cnt_training
            loss_ssim_ema[i] /= cnt_training
        print(f"loss: {loss_ema:.6f}")
        for i in range(4):
            print(f"loss_l1{i}: {loss_l1_ema[i]:.6f}, loss_ssim_{i}: {loss_ssim_ema[i]:.6f}")
        if epoch % epoch_val == 0 and epoch > 0:
            save_path_root = config['save_path']
            os.makedirs(save_path_root + '/checkpoints/nbnet_2series', exist_ok=True)
            checkpoint_e = {
                'model': model.state_dict(),
                'optimizer': optim.state_dict(),
                'epoch': epoch
            }
            torch.save(checkpoint_e, save_path_root + f"/checkpoints/nbnet_2series/nbnet_2series_1020_ep{epoch}.pth")

if __name__=="__main__":
    train(config_path='./configs/uiirecon.yaml')