import os
import sys
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from DicomLoader import MriTrainConDataset
from UNetSeries import NBNetSeries



def train(config_path):
    # Basic parameter Setting
    global config
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')

    device = torch.device(config['device'])
    torch.cuda.empty_cache()

    lambda_f = config['lambda_f']

    config_dataset = config['train_dataset']
    dataset_train = MriTrainConDataset(**config['train_dataset']['args'])
    train_loader = DataLoader(dataset_train, batch_size=config_dataset['batch_size'], shuffle=True, num_workers=1)
    torch.cuda.empty_cache()

    model = NBNetSeries(config_d=config['degradation']['args_netd'], lrate=config['optimizer']['args']['lr'])
    model = model.to(device)
    epoch_start = config['epoch_start']
    epoch_max = config['epoch_max']
    epoch_val = config['epoch_val']

    for epoch in range(epoch_start, epoch_max + 1):
        cnt_training = 0
        print(f'epoch {epoch}')
        model.train()

        pbar = tqdm(train_loader)
        loss_g_finv_ema, loss_d_finv_ema, loss_g_f_ema, loss_d_f_ema, loss_l1_finv_ema, loss_l1_f_ema = 0, 0, 0, 0, 0, 0

        for xi, labels in pbar:
            x = xi[:, 0, :, :]
            x_GT = xi[:, 1, :, :]
            x = x.unsqueeze(1).to(device)
            x_GT = x_GT.unsqueeze(1).to(device)
            labels = labels.to(device)
            x_denoised, x_addnoised = model(x, labels)
            loss_g_finv, loss_d_finv, loss_g_f, loss_d_f, loss_l1_finv, loss_l1_f = model.loss(x_denoised, x_addnoised, x_GT, x, lambda_f)
            loss_g_finv_ema += loss_g_finv.item()
            loss_d_finv_ema += loss_d_finv.item()
            loss_g_f_ema += loss_g_f.item()
            loss_d_f_ema += loss_d_f.item()
            loss_l1_finv_ema += loss_l1_finv.item()
            loss_l1_f_ema += loss_l1_f.item()
            cnt_training += 1
            loss_g_finv_cur, loss_d_finv_cur, loss_g_f_cur, loss_d_f_cur, loss_l1_finv_cur, loss_l1_f_cur = (loss_g_finv_ema / cnt_training, loss_d_finv_ema / cnt_training,
                                                                            loss_g_f_ema / cnt_training, loss_d_f_ema / cnt_training, loss_l1_finv_ema / cnt_training, loss_l1_f_ema / cnt_training)
            pbar.set_description(f"loss_g_finv: {loss_g_finv_cur:.6f} loss_d_finv: {loss_d_finv_cur:.6f} loss_g_f: {loss_g_f_cur:.6f} loss_d_f: {loss_d_f_cur:.6f} loss_l1_finv: {loss_l1_finv_cur:.6f} loss_l1_f: {loss_l1_f_cur:.6f}")
            # print(f"loss: {loss_ema:.6f}")

        loss_g_finv_ema, loss_d_finv_ema, loss_g_f_ema, loss_d_f_ema, loss_l1_finv_ema, loss_l1_f_ema = (loss_g_finv_ema / cnt_training, loss_d_finv_ema / cnt_training,
                                                                        loss_g_f_ema / cnt_training, loss_d_f_ema / cnt_training, loss_l1_finv_ema / cnt_training, loss_l1_f_ema / cnt_training)
        print(f"loss_g_finv: {loss_g_finv_ema:.6f} loss_d_finv: {loss_d_finv_ema:.6f} loss_g_f: {loss_g_f_ema:.6f} loss_d_f: {loss_d_f_ema:.6f} loss_l1_finv: {loss_l1_finv_ema:.6f} loss_l1_f: {loss_l1_f_ema:.6f}")

        if epoch % epoch_val == 0 and epoch > 0:
            save_path_root = config['save_path']
            os.makedirs(save_path_root + '/checkpoints/nbnetseries', exist_ok=True)
            checkpoint_e = {
                'model': model.state_dict(),
                'epoch': epoch
            }
            torch.save(checkpoint_e, save_path_root + f"/checkpoints/nbnetseries/nbnetseries_1020_ep{epoch}.pth")

if __name__=="__main__":
    train(config_path='./configs/uiirecon.yaml')