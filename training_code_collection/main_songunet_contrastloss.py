import os
import sys
import torch
import numpy as np
import pickle
import pydicom
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from DicomLoader import MriTrainDataset, MriValidDataset
from algorithms import REDdiff
from UNetSeries import SongUNet
from PIL import Image
from torchmetrics.functional import structural_similarity_index_measure
from collections import deque  # added
from torch.nn.functional import cosine_similarity

def train():
    # Basic parameter Setting
    device = torch.device("cuda")

    torch.cuda.empty_cache()

    n_epoch = 350
    batch_size = 2
    image_size = (256, 256)

    # reddiff hyperparameters
    n_T = 1000
    n_feat = 128  #  bottleneck vector's dim num. 128 ok, 256 better (but slower)
    lrate = 2e-4  #  ! learning rate: SOngUNet: 1e-4; UNetSeries: 2e-4
    # added
    K = 1024
    tau = 16
    weight_contrast = 0.0005


    dataset_train = MriTrainDataset()
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=1)
    # dataset_valid = MriValidDataset()
    # valid_loader = DataLoader(dataset_valid, batch_size=1, shuffle=False, num_workers=2)
    with open(f'D:/uii_dataset_preprocessed/data_final/valid_data.pkl', 'rb') as f:
        data_valid = pickle.load(f)
    data_target = np.load("D:/uii_dataset_preprocessed/data_final/valid_target.npy")
    data_target = torch.from_numpy(data_target)
    cnt_valid = 5353

    cnt_valid, psnr_tot, ssim_tot = 0, 0, 0
    for i in range(len(data_valid)):
        cnt_valid += 1
        x_val_fast, x_val_GT, labels = torch.from_numpy(data_valid[i][0]), torch.from_numpy(data_valid[i][1]), data_target[i]

        # padding
        padding_size = x_val_fast.shape[0]
        if padding_size % 16 > 0:
            padding_size = padding_size - padding_size % 16 + 16
        x_val_fast = F.pad(x_val_fast, (
            max(0, (padding_size - x_val_fast.shape[1]) // 2), max(0, (padding_size - x_val_fast.shape[1]) // 2),
            max(0, (padding_size - x_val_fast.shape[0]) // 2), max(0, (padding_size - x_val_fast.shape[0]) // 2)))
        x_val_GT = F.pad(x_val_GT, (
            max(0, (padding_size - x_val_GT.shape[1]) // 2), max(0, (padding_size - x_val_GT.shape[1]) // 2),
            max(0, (padding_size - x_val_GT.shape[0]) // 2), max(0, (padding_size - x_val_GT.shape[0]) // 2)))


        x_val_fast, x_val_GT = x_val_fast.unsqueeze(0), x_val_GT.unsqueeze(0)
        x_val_fast, x_val_GT, labels = x_val_fast.to(device), x_val_GT.to(device), labels.to(device)
        mse = torch.mean((x_val_GT.unsqueeze(1) - x_val_fast.unsqueeze(1)) ** 2, dim=(2, 3))
        psnr = 10.0 * torch.log10(1 / (mse + 1e-10))
        ssim = structural_similarity_index_measure(x_val_GT.unsqueeze(1), x_val_fast.unsqueeze(1), reduction=None)
        psnr_tot += psnr
        ssim_tot += ssim

    print(cnt_valid, psnr_tot / float(cnt_valid), ssim_tot / float(cnt_valid))


    # nn_model = ContextUnet(in_channels=1, n_feat=n_feat)
    nn_model = SongUNet(img_resolution = image_size[0], in_channels=2, out_channels=1, label_dim=3) # !
    reddiff = REDdiff(nn_model=nn_model, betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.1)
    # reddiff.load_state_dict(torch.load("/home_data/home/v-shaoych/UiiReconstruction/checkpoints/epoch50.pth", map_location=device))  # 加载预训练权重
    reddiff.to(device)
    optim = torch.optim.Adam(reddiff.parameters(), lr=lrate)

    for ep in range(n_epoch):
        print(f'epoch {ep}')
        epochs = ep + 1  ###
        reddiff.train()

        #  linear learning rate decay
        optim.param_groups[0]['lr'] = lrate * (1 - ep / n_epoch)

        pbar = tqdm(train_loader)
        loss_ema = None
        cnt_training = 0
        queue = deque()  # added
        for xi, labels in pbar:
            optim.zero_grad()
            x = xi[:, 0, :, :]
            x_GT = xi[:, 1, :, :]
            x = x.to(device)
            x_GT = x_GT.to(device)
            labels = labels.to(device)

            # added contrast learning
            loss_main, q = reddiff(x, x_GT, labels)
            loss_contrast = torch.zeros(batch_size).to(device)
            for i in range(batch_size):
                q_i = q[i].view(-1)
                l_pos, l_neg = torch.tensor([0.01, ]).to(device), torch.tensor([0.01, ]).to(device)
                for k in queue:
                    val = cosine_similarity(q_i.unsqueeze(0), k[0].unsqueeze(0))
                    if labels[i][1] == k[1]:  # positive
                        l_pos += torch.exp(val)
                    else:  # negative
                        l_neg += torch.exp(val)
                loss_contrast[i] = -torch.log(l_pos / l_neg)
                queue.append((q[i].view(-1).clone().detach(), labels[i][1]))  # enqueue
            loss = loss_main + weight_contrast * loss_contrast.mean()
            loss.backward()
            print(len(queue), loss, loss_contrast.mean())
            if len(queue) > K: # dequeue
                for i in range(batch_size):
                    queue.popleft()

            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema += loss.item()
            cnt_training += 1
            loss_now = loss_ema / cnt_training
            pbar.set_description(f"loss: {loss_now:.4f}")
            # print(f"loss: {loss_ema:.4f}")
            optim.step()
        loss_ema = loss_ema / cnt_training
        print(cnt_training)
        print(f"loss: {loss_ema:.4f}")
        if epochs % 1 == 0 and epochs > 0:
            torch.save(reddiff.state_dict(), f"./checkpoints/epoch{epochs}.pth")

        reddiff.eval()
        if epochs % 2 == 0 and epochs > 0:
            for i in range(len(data_valid)):
                if i % 500 == 0:
                    print(i)
                x_val_fast, x_val_GT, labels = torch.from_numpy(data_valid[i][0]), torch.from_numpy(data_valid[i][1]), data_target[i]

                # padding
                padding_size = x_val_fast.shape[0]
                if padding_size % 16 > 0:
                    padding_size = padding_size - padding_size % 16 + 16
                x_val_fast = F.pad(x_val_fast, (
                    max(0, (padding_size - x_val_fast.shape[1]) // 2), max(0, (padding_size - x_val_fast.shape[1]) // 2),
                    max(0, (padding_size - x_val_fast.shape[0]) // 2), max(0, (padding_size - x_val_fast.shape[0]) // 2)))
                x_val_GT = F.pad(x_val_GT, (
                    max(0, (padding_size - x_val_GT.shape[1]) // 2), max(0, (padding_size - x_val_GT.shape[1]) // 2),
                    max(0, (padding_size - x_val_GT.shape[0]) // 2), max(0, (padding_size - x_val_GT.shape[0]) // 2)))

                x_val_fast, x_val_GT, labels = x_val_fast.unsqueeze(0), x_val_GT.unsqueeze(0), labels.unsqueeze(0)
                x_val_fast, x_val_GT, labels = x_val_fast.to(device), x_val_GT.to(device), labels.to(device)
                psnr, ssim = reddiff.sample(x_val_fast, x_val_GT, x_val_fast, labels, epochs)
                psnr_tot += psnr
                ssim_tot += ssim
                if (i + 1) % 10 == 0:
                    print(i + 1, psnr_tot / float(i + 1), ssim_tot / float(i + 1))

            print("epoch", epochs, ": ", psnr_tot / float(cnt_valid), ssim_tot / float(cnt_valid))

if __name__=="__main__":
    train()