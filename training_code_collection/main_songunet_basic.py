import os
import torch
import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader
from DicomLoader import MriTrainConDataset, MriTrainUnconDataset
from algorithms import REDdiff
from UNetSeries import SongUNet
from transformers import CLIPTextModel, CLIPTokenizer
from utils import GetReports

def train(config_path):
    # Basic parameter Setting
    global config
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')

    device = torch.device(config['device'])
    torch.cuda.empty_cache()

    config_dataset = config['train_dataset']
    dataset_train = MriTrainConDataset(**config['train_dataset']['args'])
    train_loader = DataLoader(dataset_train, batch_size=config_dataset['batch_size'], shuffle=True, num_workers=1)
    torch.cuda.empty_cache()

    # nn_model = ContextUnet(in_channels=1, n_feat=n_feat)
    nn_model = SongUNet(**config['denoiser']['args'])
    config_reddiff = config['reddiff']['args']
    reddiff = REDdiff(nn_model=nn_model, betas=config_reddiff['betas'], n_T=config_reddiff['n_T'],
                      device=device, drop_prob=config_reddiff['drop_prob'])

    # checkpth = torch.load("/public_bme/data/v-shaoych/UiiReconstruction/checkpoints/1009_newSongUNet_con/newSongUNet_con_1009_ep100.pth")
    # reddiff.load_state_dict(checkpth['model_state_dict'])
    reddiff.to(device)

    # reddiff hyperparameters
    lrate = config['optimizer']['args']['lr']  # learning rate
    optim = torch.optim.Adam(reddiff.parameters(), lr=lrate)
    # optim.load_state_dict(checkpth['optimizer_state_dict'])

    epoch_start = config['epoch_start']
    epoch_max = config['epoch_max']
    epoch_val = config['epoch_val']
    text_encoder = CLIPTextModel.from_pretrained("stabilityai/stable-diffusion-2-1-base",
                                                 subfolder="text_encoder")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    for epoch in range(epoch_start, epoch_max + 1):
        cnt_training = 0
        print(f'epoch {epoch}')
        reddiff.train()

        #  linear learning rate decay
        lrate_now = lrate
        if epoch <= 20:
            lrate_now = lrate * (1 - epoch / epoch_max)
        elif epoch <= 50:
            lrate_now = lrate * (0.7 - epoch / epoch_max)
        else:
            lrate_now = lrate * 0.2
        optim.param_groups[0]['lr'] = lrate_now

        pbar = tqdm(train_loader)
        loss_ema = None
        for xi, labels in pbar:
            optim.zero_grad()
            x = xi[:, 0, :, :]
            x_GT = xi[:, 1, :, :]
            x = x.unsqueeze(1).to(device)
            x_GT = x_GT.unsqueeze(1).to(device)
            labels = labels.to(device)
            # prompt_embeds = labels
            prompt_embeds = GetReports(labels, text_encoder, tokenizer).to(device)
            # prompt_embeds = text_encoder(reports.squeeze(1))
            # print(x.shape, x_GT.shape, prompt_embeds.shape)
            loss = reddiff(x, x_GT, labels, prompt_embeds)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema += loss.item()
            cnt_training += 1
            loss_now = loss_ema / cnt_training
            pbar.set_description(f"loss: {loss_now:.6f}")
            optim.step()
        loss_ema = loss_ema / cnt_training
        print(f"loss: {loss_ema:.6f}")
        if epoch % epoch_val == 0 and epoch > 0:
            save_path_root = config['save_path']
            os.makedirs(save_path_root + '/checkpoints/SongUNet/Conditioned/1228', exist_ok=True)
            checkpoint_e = {
                'model_state_dict': reddiff.state_dict(),
                'optimizer_state_dict': optim.state_dict()
            }
            torch.save(checkpoint_e,
                       save_path_root + f"/checkpoints/SongUNet/Conditioned/1228/SongUNet_con_1228_ep{epoch}.pth")


if __name__ == "__main__":
    train(config_path='./configs/uiirecon.yaml')