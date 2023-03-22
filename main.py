import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # 불필요한 success 메시지 끄기
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1" # cuda 에러 

import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import warnings; warnings.filterwarnings("always"); warnings.filterwarnings(action='ignore')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import umap.umap_ as umap

from model import creditDataloader, Autoencoder, Clsutering

parser = argparse.ArgumentParser(description='N2D Training')

# 실험 세팅
parser.add_argument('-v', '--ver', default=1, type=int,
                    help='version of preprocessed data {1, 2}')
parser.add_argument('-k', '--k_list', nargs='+', type=int,
                    help='list of numbers of clusters')

# 모델 학습
parser.add_argument('-e', '--epochs', default=10, type=int,
                    help='number of epochs')
parser.add_argument('-b', '--batch_size', default=64, type=int,
                    help='number of batch size for AutoEncoder')
parser.add_argument('-o', '--optimizer', default='adam', type=str,
                    help='adam or adamw')
parser.add_argument('-l', '--loss_function', default='mse', type=str,
                    help='mse or smoothl1')
parser.add_argument('-c', '--clustering', default='HC', type=str,
                    help='HC or GMM')                  

# 기타
parser.add_argument('-u', '--umap', default=3, type=int,
                    help='dimension reduction by umap')
parser.add_argument('-g', '--gpu_id', default=0, type=int,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()
args.state = {k: v for k, v in args._get_kwargs()}

# random seed
SEED = 2021 
np.random.seed(SEED)
torch.manual_seed(SEED)

# torch CUDA
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
assert device != 'cuda', "cuda not available!" # print("Device: ", device)

# 데이터 가져오기
data_dir = './data/'
CSVDATA = pd.read_csv(data_dir + 'data_clustering_sobi_ver{}.csv'.format(args.ver))

if __name__ == '__main__':

    for k in tqdm(args.k_list):

        print('\n----- LOOP: k={} -----\n'.format(k))

        print('\n----- LOOP: train AE -----\n')

        dataset = creditDataloader(CSVDATA).data
        model_AE = Autoencoder(numLayers=[12, 10, 10, k]).to(device)
        # optimizer
        if args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model_AE.parameters(), lr=0.001)
        elif args.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(model_AE.parameters(), lr=0.001)
        # loss
        if args.loss_function == 'mse':
            loss_fn = nn.MSELoss()
        elif args.loss_function == 'smoothl1':
            loss_fn = nn.SmoothL1Loss()
        # train
        trainLoader_AE = DataLoader(
            dataset=dataset, batch_size=args.batch_size, shuffle=True, num_workers=16)  # num_workers=16
        model_AE.train(
            optimizer=optimizer, loss_fn=loss_fn, train_loader=trainLoader_AE, n_epochs=args.epochs, device=device)
        encoder = nn.Sequential(
            *[model_AE.layers[i] for i in range(5)]).to(device)

        print('\n----- AutoEncoder, UMAP, Clustering -----n')

        dataloader = DataLoader(
            dataset=dataset, batch_size=len(dataset), shuffle=False, num_workers=16)

        for data in dataloader: # whole batch
            data = data.to(device).view(data.shape[0], -1)
            data = torch.tensor(data, dtype=torch.float32)

            latent_1 = encoder(data) 
            latent_1 = latent_1.cpu().data.numpy()
            latent_2 = umap.UMAP(random_state=2021, n_components=args.umap).fit_transform(latent_1) 
            pred = Clsutering(method=args.clustering, data=latent_2, k=k).fit_predict()

        # 결과물 저장: ver, k, umap, epoch, batch size 등등 나중에 wandb에 저장하도록 바꾸기 
        pd.DataFrame(pred).to_csv(data_dir + 'clustering_results/ver{0}/pred_cluster{1}_umap{2}_epochs{3}_batchsize{4}_clustering{5}.csv'.format(
            args.ver, k, args.umap, args.epochs, args.batch_size, args.clustering))

