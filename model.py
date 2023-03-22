import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm
import datetime
from sklearn import mixture
from sklearn.cluster import AgglomerativeClustering

class creditDataloader(Dataset):
    def __init__(self, csv):
        self.data = csv
        self.data = self.data.to_numpy()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class Autoencoder(nn.Module):
    def __init__(self, numLayers, encoders=False):

        super().__init__()
        self.layers = nn.ModuleList()
        self.train_loss = []

        if encoders:
            for i in range(len(numLayers) - 2):
                self.layers.append(nn.Linear(numLayers[i], numLayers[i+1]))
                self.layers.append(nn.ReLU())
            self.layers.append(nn.Linear(numLayers[-2], numLayers[-1]))
        else:
            for i in range(len(numLayers) - 2):
                self.layers.append(nn.Linear(numLayers[i], numLayers[i+1]))
                self.layers.append(nn.ReLU())
            self.layers.append(nn.Linear(numLayers[-2], numLayers[-1]))
            for i in range(len(numLayers) - 1, 1, -1):
                self.layers.append(nn.Linear(numLayers[i], numLayers[i-1]))
                self.layers.append(nn.ReLU())
            self.layers.append(nn.Linear(numLayers[1], numLayers[0]))

    def forward(self, x):
        y = x
        for i in range(len(self.layers)):
            y = self.layers[i](y)
        return y


    def train(self, optimizer, loss_fn, train_loader, n_epochs, device):

        for epoch in tqdm(range(n_epochs)):
            loss_train = 0.0
            for data in train_loader:
                data = data.to(device=device).view(data.shape[0], -1)
                data = torch.tensor(data, dtype=torch.float32)
                outputs = self.forward(data) 
                loss = loss_fn(outputs, data)
                loss_train += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if (epoch + 1) % 10 == 0:
                print('{} Epoch {}, Training loss {}'.format(datetime.datetime.now(), epoch + 1, loss_train / len(train_loader)))
            # wandb.log({"AE train loss": loss_train / len(train_loader)})
            self.train_loss.append(loss_train / len(train_loader))

class Clsutering():

    def __init__(self, method, data, k):
        self.method = method
        self.data = data
        self.k = k
    
    def fit_predict(self):
        if self.method == 'HC':
            HC = AgglomerativeClustering(n_clusters=self.k, linkage='ward').fit(self.data)
            pred = HC.labels_
        elif self.method == 'GMM':
            GMM = mixture.GaussianMixture(
               covariance_type="full", n_components=self.k, random_state=2021).fit(self.data)
            pred_prob = GMM.predict_proba(self.data)
            pred = pred_prob.argmax(1)
        
        return pred
