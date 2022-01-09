"""Modified Magnet for Applicability Domain
"""
import logging
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from adad.app_domain_base import AppDomainBase
from adad.utils import create_dir, open_json, time2str, to_json

logger = logging.getLogger(__name__)


class Reshape(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.n_channels = n_channels

    def forward(self, x):
        x = torch.reshape(x, (x.size(0), self.n_channels, -1))
        return x


class AutoEncoder(nn.Module):
    """This AutoEncoder is designed for MACCS fingerprints with 167 descriptors
    (RDKit default).
    """

    def __init__(self, n_outputs):
        super().__init__()

        self.encoder = nn.Sequential(
            Reshape(1),
            nn.Conv1d(1, 16, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten(1),
            nn.Linear(1312, 128),
            nn.ReLU(),
            nn.Linear(128, n_outputs),
        )

        self.decoder = nn.Sequential(
            nn.Linear(n_outputs, 128),
            nn.ReLU(),
            nn.Linear(128, 1312),
            nn.ReLU(),
            Reshape(32),
            nn.ConvTranspose1d(32, 16, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, kernel_size=3, stride=2),
            nn.Sigmoid(),
            nn.Flatten(1),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class Magnet(AppDomainBase):
    """Magnet for Applicability Domain 

    Parameters
    ----------
    AutoEncoder: torch.nn.Module
        The autoencoder class.
    n_inputs: int, default=167
        Number of attributs from the sample. (MACCS is 166 features, 
        RDKit generates 167)
    n_outputs: int, default=16
        Number of attributes from the output of the encoder.
    n_encoders: int, default=5
        Total number of autoencoders.
    batch_size: int. default=128
        Mini-batch size.
    max_epochs: int, default=300
        Number of training iterations.
    lr: float, default=1e-4
        Learning rate.
    device: str
        The device for training autoencoders.
    """

    def __init__(self,
                 AutoEncoder,
                 n_inputs=167,
                 n_outputs=16,
                 n_encoders=5,
                 batch_size=128,
                 max_epochs=200,
                 lr=1e-4,
                 device=torch.device('cuda')):

        self.AutoEncoder = AutoEncoder
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_encoders = n_encoders
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.lr = lr
        self.device = device

        self.loss_fn = nn.MSELoss()
        self.autoencoders = []
        for _ in range(n_encoders):
            model = AutoEncoder(n_outputs).to(device)
            autoencoder = {
                'model': model,
                'idx': np.random.permutation(n_inputs),
                'losses': np.zeros(max_epochs, dtype=float),
            }
            self.autoencoders.append(autoencoder)

    def fit(self, X, y=None):
        time_start = time.perf_counter()
        for autoencoder in tqdm(self.autoencoders):
            model = autoencoder['model']
            idx = autoencoder['idx']
            # Shuffle the indices
            XX = X[:, idx]
            dataset = TensorDataset(torch.from_numpy(XX).type(torch.float32))
            dataloader = DataLoader(
                dataset, batch_size=self.batch_size, shuffle=True)
            optimizer = torch.optim.AdamW(model.parameters(),
                                          lr=self.lr,
                                          weight_decay=1e-3)
            for e in range(self.max_epochs):
                loss_tr = self.__train(dataloader, model, optimizer)
                autoencoder['losses'][e] = loss_tr
                logger.info(f'Loss after {e:d} epochs: {loss_tr:.3f}')
        time_elapsed = time.perf_counter() - time_start
        logger.info(f'Total training time: {time2str(time_elapsed)}')
        return self

    def measure(self, X):
        """Encode X using all autoencoders"""
        outputs = []
        for autoencoder in self.autoencoders:
            model = autoencoder['model']
            idx = autoencoder['idx']
            XX = X[:, idx]
            output = self.encode(XX, model)
            outputs.append(output)
        return outputs

    def save(self, path):
        create_dir(path)
        data = {
            'n_inputs': self.n_inputs,
            'n_outputs': self.n_outputs,
            'n_encoders': self.n_encoders,
            'batch_size': self.batch_size,
            'max_epochs': self.max_epochs,
            'lr': self.lr,
            'device': str(self.device),
            'autoencoders': []
        }
        for i, ae in enumerate(self.autoencoders):
            data['autoencoders'].append({
                'idx': ae['idx'],
                'losses': ae['losses'],
            })
            model = ae['model']
            torch.save(model.state_dict(),
                       os.path.join(path, f'magnet_autoencoder{i}.torch'))
        to_json(data, os.path.join(path, 'magnet.json'))

    def load(self, path):
        data = open_json(os.path.join(path, 'magnet.json'))
        self.n_inputs = data['n_inputs']
        self.n_outputs = data['n_outputs']
        self.n_encoders = data['n_encoders']
        self.batch_size = data['batch_size']
        self.max_epochs = data['max_epochs']
        self.lr = data['lr']
        self.device = torch.device(data['device'])
        for i, ae in enumerate(data['autoencoders']):
            path_model = os.path.join(path, f'magnet_autoencoder{i}.torch')
            self.autoencoders[i]['model'].load_state_dict(
                torch.load(path_model, map_location=self.device))
            self.autoencoders[i]['idx'] = ae['idx']
            self.autoencoders[i]['losses'] = ae['losses']

    def encode(self, X, model):
        n = X.shape[0]
        dataset = TensorDataset(torch.from_numpy(X).type(torch.float32))
        dataloader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False)
        X_encoded = np.empty((n, self.n_outputs), dtype=np.float32)
        start = 0
        model.eval()
        with torch.no_grad():
            for [X] in dataloader:
                X = X.to(self.device)
                output = model.encoder(X).detach().cpu().numpy()
                end = start + X.size(0)
                X_encoded[start: end] = output
                start = end
        return X_encoded

    def decode(self, X, model):
        n = X.shape[0]
        dataset = TensorDataset(torch.from_numpy(X).type(torch.float32))
        dataloader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False)
        X_decode = np.empty((n, self.n_inputs), dtype=np.float32)
        start = 0
        model.eval()
        with torch.no_grad():
            for [X] in dataloader:
                X = X.to(self.device)
                output = model.decoder(X).detach().cpu().numpy()
                end = start + X.size(0)
                X_decode[start: end] = output
                start = end
        return X_decode

    def __train(self, dataloader, model, optimizer):
        n_batches = len(dataloader)
        loss_avg = 0

        model.train()
        for [X] in dataloader:
            X = X.to(self.device)

            optimizer.zero_grad()
            output = model(X)
            loss = self.loss_fn(output, X)
            loss.backward()
            optimizer.step()

            loss_avg += loss.item()

        loss_avg /= n_batches
        return loss_avg
