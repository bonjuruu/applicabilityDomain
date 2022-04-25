import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .utils import create_dir, open_json, to_json


def numpy_2_dataloader(X, y, batch_size=128, shuffle=False):
    """Convert data from numpy array to PyTorch DataLoader"""
    assert X.shape[0] == y.shape[0], 'X and y must have the same length.'

    dataset = TensorDataset(torch.from_numpy(X).type(torch.float32),
                            torch.from_numpy(y).type(torch.int64))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


class EarlyStopping:
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """

    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            # print(
            #     f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                # print('INFO: Early stopping')
                self.early_stop = True


def train(dataloader, model, loss_fn, optimizer, device):
    n = len(dataloader.dataset)
    n_batches = len(dataloader)
    loss_avg, correct = 0, 0

    model.train()
    for X, y in dataloader:
        X = X.to(device)
        y = y.to(device)
        assert X.size(0) == y.size(0)

        optimizer.zero_grad()
        output = model(X)
        assert output.size(0) == y.size(0)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()

        loss_avg += loss.item()
        correct += (output.argmax(1) == y).type(torch.float).sum().item()

    loss_avg /= n_batches
    acc = correct / n
    return acc, loss_avg


def evaluate(dataloader, model, loss_fn, device):
    n = len(dataloader.dataset)
    n_batches = len(dataloader)
    loss_avg, correct = 0, 0

    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            output = model(X)
            assert output.size(0) == y.size(0)
            loss_avg += loss_fn(output, y).item()
            correct += (output.argmax(1) == y).type(torch.float).sum().item()

    loss_avg /= n_batches
    acc = correct / n
    return acc, loss_avg


def train_model(model, dataloader, optimizer, loss_fn, device, max_epochs):
    early_stopping = EarlyStopping()

    for epoch in range(max_epochs):
        acc_train, loss_train = train(
            dataloader, model, loss_fn, optimizer, device)
        early_stopping(loss_train)
        if early_stopping.early_stop:
            # print('Stop at: {}'.format(epoch))
            break
    return acc_train, loss_train


def predict(X, model, device, batch_size=128):
    n = len(X)
    dataset = TensorDataset(torch.from_numpy(X).type(torch.float32))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    pred = torch.empty(n, dtype=torch.long)

    start = 0
    model.eval()
    with torch.no_grad():
        for [x] in dataloader:
            x = x.to(device)
            output = model(x)
            end = start + x.size(0)
            pred[start: end] = output.argmax(1).detach().cpu()
            start = end

    return pred.numpy()


def predict_proba(X, model, device, batch_size=128):
    n = len(X)
    dataset = TensorDataset(torch.from_numpy(X).type(torch.float32))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    it = iter(dataloader)
    [x] = it.next()
    output = model(x.to(device))
    n_output = output.size(-1)
    pred = torch.empty((n, n_output), dtype=torch.float32)

    start = 0
    model.eval()
    with torch.no_grad():
        for [x] in dataloader:
            x = x.to(device)
            output = torch.softmax(model(x), 1)
            end = start + x.size(0)
            pred[start: end] = output.detach().cpu()
            start = end

    return pred.numpy()


def get_correct_examples(model, X, Y, device='cpu', batch_size=128):
    """Removes incorrect predictions."""
    model.eval()
    corrects = torch.zeros(len(X), dtype=torch.bool)
    dataloader = numpy_2_dataloader(X, Y, batch_size=batch_size, shuffle=False)
    start = 0
    with torch.no_grad():
        for x, y in dataloader:
            n = x.size(0)
            end = start + n
            x = x.to(device)
            y = y.to(device)
            outputs = model(x)
            pred = outputs.max(1, keepdim=True)[1]
            corrects[start:end] = y.eq(pred.view_as(y)).cpu()
            start += n
    indices = torch.squeeze(torch.nonzero(corrects), 1).numpy()
    return (X[indices], Y[indices])


class NeuralNet(nn.Module):
    """A simple fullly-connected neural network with 1 hidden-layer"""

    def __init__(self, input_dim, hidden_dim=512, output_dim=2):
        super(NeuralNet, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x


class NNClassifier:
    def __init__(self,
                 input_dim=167,
                 hidden_dim=512,
                 output_dim=2,
                 batch_size=128,
                 max_epochs=300,
                 lr=1e-3,
                 momentum=0.9,
                 weight_decay=1e-4,
                 Model=NeuralNet,
                 device='cuda'):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.Model = Model
        self.device = torch.device(device)

        self.clf = Model(input_dim, hidden_dim, output_dim).to(self.device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(
            self.clf.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    def fit(self, X, y):
        dataset = TensorDataset(
            torch.from_numpy(X).type(torch.float32),
            torch.from_numpy(y).type(torch.int64)
        )
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        train_model(self.clf, dataloader, self.optimizer, self.loss_fn,
                    self.device, self.max_epochs)

    def predict(self, X):
        return predict(X, self.clf, self.device, self.batch_size)

    def predict_proba(self, X):
        return predict_proba(X, self.clf, self.device, self.batch_size)

    def score(self, X, y):
        dataset = TensorDataset(
            torch.from_numpy(X).type(torch.float32),
            torch.from_numpy(y).type(torch.int64)
        )
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        acc, _ = evaluate(dataloader, self.clf, self.loss_fn, self.device)
        return acc

    def save(self, path):
        create_dir(path)
        torch.save(self.clf.state_dict(), os.path.join(path, 'NeuralNet.torch'))
        params = {
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
            'batch_size': self.batch_size,
            'max_epochs': self.max_epochs,
            'lr': self.lr,
            'momentum': self.momentum,
            'weight_decay': self.weight_decay,
            'device': str(self.device),
        }
        to_json(params, os.path.join(path, 'NNClassifier.json'))

    def load(self, path):
        params = open_json(os.path.join(path, 'NNClassifier.json'))
        self.input_dim = params['input_dim']
        self.hidden_dim = params['hidden_dim']
        self.output_dim = params['output_dim']
        self.batch_size = params['batch_size']
        self.max_epochs = params['max_epochs']
        self.lr = params['lr']
        self.device = torch.device(params['device'])

        self.clf = self.Model(
            self.input_dim,
            self.hidden_dim,
            self.output_dim
        ).to(self.device)
        self.clf.load_state_dict(
            torch.load(
                os.path.join(path, 'NeuralNet.torch'),
                map_location=self.device
            )
        )
        self.optimizer = torch.optim.SGD(self.clf.parameters(),
                                         lr=self.lr,
                                         momentum=0.9,
                                         weight_decay=1e-4)
