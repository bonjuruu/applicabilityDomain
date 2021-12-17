import torch
import torch.nn.functional as F


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
