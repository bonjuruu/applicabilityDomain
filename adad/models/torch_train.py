import torch
from torch.utils.data import DataLoader, TensorDataset

from .earlystopping import EarlyStopping


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

    for _ in range(max_epochs):
        acc_train, loss_train = train(
            dataloader, model, loss_fn, optimizer, device)
        early_stopping(loss_train)
        if early_stopping.early_stop:
            break
    return acc_train, loss_train


def get_correct_examples(model, dataset, shape, device='cuda',
                         batch_size=512, return_tensor=True):
    """Removes incorrect predictions."""
    model.eval()
    X = torch.zeros(shape, dtype=torch.float32)
    Y = torch.zeros(len(dataset), dtype=torch.long)
    corrects = torch.zeros(len(dataset), dtype=torch.bool)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    start = 0
    with torch.no_grad():
        for x, y in dataloader:
            n = x.size(0)
            end = start + n
            X[start:end] = x
            Y[start:end] = y
            x = x.to(device)
            y = y.to(device)
            outputs = model(x)
            pred = outputs.max(1, keepdim=True)[1]
            corrects[start:end] = y.eq(pred.view_as(y)).cpu()
            start += n
    indices = torch.squeeze(torch.nonzero(corrects), 1)
    if return_tensor:
        return X[indices], Y[indices]
    dataset = TensorDataset(X[indices], Y[indices])
    return dataset
