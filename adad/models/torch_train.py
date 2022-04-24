import torch
from torch.utils.data import DataLoader, TensorDataset

from adad.torch_utils import numpy_2_dataloader


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
