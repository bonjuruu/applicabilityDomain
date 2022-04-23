import torch
import torch.nn as nn
import torch.nn.functional as F


class NumericModel(nn.Module):
    def __init__(self, n_features, n_hidden, n_classes):
        super(NumericModel, self).__init__()
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_classes = n_classes

        self.ln1 = nn.Linear(n_features, n_hidden)
        self.ln2 = nn.Linear(n_hidden, n_hidden)
        self.ln3 = nn.Linear(n_hidden, n_classes)

    def forward(self, x):
        x = F.relu(self.ln1(x))
        x = F.relu(self.ln2(x))
        x = self.ln3(x)
        return x


def test(model):
    model.to('cpu')
    X = torch.randn(5, 4, dtype=torch.float32)
    y = model(X)
    assert y.size() == (5, 3) 
    print('Pass test')


if __name__ == '__main__':
    test(NumericModel(4, 16, 3))
