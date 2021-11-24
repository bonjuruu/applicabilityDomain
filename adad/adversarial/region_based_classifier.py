"""Region-based classification for PyTorch and sklearn.
"""
import datetime
import logging
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from adad import AppDomainBase

logger = logging.getLogger(__name__)


if __name__ == '__main__':
    model = AppDomainBase()
    X = np.random.rand((5, 4))
    y = np.zeros(4)
    model.fit(X, y)
