import datetime
import json
import logging
import os
import random
import time

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype

logger = logging.getLogger(__name__)


def category2code(df):
    """Change all columns to ordinal data, represent in codes.
    (e.g.: from {2, 5, 8} to {0, 1, 2})
    """
    result = df.copy()
    for c in df.columns:
        categories = np.sort(np.unique(df[c].to_numpy()))
        cattype = CategoricalDtype(categories=categories, ordered=True)
        result[c] = result[c].astype(cattype).cat.codes
    return result


def drop_redundant_col(df):
    """Drop columns that use same value for all samples."""
    df_filtered = df.copy()
    df_filtered = df_filtered.iloc[:, np.where(df_filtered.nunique() > 1)[0]]
    return df_filtered


def get_range(X):
    """Return a tuple of minimal and maximum values"""
    if isinstance(X, pd.DataFrame):
        return (X.min(axis=0).to_numpy(), X.max(axis=0).to_numpy())
    return X.min(axis=0), X.max(axis=0)


def maccs2binary(data, dtype=float):
    """Convert MACCS into binary data"""
    try:
        if not isinstance(data, np.ndarray):
            raise ValueError
        data = np.copy(data)
        data[data >= 1] = 1
        data[data < 1] = 0
        data = data.astype(dtype)
        return data
    except ValueError:
        logger.error(f'Expecting data to be a numpy.ndarray. Got {type(data)}')


def set_seed(random_state=None):
    """Reset RNG seed."""
    if random_state is None:
        random_state = random.randint(1, 999999)
    random_state = int(random_state)
    random.seed(random_state)
    np.random.seed(random_state)
    # torch.manual_seed(random_state)
    # torch.cuda.manual_seed_all(random_state)
    # torch.backends.cudnn.deterministic = True
    logger.info(f'Set random state to: {random_state}')
    return random_state


def time2str(time_elapsed, formatstr='%Hh%Mm%Ss'):
    """Format millisecond to string."""
    return time.strftime(formatstr, time.gmtime(time_elapsed))


def to_json(data_dict, path):
    """Save dictionary as JSON."""
    def converter(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime.datetime):
            return obj.__str__()

    with open(path, 'w') as file:
        logger.info(f'Save to: {path}')
        json.dump(data_dict, file, default=converter)


def open_json(path):
    """Read JSON file."""
    try:
        with open(path, 'r') as file:
            data_json = json.load(file)
            return data_json
    except:
        logger.error(f'Cannot open {path}')


def create_dir(path):
    """Create directory if the input path is not found."""
    if not os.path.exists(path):
        logger.info(f'Creating directory: {path}')
        os.makedirs(path)


def open_csv(path_data, label_name='y'):
    """Read data from a CSV file, return X, y and column names."""
    logger.info(f'Load from: {path_data}')
    df_data = pd.read_csv(path_data)
    y = df_data[label_name].to_numpy()
    df_data = df_data.drop([label_name], axis=1)
    cols = df_data.columns
    X = df_data.to_numpy()
    return X, y, cols


def to_csv(X, y, cols, path_data):
    """Save data into a CSV file."""
    logger.info(f'Save to: {path_data}')
    df = pd.DataFrame(X, columns=cols, dtype=np.float32)
    labels = len(np.unique(y))
    df['y'] = pd.Series(y, dtype=int)
    df['y'] = df['y'].astype('category').cat.codes
    df.to_csv(path_data, index=False)
