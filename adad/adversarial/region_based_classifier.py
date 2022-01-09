"""Region-based classification for PyTorch and scikit-learn.
"""
import logging
import os
import time

import numpy as np
from tqdm import tqdm

from adad.app_domain_base import AppDomainBase
from adad.utils import create_dir, open_json, time2str, to_json

logger = logging.getLogger(__name__)


def generate_random_samples(x, x_min, x_max, r, size):
    """Generates uniformly distributed random samples around x within hypercube
    B(x, r), where r is L-infinity distance.
    """
    shape = tuple([size] + list(x.shape))
    noise = np.random.uniform(low=-r, high=r, size=shape).astype(x.dtype)
    rng_samples = np.repeat([x], repeats=size, axis=0) + noise
    rng_samples = np.clip(rng_samples, x_min, x_max)
    return rng_samples


def generate_random_int_samples(x, x_min, x_max, r, size):
    """Generate random samples around integer value x, bounded by Manhattan
    distance [-1, 1].
    """
    dtype = x.dtype
    n_features = len(x)
    n_flip = int(np.floor(r * n_features))
    shape = (size, n_flip)
    # Find the indices want to add noise (5% of 100 attributes mean we choose 5 indices)
    indices = np.zeros(shape, dtype=int)
    for i in range(size):
        indices[i] = np.random.choice(
            np.arange(n_features), size=n_flip, replace=False)
    # Generate a list of {-1, 1} with same length of the indices to represent noise
    noise = np.random.randint(0, 2, size=shape).astype(dtype)
    noise[noise == 0] = -1
    # noise = np.ones(shape=shape).astype(dtype)
    rng_samples = np.repeat([x], repeats=size, axis=0)
    # Added noise to the sample
    for i_sample, i_feature in enumerate(indices):
        rng_samples[i_sample, i_feature] += noise[i_sample]
    # We cant the random sample <= than the radius not always = the radius
    # To handle the situation where the majority of a sparse matrix only contain 0.
    # rng_samples[rng_samples < 0] = 1
    # Ensure the random samples are valid.
    rng_samples = np.clip(rng_samples, a_min=x_min, a_max=x_max)
    return rng_samples


class SklearnRegionBasedClassifier(AppDomainBase):
    """Region-based Classifier for Applicability Domain (scikit-learn version)

    Parameters
    ----------
    clf: classifier
        A pre-trained scikit-learn classifier.
    sample_size: int, default=1000
        The number of samples generated within the radius.
    x_min: float or array, default=0.0
        Default value is NOT suitable for discrete data.
    x_max: float or array, default=1.0
        Default value is NOT suitable for discrete data.
    r_min: float, optional, default=0.0
        Minimum searching radius.
    r_max: float, optional, default=0.2
        Maximum searching radius. (Use 0.4 for continuous data)
    step_size:float, default=1
        Step size for each iteration. Default value is NOT suitable for continuous
        data. (Use 0.01 for continuous data)
    eps: float, default=0.01
        The tolerance parameter for searching radius.
    data_type: {'continue', 'discrete'}, default='discrete'
    ci: float, default=0.9
        Confidence interval for regiona-based classifier. It should in-between (0, 1].
    verbose: int, default=0
        Report information if verbose is greater than 0.
    """

    def __init__(self, clf, sample_size=1000, x_min=0.0, x_max=1.0,
                 r_min=0., r_max=0.2, step_size=1, eps=0.01,
                 data_type='discrete', verbose=0):
        self.clf = clf
        self.sample_size = sample_size
        self.x_min = x_min
        self.x_max = x_max
        self.r_min = r_min
        self.r_max = r_max
        self.step_size = step_size
        self.eps = eps
        self.data_type = data_type
        self.verbose = verbose

        if self.verbose > 0:
            logging.basicConfig(
                format='[%(name)s]:%(message)s', level=logging.INFO)

        if r_min >= r_max:
            raise ValueError('Invalid initial r value!')
        if not data_type in ['discrete', 'continue']:
            raise ValueError(
                'data_type can only be either discrete or continue!')

        if data_type == 'discrete':
            self.rng_sample_generator = generate_random_int_samples
        else:
            self.rng_sample_generator = generate_random_samples

        # Initial radius without training
        self.r = r_max
        self.n_class = 0

    def fit(self, X, y):
        """Search the threshold using given training data"""
        time_start = time.perf_counter()

        acc_pt = self.clf.score(X, y)
        logger.info(f'Point-based accuracy: {acc_pt * 100:.2f}%.')
        self.n_class = len(np.unique(y))
        logger.info(f'n_class: {self.n_class}')

        if self.data_type == 'discrete':
            n_features = X.shape[1]
            min_step = self.step_size / n_features + 0.001
        else:
            min_step = self.step_size

        # Apply Binary search to find r value.
        r_low = self.r_min
        r_high = self.r_max
        while r_low < r_high:
            r_mid = (r_high + r_low) / 2.
            logger.info('Search Low: {:.4f} Mid: {:.4f} High: {:.4f}'.format(
                r_low, r_mid, r_high))
            acc_mid = self.score_region(X, y, r_mid)
            # If x is greater, ignore left half
            if acc_mid + self.eps > acc_pt:
                r_low = r_mid + min_step
            # If x is smaller, ignore right half
            elif acc_mid - self.eps < acc_pt:
                r_high = r_mid - min_step
            else:
                break

        time_elapsed = time.perf_counter() - time_start
        logger.info(f'Total training time: {time2str(time_elapsed)}')

        self.r = r_mid
        logger.info(f'Best r: {r_mid:.4f}')
        return self

    def set_params(self, r, n_class=2):
        """Set the parameter without training (calling the fit method)."""
        self.r = r
        self.n_class = n_class

    def measure(self, X):
        """Check AD on X. Smaller value indicates it within the domain."""
        pred_pt = self.clf.predict(X).astype(int)
        measure = np.zeros_like(pred_pt, dtype=float)

        n = X.shape[0]
        # Region-based prediction is performed 1-by-1.
        pbar = tqdm(range(n), ncols=100) if self.verbose else range(n)
        for i in pbar:
            bincount = self.__get_bin(X[i], r=self.r)
            # Record how many random samples match the point-based prediction.
            measure[i] = bincount[pred_pt[i]]

        results = measure / self.sample_size
        return results

    def predict_region(self, X, r=None):
        """Predict class labels using Region-based classifier."""
        if r is None:
            r = self.r
        pred_region = -np.ones(len(X), dtype=int)

        n = X.shape[0]
        # Region-based prediction is performed 1-by-1.
        pbar = tqdm(range(n), ncols=100) if self.verbose else range(n)
        for i in pbar:
            bincount = self.__get_bin(X[i], r=r)
            pred_region[i] = bincount.argmax()
        return pred_region

    def score_region(self, X, y, r=None):
        """Compute accuracy using Regin-based classifier."""
        pred = self.predict_region(X, r)
        return np.mean(pred == y)

    def __get_bin(self, x, r):
        x_rng = self.rng_sample_generator(
            x, x_min=self.x_min, x_max=self.x_max,
            r=r, size=self.sample_size)
        pred_rng = self.clf.predict(x_rng).astype(int)
        # Build a histogram
        bincount = np.bincount(pred_rng, minlength=self.n_class).astype(int)
        return bincount

    def save(self, path):
        params = {
            'sample_size': self.sample_size,
            'x_min': self.x_min,
            'x_max': self.x_max,
            'r_min': self.r_min,
            'r_max': self.r_max,
            'step_size': self.step_size,
            'eps': self.eps,
            'data_type': self.data_type,
            'verbose': self.verbose,
            'r': self.r,
            'n_class': self.n_class,
        }
        create_dir(path)
        to_json(params, os.path.join(path, 'rc_params.json'))

    def load(self, path):
        params = open_json(os.path.join(path, 'rc_params.json'))
        self.sample_size = params['sample_size']
        self.x_min = params['x_min']
        self.x_max = params['x_max']
        self.r_min = params['r_min']
        self.r_max = params['r_max']
        self.step_size = params['step_size']
        self.eps = params['eps']
        self.data_type = params['data_type']
        self.verbose = params['verbose']
        self.r = params['r']
        self.n_class = params['n_class']
