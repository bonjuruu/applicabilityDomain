"""Region-based classification for PyTorch and scikit-learn.
"""
import logging
import time

import numpy as np
from tqdm import tqdm

from adad.app_domain_base import AppDomainBase
from adad.utils import time2str

logger = logging.getLogger(__name__)


def generate_random_samples(x, x_min, x_max, r, size):
    """Generates uniformly distributed random samples around x within hypercube
    B(x, r), where r is L-infinity distance.
    """
    shape = tuple([size] + list(x.shape))
    dtype = x.dtype
    noise = np.random.uniform(low=-abs(r), high=abs(r),
                              size=shape).astype(dtype)
    rng_samples = np.repeat([x], repeats=size, axis=0) + noise
    rng_samples = np.clip(rng_samples, x_min, x_max)
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
    x_max: float or array, default=1.0
    max_r: float, optional, default=0.4
        Maximum searching radius.
    step_size:float, default=0.01
        Step size for each iteration.
    eps: float, default=0.01
        The tolerance parameter for searching radius.
    verbose: int, default=0
        Report information if verbose is greater than 0.
    """

    def __init__(self, clf, sample_size=1000, x_min=0.0, x_max=1.0,
                 max_r=0.4, step_size=0.01, eps=0.99, verbose=0):
        self.clf = clf
        self.sample_size = sample_size
        self.x_min = x_min
        self.x_max = x_max
        self.max_r = max_r
        self.step_size = step_size
        self.eps = eps
        self.verbose = verbose

        # Initial radius without training
        self.r = max_r

    def fit(self, X, y):
        """Search the threshold using given training data"""
        time_start = time.perf_counter()
        acc_pt = self.clf.score(X, y)
        if self.verbose > 0:
            logger.info(f'Point-based accuracy: {acc_pt * 100:.2f}%.')

        # Apply Binary search to find r value.
        r_low = 0.
        r_high = self.max_r
        while r_low < r_high:
            r_mid = (r_high + r_low) / 2.
            acc_mid = self.score_region(X, y, r_mid)
            # If x is greater, ignore left half
            if acc_mid + self.eps > acc_pt:
                r_low = r_mid + self.step_size
            # If x is smaller, ignore right half
            elif acc_mid - self.eps < acc_pt:
                r_high = r_mid - self.step_size
            else:
                r_best = r_mid
                break

        if self.verbose > 0:
            time_elapsed = time.perf_counter() - time_start
            logger.info(f'Total training time: {time2str(time_elapsed)}')

        self.r = r_best
        return self

    def measure(self, X):
        """Check AD on X. Smaller value indicates it within the domain."""
        pred_pt = self.clf.predict(X)
        results = np.zeros_like(pred_pt, dtype=float)

        n = X.shape[0]
        # Region-based prediction is performed 1-by-1.
        pbar = tqdm(range(n), ncols=100) if self.verbose else range(n)
        for i in pbar:
            x_rng = generate_random_samples(
                X[i], x_min=self.x_min, x_max=self.x_max,
                r=self.r, size=self.sample_size)
            pred_rng = self.model.predict(x_rng)
            bincount = np.bincount(pred_rng)  # Build a histogram
            # Record how many random samples match the point-based prediction.
            results[i] = bincount[pred_pt]

        # Since 0 means perfectly within the domain, so invert the value.
        results = (self.sample_size - results) / self.sample_size
        return results

    def predict_region(self, X, r=None):
        """Predict class labels using Region-based classifier."""
        if r is None:
            r = self.r
        pred_region = -np.ones(len(X), dtype=float)

        n = X.shape[0]
        # Region-based prediction is performed 1-by-1.
        pbar = tqdm(range(n), ncols=100) if self.verbose else range(n)
        for i in pbar:
            x_rng = generate_random_samples(
                X[i], x_min=self.x_min, x_max=self.x_max,
                r=r, size=self.sample_size)
            pred_rng = self.model.predict(x_rng)
            # Build a histogram
            pred_region[i] = np.bincount(pred_rng).argmax()
        return pred_region

    def score_region(self, X, y, r=None):
        """Compute accuracy using Regin-based classifier."""
        pred = self.predict_region(X, r)
        return np.mean(pred == y)
