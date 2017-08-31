"""
https://pymc-devs.github.io/pymc3/notebooks/pmf-pymc.html
"""

from collections import OrderedDict
import numpy as np
import pandas as pd

# Define our evaluation function.
def rmse(test_data, predicted):
    """Calculate root mean squared error.
    Ignoring missing values in the test data.
    """
    I = ~np.isnan(test_data)   # indicator for missing values
    N = I.sum()                # number of non-missing values
    sqerror = abs(test_data - predicted) ** 2  # squared error array
    mse = sqerror[I].sum() / N                 # mean squared error
    return np.sqrt(mse)                        # RMSE


# Create a base class with scaffolding for our 3 baselines.
def split_title(title):
    """Change "BaselineMethod" to "Baseline Method"."""
    words = []
    tmp = [title[0]]
    for c in title[1:]:
        if c.isupper():
            words.append(''.join(tmp))
            tmp = [c]
        else:
            tmp.append(c)
    words.append(''.join(tmp))
    return ' '.join(words)


class Baseline(object):
    """Calculate baseline predictions."""

    def __init__(self, train_data):
        """Simple heuristic-based transductive learning to fill in missing
        values in data matrix."""
        self.predict(train_data.copy())

    def predict(self, train_data):
        raise NotImplementedError(
            'baseline prediction not implemented for base class')

    def rmse(self, test_data):
        """Calculate root mean squared error for predictions on test data."""
        # train_nan_mask = np.isnan(self.train_data)
        # test_data[train_nan_mask] = np.nan
        return rmse(test_data, self.predicted)

    def __str__(self):
        return split_title(self.__class__.__name__)

# Implement the 3 baselines.
class UniformRandomBaseline(Baseline):
    """Fill missing values with uniform random values."""

    def predict(self, train_data):
        self.train_data = train_data
        nan_mask = np.isnan(train_data)
        masked_train = np.ma.masked_array(train_data, nan_mask)
        pmin, pmax = masked_train.min(), masked_train.max()
        N = nan_mask.sum()
        train_data[nan_mask] = np.random.uniform(pmin, pmax, N)
        self.predicted = train_data


class GlobalMeanBaseline(Baseline):
    """Fill in missing values using the global mean."""

    def predict(self, train_data):
        self.train_data = train_data
        nan_mask = np.isnan(train_data)
        train_data[nan_mask] = train_data[~nan_mask].mean()
        self.predicted = train_data


class MeanOfMeansBaseline(Baseline):
    """Fill in missing values using mean of user/item/global means."""

    def predict(self, train_data):
        self.train_data = train_data
        nan_mask = np.isnan(train_data)
        masked_train = np.ma.masked_array(train_data, nan_mask)
        global_mean = masked_train.mean()
        user_means = masked_train.mean(axis=1)
        item_means = masked_train.mean(axis=0)
        self.predicted = train_data.copy()
        n, m = train_data.shape
        for i in range(n):
            for j in range(m):
                if np.ma.isMA(item_means[j]):
                    self.predicted[i,j] = np.mean(
                        (global_mean, user_means[i]))
                else:
                    self.predicted[i,j] = np.mean(
                        (global_mean, user_means[i], item_means[j]))

def train_test(data, ratio=0.1):
    data = data.copy().values
    nrows, ncols = data.shape
    test_ind = np.random.choice(np.arange(data.size), size=int(data.size*ratio), replace=False)

    train = data.copy()
    train = train.reshape(nrows*ncols)
    train[test_ind] = np.nan
    train = train.reshape((nrows, ncols))

    test = data.copy()
    test[~np.isnan(train)] = np.nan

    return test, train


if __name__ == "__main__":
    baseline_methods = OrderedDict()
    baseline_methods['ur'] = UniformRandomBaseline
    baseline_methods['gm'] = GlobalMeanBaseline
    baseline_methods['mom'] = MeanOfMeansBaseline

    data = pd.read_csv('~griggs/Galvanize_Course_Work/Week8/Day5_Case_Study_Recommender/data/jokes/jester-dataset-v1-dense-first-1000.csv')
    test, train = train_test(data)

    baselines = {}
    for name in baseline_methods:
        Method = baseline_methods[name]
        method = Method(train)
        baselines[name] = method.rmse(test)
        print('%s RMSE:\t%.5f' % (method, baselines[name]))
