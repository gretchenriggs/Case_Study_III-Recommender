import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import shutil
import pymc3 as pm
import datetime
import time
import theano
import scipy as sp
import logging
import hashlib
try:
    import ujson as json
except ImportError:
    import json
DATA_DIR = '~griggs/Galvanize_Course_Work/Week8/Day5_Case_Study_Recommender/data/'

from Baselines_gmr import UniformRandomBaseline, GlobalMeanBaseline, MeanOfMeansBaseline
from collections import OrderedDict

# Enable on-the-fly graph computations, but ignore absence of intermediate test
# values.
theano.config.compute_test_values = 'ignore'

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def train_test(data, ratio=0.2):
    test = data.values
    train = data.values.flatten()
    train[np.random.randint(0, data.size, int(data.size*ratio))] = np.nan
    train = train.reshape(data.shape[0],data.shape[1])
    return test, train

class PMF(object):
    """Probabilistic Matrix Factorization model using pymc3."""

    def __init__(self, train, dim, alpha=2, std=0.01, bounds=(-10, 10)):
        """Build the Probabilistic Matrix Factorization model using pymc3.

        :param np.ndarray train: The training data to use for learning the model.
        :param int dim: Dimensionality of the model; number of latent factors.
        :param int alpha: Fixed precision for the likelihood function.
        :param float std: Amount of noise to use for model initialization.
        :param (tuple of int) bounds: (lower, upper) bound of ratings.
            These bounds will simply be used to cap the estimates produced for R.

        """
        self.dim = dim
        self.alpha = alpha
        self.std = np.sqrt(1.0 / alpha)
        self.bounds = bounds
        self.data = train.copy()
        n, m = self.data.shape

        # Perform mean value imputation
        nan_mask = np.isnan(self.data)
        self.data[nan_mask] = self.data[~nan_mask].mean()

        # Low precision reflects uncertainty; prevents overfitting.
        # Set to the mean variance across users and items.
        self.alpha_u = 1 / self.data.var(axis=1).mean()
        self.alpha_v = 1 / self.data.var(axis=0).mean()

        # Specify the model.
        logging.info('building the PMF model')
        with pm.Model() as pmf:
            U = pm.MvNormal(
                'U', mu=0, tau=self.alpha_u * np.eye(dim),
                shape=(n, dim), testval=np.random.randn(n, dim) * std)
            V = pm.MvNormal(
                'V', mu=0, tau=self.alpha_v * np.eye(dim),
                shape=(m, dim), testval=np.random.randn(m, dim) * std)
            R = pm.Normal(
                'R', mu=theano.tensor.dot(U, V.T), tau=self.alpha * np.ones((n, m)),
                observed=self.data)

        logging.info('done building the PMF model')
        self.model = pmf

    def __str__(self):
        return self.name


# MAP Estimation


# First define functions to save our MAP estimate after it is found.
# We adapt these from `pymc3`'s `backends` module, where the original
# code is used to save the traces from MCMC samples.
def save_np_vars(vars, savedir):
    """Save a dictionary of numpy variables to `savedir`. We assume
    the directory does not exist; an OSError will be raised if it does.
    """
    logging.info('writing numpy vars to directory: %s' % savedir)
    if not os.path.isdir(savedir):
        os.mkdir(savedir)
    shapes = {}
    for varname in vars:
        data = vars[varname]
        var_file = os.path.join(savedir, varname + '.txt')
        np.savetxt(var_file, data.reshape(-1, data.size))
        shapes[varname] = data.shape

        ## Store shape information for reloading.
        shape_file = os.path.join(savedir, 'shapes.json')
        with open(shape_file, 'w') as sfh:
            json.dump(shapes, sfh)


def load_np_vars(savedir):
    """Load numpy variables saved with `save_np_vars`."""
    shape_file = os.path.join(savedir, 'shapes.json')
    with open(shape_file, 'r') as sfh:
        shapes = json.load(sfh)

    vars = {}
    for varname, shape in shapes.items():
        var_file = os.path.join(savedir, varname + '.txt')
        vars[varname] = np.loadtxt(var_file).reshape(shape)

    return vars


# Now define the MAP estimation infrastructure.
def _map_dir(self):
    basename = 'pmf-map-d%d' % self.dim
    return os.path.join(DATA_DIR, basename)

def _find_map(self):
    """Find mode of posterior using Powell optimization."""
    tstart = time.time()
    with self.model:
        logging.info('finding PMF MAP using Powell optimization...')
        self._map = pm.find_MAP(fmin=sp.optimize.fmin_powell, disp=True)

    elapsed = int(time.time() - tstart)
    logging.info('found PMF MAP in %d seconds' % elapsed)

    # This is going to take a good deal of time to find, so let's save it.
#    save_np_vars(self._map, self.map_dir)

def _load_map(self):
    self._map = load_np_vars(self.map_dir)

def _map(self):
    try:
        return self._map
    except:
        if os.path.isdir(self.map_dir):
            self.load_map()
        else:
            self.find_map()
        return self._map


# Draw MCMC samples.
def _trace_dir(self):
    basename = 'pmf-mcmc-d%d' % self.dim
    return os.path.join(DATA_DIR, basename)

def _draw_samples(self, nsamples=1000, njobs=2):
    # First make sure the trace_dir does not already exist.
    if os.path.isdir(self.trace_dir):
        shutil.rmtree(self.trace_dir)

    with self.model:
        logging.info('drawing %d samples using %d jobs' % (nsamples, njobs))
        backend = pm.backends.Text(self.trace_dir)
        logging.info('backing up trace to directory: %s' % self.trace_dir)
        self.trace = pm.sample(draws=nsamples, init='advi',
                               n_init=150000, njobs=njobs, trace=backend)

def _load_trace(self):
    with self.model:
        self.trace = pm.backends.text.load(self.trace_dir)

def _predict(self, U, V):
    """Estimate R from the given values of U and V."""
    R = np.dot(U, V.T)
    n, m = R.shape
    sample_R = np.array([
        [np.random.normal(R[i,j], self.std) for j in range(m)]
        for i in range(n)
    ])

    # bound ratings
    low, high = self.bounds
    sample_R[sample_R < low] = low
    sample_R[sample_R > high] = high
    return sample_R

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


def eval_map(pmf_model, train, test):
    U = pmf_model.map['U']
    V = pmf_model.map['V']

    # Make predictions and calculate RMSE on train & test sets.
    predictions = pmf_model.predict(U, V)
    train_rmse = rmse(train, predictions)
    test_rmse = rmse(test, predictions)
    overfit = test_rmse - train_rmse

    # Print report.
    print('PMF MAP training RMSE: %.5f' % train_rmse)
    print('PMF MAP testing RMSE:  %.5f' % test_rmse)
    print('Train/test difference: %.5f' % overfit)

    return test_rmse

def _norms(pmf_model, monitor=('U', 'V'), ord='fro'):
    """Return norms of latent variables at each step in the
    sample trace. These can be used to monitor convergence
    of the sampler.
    """
    monitor = ('U', 'V')
    norms = {var: [] for var in monitor}
    for sample in pmf_model.trace:
        for var in monitor:
            norms[var].append(np.linalg.norm(sample[var], ord))
    return norms


def _traceplot(pmf_model):
    """Plot Frobenius norms of U and V as a function of sample #."""
    trace_norms = pmf_model.norms()
    u_series = pd.Series(trace_norms['U'])
    v_series = pd.Series(trace_norms['V'])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    u_series.plot(kind='line', ax=ax1, grid=False,
                  title="$\|U\|_{Fro}^2$ at Each Sample")
    v_series.plot(kind='line', ax=ax2, grid=False,
                  title="$\|V\|_{Fro}^2$ at Each Sample")
    ax1.set_xlabel("Sample Number")
    ax2.set_xlabel("Sample Number")

def _running_rmse(pmf_model, test_data, train_data, burn_in=0, plot=True):
    """Calculate RMSE for each step of the trace to monitor convergence.
    """
    burn_in = burn_in if len(pmf_model.trace) >= burn_in else 0
    results = {'per-step-train': [], 'running-train': [],
               'per-step-test': [], 'running-test': []}
    R = np.zeros(test_data.shape)
    for cnt, sample in enumerate(pmf_model.trace[burn_in:]):
        sample_R = pmf_model.predict(sample['U'], sample['V'])
        R += sample_R
        running_R = R / (cnt + 1)
        results['per-step-train'].append(rmse(train_data, sample_R))
        results['running-train'].append(rmse(train_data, running_R))
        results['per-step-test'].append(rmse(test_data, sample_R))
        results['running-test'].append(rmse(test_data, running_R))

    results = pd.DataFrame(results)

    if plot:
        results.plot(
            kind='line', grid=False, figsize=(15, 7),
            title='Per-step and Running RMSE From Posterior Predictive')

    # Return the final predictions, and the RMSE calculations
    return running_R, results




# Update our class with the new MAP infrastructure.
PMF.find_map = _find_map
PMF.load_map = _load_map
PMF.map_dir = property(_map_dir)
PMF.map = property(_map)

# Update our class with the sampling infrastructure.
PMF.trace_dir = property(_trace_dir)
PMF.draw_samples = _draw_samples
PMF.load_trace = _load_trace

PMF.predict = _predict

# Add eval function to PMF class.
PMF.eval_map = eval_map

PMF.norms = _norms
PMF.traceplot = _traceplot
PMF.running_rmse = _running_rmse


if __name__ == '__main__':
    data = pd.read_csv('~griggs/Galvanize_Course_Work/Week8/Day5_Case_Study_Recommender/data/jokes/jester-dataset-v1-dense-first-1000.csv')
    ratio = 0.10
    test, train = train_test(data, ratio)
    test_size = data.size * ratio
    # Finally, hash the indices and save the train/test sets.
    # tosample = np.where(~np.isnan(train))       # ignore nan values in data
    # idx_pairs = zip(tosample[0], tosample[1])   # tuples of row/col index pairs
    # indices = np.arange(len(idx_pairs))         # indices of index pairs
    # sample = np.random.choice(indices, replace=False, size=test_size)
    # index_string = ''.join(map(str, np.sort(sample)))
    # name = hashlib.sha1(index_string).hexdigest()
    # savedir = os.path.join(DATA_DIR, name)
    # save_np_vars({'train': train, 'test': test}, savedir)

    ALPHA = 2
    DIM = 5
    pmf = PMF(train, DIM, ALPHA, std=0.05)

    # Find MAP for PMF.
    pmf.find_map()

    # Evaluate PMF MAP estimates.
    pmf_map_rmse = pmf.eval_map(train, test)

    # Let's see the results:
    baseline_methods = OrderedDict()
    baseline_methods['ur'] = UniformRandomBaseline
    baseline_methods['gm'] = GlobalMeanBaseline
    baseline_methods['mom'] = MeanOfMeansBaseline

    baselines = {}
    for name in baseline_methods:
        Method = baseline_methods[name]
        method = Method(train)
        baselines[name] = method.rmse(test)
        print('%s RMSE:\t%.5f' % (method, baselines[name]))

    pmf_improvement = baselines['mom'] - pmf_map_rmse
    print('PMF MAP Improvement:   %.5f' % pmf_improvement)

    # Draw MCMC samples.
    pmf.draw_samples(300)

    # uncomment to load previous trace rather than drawing new samples.
    # pmf.load_trace()

    pmf.traceplot()

    predicted, results = pmf.running_rmse(test, train, burn_in=200)
