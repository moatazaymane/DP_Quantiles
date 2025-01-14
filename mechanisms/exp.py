import numpy as np
import math
from utils.preprocessing import prepare_samples
from typing import Tuple


def exponential_mechanism(
    samples: np.ndarray, 
    q: float, 
    quantile: float, 
    epsilon: float, 
    a: float, 
    b: float, 
    point_estimator=False, 
    seed: int | None = None, 
    tol=1e-9
) -> Tuple[np.ndarray, float]:
    
    """
    Applies the exponential mechanism to a set of samples, computing the probability 
    distribution over intervals between consecutive samples.
    
    Parameters:
    ----------
    samples : numpy.ndarray
        The array of samples to which the exponential mechanism will be applied. These 
        are original data points used for creating intervals.

    q : float
        The quantile of interest, specified as a value between 0 and 1. For example, 
        q=0.5 corresponds to the median.

    quantile : float
        The value of the quantile, derived as `samples[qn]`, where `qn` is the index 
        corresponding to the q-th quantile. This is primarily used for debugging.

    epsilon : float
        The privacy budget parameter, controlling the level of differential privacy. Smaller 
        epsilon values enforce stricter privacy but may reduce utility.

    a : float
        The lower bound for the range of the mechanism. Defines the minimum allowable value 
        for interval selection.

    b : float
        The upper bound for the range of the mechanism. Defines the maximum allowable value 
        for interval selection.

    point_estimator : bool, optional (default=False)
        If set to True, the function returns a point estimator for the target quantile 
        instead of a probability distribution over intervals.

    seed : int or None, optional (default=None)
        The random seed for reproducibility. Setting a specific seed ensures consistent 
        probabilistic outputs across runs.

    tol : float, optional (default=1e-9)
        The tolerance level for numerical computations, used to handle edge cases or 
        prevent division by zero.

    Returns:
    -------
    (probabilities, point_estimator) : (numpy.ndarray, float)
        - probabilities: A probability distribution over the intervals between consecutive 
          samples (n - 1 intervals in total). Probabilities are influenced by the proximity 
          of the intervals to the quantile.
        - point_estimator: A point value representing the quantile of interest, sampled 
          based on the computed probabilities.

    """


        
    sorted_samples = prepare_samples(samples, q, a, b, tol, quantile=quantile)
    n = len(sorted_samples)
    #print(('LOGGED', quantile, sorted_samples[math.ceil(q * (n-2))]))

    if len(quantile) == 1:
        assert quantile == sorted_samples[math.ceil(q * (n-2))]

    probabilities = np.array([])
    exists = False

    for i in range(len(sorted_samples) - 1):

        fact = epsilon / 2

        r_i = np.abs(i + 1 - (math.ceil(q * (n-2)) + 1))

        assert np.abs(sorted_samples[i+1] - sorted_samples[i]) >= 0
        probabilities = np.append(probabilities, np.log(sorted_samples[i+1] - sorted_samples[i])  - fact * r_i)

        if not exists and r_i == 0:
            exists = True

    quantile_estimator = None
    if point_estimator:
        np.random.seed(seed=seed)
        r_probabilities = probabilities + np.random.gumbel(loc=0.0, scale=1.0, size=len(probabilities))
        centered_sample_index= np.argmax(r_probabilities)
        sampled_interval_exp= sorted_samples[centered_sample_index: centered_sample_index + 2]
        quantile_estimator = np.random.uniform(sampled_interval_exp[0], sampled_interval_exp[1])

    assert exists 
    return probabilities, quantile_estimator, sorted_samples
