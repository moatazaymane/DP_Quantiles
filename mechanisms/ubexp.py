import numpy as np
from utils.preprocessing import prepare_samples_ubexp
from typing import Tuple



def unbiased_exponential_mechanism(
    samples: np.ndarray, 
    q: float, 
    epsilon: float, 
    quantile: float, 
    b: float, 
    verify_unbiased=True, 
    distribution=None, 
    gap_q=None, 
    point_estimator=False, 
    seed: int | None = None, 
    sigma=None, 
    tol=1e-9
) -> Tuple[np.ndarray, float]:
    """
    Applies the unbiased exponential mechanism to a set of samples, computing the probability 
    distribution over intervals between consecutive samples.

    Parameters:
    ----------
    samples : numpy.ndarray
        An array of samples to which the unbiased exponential mechanism will be applied.

    q : float
        The quantile of interest, specified as a value between 0 and 1. For example, q=0.5 
        corresponds to the median.

    epsilon : float
        The privacy budget parameter, controlling the level of differential privacy. Smaller 
        epsilon values enforce stricter privacy but may reduce utility.

    quantile : float
        The value of the quantile, derived as `samples[qn]`, where `qn` is the index 
        corresponding to the q-th quantile. This value is used for debugging and validation.

    b : float
        A parameter defining the scale of the utility function used in the exponential mechanism. 
        It influences the weight assigned to intervals.

    verify_unbiased : bool, optional (default=True)
        If set to True, ensures that the mechanism remains unbiased with respect to the quantile. 
        Useful for validation purposes.

    distribution : callable or None, optional (default=None)
        the distribution of the data if set to uniform or truncated normal the range is stretched considerably less

    gap_q : float or None, optional (default=None)
        an upper bound on the sensitibity of the quantile function

    point_estimator : bool, optional (default=False)
        If set to True, the function returns a point estimator for the target quantile instead 
        of a probability distribution over intervals.

    seed : int or None, optional (default=None)
        The random seed for reproducibility. Setting a specific seed ensures consistent 
        probabilistic outputs across runs.

    sigma : float or None, optional (default=None)
        The standard deviation of Gaussian noise added to the samples prior to applying the mechanism. 
        If None, no noise is added.

    tol : float, optional (default=1e-9)
        The tolerance level for numerical computations, used to handle edge cases or prevent 
        division by zero.

    Returns:
    -------
    (probabilities, point_estimator) : (numpy.ndarray, float)
        - probabilities: A probability distribution over the intervals between consecutive 
          samples (n - 1 intervals in total), normalized to sum to 1.
        - point_estimator: A point value representing the quantile of interest, sampled based 
          on the computed probabilities.
    """

        
    sorted_samples, mn_q = prepare_samples_ubexp(samples=samples, q=q, b=b, n = len(samples), distribution=distribution, gap_q=gap_q, sigma=sigma)
    n = len(sorted_samples)

    assert np.array_equal(sorted_samples, np.sort(sorted_samples)), "The array is not sorted."

    assert n == 2*mn_q + 3 and quantile == sorted_samples[mn_q + 1]

    probas = np.zeros(2 * mn_q + 2) # setting a probabilit for each of the 2*mn_q + 2 intervals

    fact = epsilon / 4
    for i in range(1, mn_q + 2):

        r_i = i - 1
        probas[mn_q + i] = np.log(sorted_samples[mn_q + i  +1] - sorted_samples[mn_q + i])  - fact * r_i
        probas[mn_q - i + 1] = np.log(sorted_samples[mn_q - i + 2] - sorted_samples[mn_q - i + 1])  - fact * r_i


        if verify_unbiased:
            
            assert np.isclose(sorted_samples[mn_q + 1 + i - 1] + sorted_samples[mn_q + 1 - i + 1], 2 * sorted_samples[mn_q + 1], atol = tol)
            assert np.isclose(sorted_samples[mn_q - i + 1] + sorted_samples[mn_q + i + 1], 2 * sorted_samples[mn_q + 1], atol = tol)

            assert np.isclose(sorted_samples[mn_q + i  +1] - sorted_samples[mn_q + i],sorted_samples[mn_q - i + 2] - sorted_samples[mn_q - i + 1], atol = tol)
            assert np.isclose(probas[mn_q + i], probas[mn_q - i + 1], atol = tol)

    quantile_estimator = None
    if point_estimator:
        np.random.seed(seed=seed)
        r_probabilities = probas + np.random.gumbel(loc=0.0, scale=1.0, size=len(probas))
        centered_sample_index= np.argmax(r_probabilities)
        sampled_interval_exp= sorted_samples[centered_sample_index: centered_sample_index + 2]
        quantile_estimator = np.random.uniform(sampled_interval_exp[0], sampled_interval_exp[1])


    return probas, quantile_estimator, sorted_samples
