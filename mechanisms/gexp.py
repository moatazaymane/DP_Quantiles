import numpy as np
import math
from typing import Tuple
import scipy.stats as stats
from scipy.stats import truncnorm
from utils.preprocessing import prepare_samples


def gaussian_exponential_mechanism(
    samples: np.ndarray, 
    quantile, 
    sigma_gexp: float, 
    q: float, 
    epsilon: float, 
    eps_mu: float, 
    attenuation: float, 
    omega: float, 
    a: float, 
    b: float, 
    point_estimator=False, 
    seed: int | None = None, 
    tol=1e-9
) -> Tuple[np.ndarray, float]:
    """
    Applies the Gaussian exponential mechanism to a set of samples, computing the probability 
    distribution over intervals between consecutive samples.

    Parameters:
    ----------
    samples : numpy.ndarray
        The array of samples to which the Gaussian exponential mechanism will be applied. 
        These are typically preprocessed data points used to create intervals.

    quantile : float
        The value of the quantile, derived as `samples[qn]`, where `qn` is the index corresponding 
        to the q-th quantile. This is mainly used for debugging.

    sigma_gexp : float
        The standard deviation of the truncated normal distribution used to choose values from the selected intervals.

    q : float
        The quantile of interest, specified as a value between 0 and 1. For example, q=0.5 
        corresponds to the median.

    epsilon : float
        The privacy budget parameter, controlling the level of differential privacy. Smaller 
        epsilon values enforce stricter privacy but may reduce utility.

    eps_mu : float
        A parameter used to enhance the probability of selecting the maximum utility centered 
        interval. Adjusting this can modify how intervals around the quantile are prioritized.

    attenuation : float
        The attenuation factor that controls the influence of larger intervals. Higher attenuation 
        reduces the likelihood of selecting intervals with large widths.

    omega : float
        A parameter for the Gaussian distribution used in the mechanism, affecting the spread of 
        probabilities across intervals.

    a : float
        The lower bound for the range of the mechanism. This defines the minimum allowable value 
        for interval selection.

    b : float
        The upper bound for the range of the mechanism. This defines the maximum allowable value 
        for interval selection.

    point_estimator : bool, optional (default=False)
        If set to True, the function returns a point estimator for the target quantile 
        instead of a probability distribution over intervals.

    seed : int or None, optional (default=None)
        The random seed for reproducibility. Setting a specific seed ensures consistent 
        probabilistic outputs across runs.

    tol : float, optional (default=1e-9)
        The tolerance level for numerical computations, used to handle edge cases or prevent 
        division by zero.

    Returns:
    -------
    (probabilities, point_estimator) : (numpy.ndarray, float)
        - probabilities: A probability distribution over the intervals between consecutive 
          samples (2n - 1 intervals in total). Probabilities are influenced by the proximity 
          of the intervals to the quantile.
        - point_estimator: A point value representing the quantile of interest, sampled based 
          on the computed probabilities.


    """



    sorted_samples = prepare_samples(samples, q, a, b, omega + tol)
    n = len(sorted_samples)
    probabilities = [0 for _ in range(2*n - 1)]
    normalization =  np.sqrt(2*np.pi) * sigma_gexp * (stats.norm.cdf(omega / sigma_gexp) - stats.norm.cdf(-omega / sigma_gexp))

    spent = max(np.log(normalization / 2*omega), eps_mu + attenuation + np.log(2* omega / normalization), omega**2 / (2 * sigma_gexp ** 2) + eps_mu + attenuation / 2, omega**2 / (2 * sigma_gexp ** 2) + np.log(normalization / 2*omega) , attenuation)
    
    #print((normalization, np.log(normalization / 2*omega), eps_mu + attenuation + np.log(2* omega / normalization), omega**2 / (2 * sigma_gexp ** 2) + eps_mu + attenuation / 2, omega**2 / (2 * sigma_gexp ** 2) + np.log(normalization / 2*omega) , attenuation))
    rem = epsilon - spent
    if len(quantile) == 1:
        assert quantile == sorted_samples[math.ceil(q * (n-2))]

    if rem < 0:
        print(('Log remaining', rem, np.log(normalization / 2*omega), eps_mu + attenuation + np.log(2* omega / normalization), omega**2 / (2 * sigma_gexp ** 2) + eps_mu + attenuation / 2, omega**2 / (2 * sigma_gexp ** 2) + np.log(normalization / 2*omega) , attenuation))

    assert rem >=0, "You need to increase the privacy budget"

    fact_a = attenuation / 2

    for i in range(len(sorted_samples)):

        r_i = np.abs(i + 1 - (math.ceil(q * (n-2)) + 1))
        util = -(rem * r_i) / 2


        if r_i == 0:
            #print(f"Amplified target {sorted_samples[i]}")
            #assert (i + 1) % 2 != 0
            probabilities[2*i] = util + eps_mu + np.log(2*omega) 
        
        else: 
            probabilities[2*i] =  util + np.log(2*omega) 

        if i + 1 != len(sorted_samples):
            assert np.abs(sorted_samples[i+1] - 2*omega - sorted_samples[i]) >= 0
            probabilities[2*i + 1] = np.log(np.abs(sorted_samples[i+1] - 2*omega - sorted_samples[i]))  +  util - fact_a


    quantile_estimator = None

    if point_estimator == True:
        np.random.seed(seed=seed)
        r_probabilities = probabilities + np.random.gumbel(loc=0.0, scale=1.0, size=len(probabilities))
        gexp_sample_index= np.argmax(r_probabilities)
        idx_sample = math.ceil((gexp_sample_index + 1) / 2) - 1

        if (gexp_sample_index + 1) % 2 == 0: 
            quantile_estimator = np.random.uniform(sorted_samples[idx_sample] + omega , sorted_samples[idx_sample+1] - omega)

        else: 

            truncated_normal = truncnorm(- omega / sigma_gexp, omega / sigma_gexp, loc=sorted_samples[idx_sample], scale=sigma_gexp)
            quantile_estimator = truncated_normal.rvs(1)[0]
            assert quantile_estimator >= sorted_samples[idx_sample] - omega and quantile_estimator <=  sorted_samples[idx_sample] + omega


    return probabilities, quantile_estimator, sorted_samples
