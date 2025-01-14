import numpy as np
import math
from typing import Tuple
from utils.preprocessing import prepare_samples


def centered_exponential_mechanism(
    samples: np.ndarray, 
    quantile, 
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
    Parameters:
    ----------
    samples : numpy.ndarray
        The array of samples to which the centered exponential mechanism will be applied.
        These are the original data points.

    quantile : float
        The actual quantile samples_{([qn])}.

    q : float
        The specific quantile value (between 0 and 1) of interest. This determines the target 
        quantile for the exponential mechanism.

    epsilon : float
        The privacy budget parameter, which controls the level of differential privacy. Smaller 
        values of epsilon provide stronger privacy guarantees.

    eps_mu : float
        A parameter used to enhance the probability of selecting the maximum utility centered 
        interval. Adjusting this can modify how intervals around the quantile are prioritized.

    attenuation : float
        The attenuation factor controls the influence of larger intervals. Higher attenuation 
        reduces the probability of choosing intervals with large widths.

    omega : float
        Parameter controls the legth of the centered intervals, affecting 
        the spread of probabilities across intervals.

    a : float
        The lower bound for the range of the mechanism. This defines the minimum allowable 
        value for the interval selection.

    b : float
        The upper bound for the range of the mechanism. This defines the maximum allowable 
        value for the interval selection.

    point_estimator : bool, optional (default=False)
        If set to True, the function returns a point estimator for the target quantile instead 
        of a probability distribution over intervals.

    seed : int or None, optional (default=None)
        The random seed for reproducibility. Setting a specific seed ensures consistent 
        probabilistic outputs across runs.

    tol : float, optional (default=1e-9)
        The tolerance level for numerical computations. This is used to handle edge cases or 
        prevent division by zero in calculations.

    Returns:
    -------
    (probabilities, point_estimator) : (numpy.ndarray, float)
        - probabilities: A probability distribution over the intervals between consecutive samples 
          (2n - 1 intervals in total). Probabilities reflect the likelihood of selecting each 
          interval based on the exponential mechanism.
        - point_estimator: A point value representing the quantile of interest.
    """


    sorted_samples = prepare_samples(samples, q, a, b, omega + tol)
    n = len(sorted_samples)
    probabilities = [0 for _ in range(2*n - 1)]
    exists = False

    if len(quantile) == 1:
        # we want to estimate x_{ceil(qn)} we use n-2 because the endpoints a, b are automatically added to the samples (we also use zero based indexing)
        new_q_rank = math.ceil(q * (n-2)) + 1
        assert quantile == sorted_samples[new_q_rank - 1]

    rem = epsilon - eps_mu - attenuation

    assert rem >=0, "You need to increase the privacy budget"

    fact_a = attenuation / 2

    for i in range(len(sorted_samples)):

        r_i = np.abs(i + 1 - (math.ceil(q * (n-2)) + 1))
        util = -(rem * r_i) / 2


        if r_i == 0:
            exists = True
            probabilities[2*i] = util + eps_mu + np.log(2*omega)  #eps_mu increases the probability of the maximum utility interval

        else: 
            probabilities[2*i] =  util + np.log(2*omega) 

        if i + 1 != len(sorted_samples):
            assert np.abs(sorted_samples[i+1] - 2*omega - sorted_samples[i]) >= 0
            probabilities[2*i + 1] = np.log(np.abs(sorted_samples[i+1] - 2*omega - sorted_samples[i]))  +  util - fact_a

    assert exists and len(probabilities) == 2*n - 1, "No interval corresponds exactly to the quantile."

    quantile_estimator = None

    if point_estimator == True:
        np.random.seed(seed=seed)
        r_probabilities = probabilities + np.random.gumbel(loc=0.0, scale=1.0, size=len(probabilities))

        cexp_sample_index = np.argmax(r_probabilities)
        idx_sample = math.ceil((cexp_sample_index + 1) / 2) - 1

        if (cexp_sample_index + 1) % 2 == 1: 
            quantile_estimator = np.random.uniform(sorted_samples[idx_sample] - omega, sorted_samples[idx_sample] + omega)
        else:
            
            quantile_estimator = np.random.uniform(sorted_samples[idx_sample] + omega, sorted_samples[idx_sample + 1] - omega)


    return probabilities, quantile_estimator, sorted_samples
