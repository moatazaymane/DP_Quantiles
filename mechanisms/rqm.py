import numpy as np
import math
from typing import Tuple
from utils.preprocessing import prepare_samples



def compute_log_delta(measurement_noise_std, omega):
    term1 = .5 * np.log(2 / np.pi) 
    term2 = np.log(measurement_noise_std / (omega)) 
    term3 = - (omega)**2 / (2*measurement_noise_std**2) 

    print(f'TERMS log delta {(term1, term2, term3)}')
    return term1 + term2 + term3


def compute_delta(measurement_noise_std, omega):
    term1 = np.sqrt(2 / np.pi)
    term2 = measurement_noise_std / (omega)
    exponent = - (omega) ** 2 / measurement_noise_std ** 2
    term3 = np.exp(exponent)
    result = term1 * term2 * term3
    return result

def randomized_data_quantile_mechanism(
    perturbed_samples: np.ndarray, 
    delta: float, 
    measurement_noise_std: float, 
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
    tol=1e-9, 
    perturbed_target=True,
    verbose=False
) -> Tuple[np.ndarray, float]:
    """
    Applies the randomized data quantile mechanism, computing the probability distribution 
    over intervals between consecutive perturbed samples.

    Parameters:
    ----------
    perturbed_target : float
        The target value (e.g., the quantile value) perturbed with Gaussian noise.

    perturbed_samples : numpy.ndarray
        An array of samples perturbed with Gaussian noise. These can be averaged across 
        independent measurements or sampled from a normal distribution 
        \(N(0, \sigma^2 / m^2)\), where \(\sigma\) is the standard deviation.

    delta : float
        A privacy parameter ensuring differential privacy for the mechanism. Smaller delta 
        values indicate stronger privacy guarantees.

    measurement_noise_std : float
        The standard deviation of the Gaussian noise added to measurements, controlling 
        the amount of randomness introduced during perturbation.

    quantile : float
        The value of the quantile, derived as `perturbed_samples[qn]`, where `qn` is the index 
        corresponding to the q-th quantile. This is mainly used for debugging or validation.

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
        A parameter for the Gaussian exponential mechanism, affecting the spread of probabilities 
        across intervals.

    a : float
        The lower bound for the range of the mechanism. This defines the minimum allowable value 
        for interval selection.

    b : float
        The upper bound for the range of the mechanism. This defines the maximum allowable value 
        for interval selection.

    point_estimator : bool, optional (default=False)
        If set to True, the function returns a point estimator for the target quantile instead 
        of a probability distribution over intervals.

    seed : int or None, optional (default=None)
        The random seed for reproducibility. Setting a specific seed ensures consistent 
        probabilistic outputs across runs.

    tol : float, optional (default=1e-9)
        The tolerance level for numerical computations, used to handle edge cases or prevent 
        division by zero.

    perturbed_target : bool, optional (default=True)
        If set to True, the target value (quantile) is assumed to be perturbed with Gaussian 
        noise. Otherwise, the mechanism operates on the unperturbed target.

    Returns:
    -------
    (probabilities, point_estimator) : (numpy.ndarray, float)
        - probabilities: A probability distribution over the intervals between consecutive 
          perturbed samples (2n - 1 intervals in total). Probabilities are influenced by the 
          proximity of the intervals to the perturbed quantile.
        - point_estimator: A point value representing the quantile of interest, sampled based 
          on the computed probabilities.

    """

    _log_delta =  compute_log_delta(measurement_noise_std = measurement_noise_std, omega=omega)  
    #_delta =  compute_delta(measurement_noise_std = measurement_noise_std, omega=omega)  


    #print(f'MAX DELTA : {np.exp(_log_delta)}')
    assert _log_delta <= np.log(delta), "You need to further decreate the ratio sigma / d"


    exists = False

    sorted_samples = prepare_samples(perturbed_samples, q, a, b, omega + tol)
    n = len(sorted_samples)

    if perturbed_target and len(quantile) == 1:
        assert quantile == sorted_samples[math.ceil(q * (n-2))]


    n = len(sorted_samples)
    #assert n%2 != 0
    probabilities = [0 for _ in range(2*n - 1)]
    normalization =  np.sqrt(2*np.pi) * measurement_noise_std

    spent = max(eps_mu + attenuation + np.log((2 * omega) / normalization), (omega**2) / (2 * measurement_noise_std ** 2) + eps_mu + attenuation / 2, (omega**2) / (2 * measurement_noise_std ** 2) + np.log(normalization / (2 * omega)) , attenuation)
    rem = epsilon - spent
    #print(("MIX", np.log((2 * omega) / normalization), np.log(normalization / (2 * omega)), (2 * omega **2) / measurement_noise_std ** 2))
    
    #print(f"REM RQM , {(epsilon, rem)}")

    #if rem < 0:
    #print(('LOG Assertion  1', _log_delta, np.log(delta), _delta, delta))
    #print(f"Spent MAX {(eps_mu + attenuation + np.log((2 * omega) / normalization), (omega**2) / (2 * measurement_noise_std ** 2) + eps_mu + attenuation / 2, (omega**2) / (2 * measurement_noise_std ** 2) + np.log(normalization / (2 * omega)) , attenuation)}")

    assert rem >=0, "You need to increase the privacy budget"

    fact_a = attenuation / 2

    for i in range(len(sorted_samples)):

        r_i = np.abs(i + 1 - (math.ceil(q * (n-2)) + 1))
        util = -(rem * r_i) / 2

        if len(quantile) == 1:
            assert quantile == sorted_samples[math.ceil(q * (n-2))]

        if r_i == 0:
            exists = True
            probabilities[2*i] = util + eps_mu + np.log(2*omega) 
        
        else: 
            probabilities[2*i] =  util + np.log(2*omega) 

        if i + 1 != len(sorted_samples):
            assert np.abs(sorted_samples[i+1] - 2*omega - sorted_samples[i]) >= 0
            probabilities[2*i + 1] = np.log(np.abs(sorted_samples[i+1] - 2*omega - sorted_samples[i]))  +  util - fact_a


    quantile_estimator = None
    assert exists
    if point_estimator == True:
        np.random.seed(seed=seed)
        r_probabilities = probabilities + np.random.gumbel(loc=0.0, scale=1.0, size=len(probabilities))
        gexp_sample_index= np.argmax(r_probabilities)
        idx_sample = math.ceil((gexp_sample_index + 1) / 2) - 1
        r_i = np.abs(idx_sample + 1 - (math.ceil(q * (n-2)) + 1))

        if (gexp_sample_index + 1) % 2 == 0 or r_i != 0: 
            mn, mx = min(sorted_samples[idx_sample+1] - omega, sorted_samples[idx_sample] + omega), max(sorted_samples[idx_sample+1] - omega, sorted_samples[idx_sample] + omega)
            quantile_estimator = np.random.uniform(mn , mx)

        else: 

            if verbose:
                print(f'RETURNING ACTUAL R_I = {r_i}')
            quantile_estimator = sorted_samples[idx_sample]

            if perturbed_target and len(quantile) == 1:
                assert quantile == quantile_estimator


    return probabilities, quantile_estimator, sorted_samples
