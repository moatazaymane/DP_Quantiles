import numpy as np
from utils.order_stats_gap import compute_delta

def prepare_samples(samples: np.ndarray, q: float, a: float, b: float, omega: float, quantile: float | None = None):

    """
    Prepare a set of samples by ensuring a minimum distance of 2 * omega between them (see Definition 4.1 (Centered Exponential Mechanism)).

    Parameters:
    - samples (np.ndarray): Array of sampled points, typically raw sample.
    - q (float): The quantile of interest.
    - a (float): The lower bound of the interval to clip or process the samples.
    - b (float): The upper bound of the interval to clip or process the samples.
    - omega (float): The minimum half-distance between samples. The total enforced distance between samples is 2 * omega.
    - quantile (float, optional): the actual quantile value (samples[ceinqn]) used for validation purposes.

    Returns:
    - np.ndarray: Array of processed samples with enforced spacing.

   """

    npr = len(samples)
    n = npr + 2
    

    X_o = np.zeros(n)
    X_o[1:npr+1] = samples
    X_o[0] = a  
    X_o[-1] = b 

    quantile_pos = int(np.ceil(q * npr)) + 1
    X = np.zeros(n)
    X[quantile_pos - 1] = X_o[quantile_pos - 1] 

    if quantile is not None and len(quantile) == 1:
        assert all(quan == X[quantile_pos - 1] == X_o[quantile_pos - 1] for quan in quantile), \
            "Not all quantiles match the expected values."

    for k in range(0, n - quantile_pos):
        t_plus = max(2 * omega, X_o[quantile_pos + k] - X[quantile_pos + k - 1])
        X[quantile_pos + k] = X[quantile_pos + k - 1] + t_plus

    for k in range(1, quantile_pos):
        t_minus = max(2 * omega, X_o[quantile_pos - k] - X_o[quantile_pos - k - 1])
        X[quantile_pos - 1 - k] = X[quantile_pos - k] - t_minus

    assert X[quantile_pos - 1] == X_o[quantile_pos - 1] 

    return X


def prepare_samples_ubexp(samples: np.ndarray, q: float,n: int, b: float, gap_q: float | None =  None, delta: float = None, distribution :str = None, tol = 1e-9, sigma: float | None =None):

    """
    Prepares samples using additional constraints, including enforcing gaps, 
    working with quantiles, and incorporating distribution-specific adjustments.

    Parameters:
    - samples (np.ndarray): Array of raw samples, typically drawn from some distribution.
    - q (float): quantile of interest (0, 1).
    - n (int): Number of samples to retain after processing.
    - b (float): The upper bound of the interval for the samples. The lower bound is implicitly assumed to be 0.
    - gap_q (float, optional):  only needed if the data distribution is either (uniform or truncated normal), Minimum gap or separation required between samples based on quantile constraints.
    - delta (float, optional): A threshold for additional constraints or adjustments (e.g., privacy guarantees).
    - distribution (str, optional): only needed if the data distribution is either (uniform or truncated normal) 
      for applying distribution-specific rules.
    - tol (float, optional): Tolerance for numerical computations, default is 1e-9.
    - sigma (float, optional): if the data distribution is a truncated normal (ensures that the added gap_q is small).

    Returns:
    - np.ndarray: Array of processed samples after applying constraints and adjustments.
    """

    if gap_q == None:
        gap_q = 2*b

    else:
        _delta = compute_delta(b=b, n=n, q=q, gap_q = gap_q, distribution = distribution, verbose = False, sigma=sigma)

        if delta:
            assert _delta <= delta

    npr = len(samples)
    X_o = np.sort(samples)
    quantile_pos_o = int(np.ceil(q * npr))
    quantile = X_o[quantile_pos_o - 1]

    mn_q = min(quantile_pos_o - 1, npr - quantile_pos_o)
    new_array = X_o[quantile_pos_o - 1 - mn_q:quantile_pos_o + mn_q]

    assert len(new_array) == 2*mn_q + 1
    X = np.concatenate([[-b], new_array, [b]])

    assert np.array_equal(X, np.sort(X)), "The array is not sorted."

    quantile_pos_new = mn_q + 1#0-based indexing rank mn_q + 2

    for k in range(1, mn_q + 2):

        t_k = max(X[quantile_pos_new] - X[quantile_pos_new - k], 
                  X[quantile_pos_new + k] - X[quantile_pos_new])
        
        assert t_k > 0 and gap_q > 0
        X[quantile_pos_new + k] = X[quantile_pos_new] + k* gap_q + t_k
        X[quantile_pos_new - k] = X[quantile_pos_new] - k * gap_q - t_k
        #X[quantile_pos_new - k] =  2 * X[quantile_pos_new] - X[quantile_pos_new + k]
        assert X[quantile_pos_new + k] >= X[quantile_pos_new + k - 1]
        assert np.isclose(X[quantile_pos_new + k] + X[quantile_pos_new - k], 2 *X[quantile_pos_new], atol=tol)
    

    assert np.isclose(X[quantile_pos_new], quantile, atol=tol)
    return X, mn_q
