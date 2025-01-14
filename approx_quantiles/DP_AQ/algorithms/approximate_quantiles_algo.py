# the script is adapted from https://github.com/ShacharSchnapp/DP_AQ/blob/master/algorithms/approximate_quantiles_algo.py

import numpy as np
from mechanisms.cexp import centered_exponential_mechanism
from mechanisms.gexp import gaussian_exponential_mechanism
from mechanisms.ubexp import unbiased_exponential_mechanism
from mechanisms.exp import exponential_mechanism
from mechanisms.rqm import randomized_data_quantile_mechanism

def get_epsilon(n_quantiles, epsilon, swap, cdp):
    layers = np.log2(n_quantiles) + 1

    if swap:
        composition = 2 * layers
    else:
        composition = layers

    if cdp:
        epsilon = np.sqrt((2 * epsilon) / composition)
    else:
        epsilon = epsilon / composition

    return epsilon


def split_by_number(array, m):
    return array[array <= m], array[array >= m]

def approximate_quantiles_algo(seed, samples, array, quantiles, bounds, epsilon, swap=False, cdp=False, method='centered', gap_q=None, distribution = None, sigma_gexp=None, omega=None, point_estimator=True, delta=None, attenuation=None, measurement_error=None, eps_mu=None, target = None, perturbed_target=True, sigma=False, verbose=False):
    epsilon = get_epsilon(len(quantiles), epsilon, swap, cdp)
    
    if verbose:
        print(('IND EPS', epsilon))
    quantiles = np.array(quantiles)
    
    def algo_helper(array, quantiles, bounds):
        m = len(quantiles)
        a, b = bounds
        if m == 0:
            return []

        q_mid = quantiles[m // 2]

        # Select the appropriate method
        if method == 'centered':
            _, v, _ = centered_exponential_mechanism(a=a, b=b, seed=seed, attenuation=attenuation, eps_mu=eps_mu, samples=samples, q=q_mid, epsilon=epsilon, omega=omega, point_estimator=point_estimator, quantile=target)
        elif method == 'gaussian':
            _, v, _ = gaussian_exponential_mechanism(a=a, b=b, sigma_gexp=sigma_gexp, seed=seed, attenuation=attenuation, eps_mu=eps_mu, samples=samples, q=q_mid, epsilon=epsilon, omega=omega, point_estimator=point_estimator, quantile=target)
        elif method == 'unbiased':
            _, v, _ = unbiased_exponential_mechanism(sigma=sigma,b=b, seed=seed, gap_q=gap_q, distribution=distribution, samples=samples, q=q_mid, epsilon=epsilon, point_estimator=point_estimator, quantile=target)
        elif method == 'randomized_data':
            _, v, _ = randomized_data_quantile_mechanism(a=a, b=b, delta=delta, measurement_noise_std=measurement_error, seed=seed, attenuation=attenuation, eps_mu=eps_mu, perturbed_samples=samples, q=q_mid, epsilon=epsilon, omega=omega, point_estimator=point_estimator, quantile=target, perturbed_target=perturbed_target)
        else:
            # Default to the exponential mechanism if no specific method is specified
            _, v, _ = exponential_mechanism(a=a, b=b, seed=seed, samples=samples, q=q_mid, epsilon=epsilon, point_estimator=point_estimator, quantile=target)

        if m == 1:
            return [v]  # Return v as a list for concatenation purposes
        
        d_l, d_u = split_by_number(array, v)
        q_l, q_u = np.array_split(quantiles[quantiles != q_mid], 2)
        q_l, q_u = q_l / q_mid, (q_u - q_mid) / (1 - q_mid)

        return algo_helper(d_l, q_l, (a, v)) + [v] + algo_helper(d_u, q_u, (v, b))
    return algo_helper(np.sort(array), quantiles, bounds)
