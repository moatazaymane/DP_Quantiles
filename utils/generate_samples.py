import numpy as np
import math
from scipy.stats import truncnorm


def generate_samples(fixed_seed, n, gauss_mech_std, args, changing_seed, q, perturbed_target=True, min_dist_all=0, min_dist=0):
    """
    Generates samples from a specified distribution and applies optional perturbations and constraints.

    Parameters:
    - fixed_seed (int): Seed for reproducibility when generating the initial samples.
    - n (int): The number of samples to generate.
    - gauss_mech_std (float): Standard deviation of the Gaussian noise to be added as perturbation.
    - args (dict): A dictionary specifying the distribution and its parameters:
        - 'method' (str): The type of distribution ('norm', 'cauchy', 'lognorm', 'uniform', etc.).
        - 'mean' (float): Mean of the distribution (for applicable methods).
        - 'sigma' (float): Standard deviation (or scale) of the distribution (for applicable methods).
        - 'a', 'b' (float): Lower and upper bounds of the distribution (for uniform or truncated distributions).
        - 'min_dist' (float): Minimum distance constraint for samples in the 'uniform2' method.
    - changing_seed (int): Seed for reproducibility when generating Gaussian perturbations.
    - q (list[float]): List of quantile values (between 0 and 1) to compute from the generated samples.
    - perturbed_target (bool, optional): If True, updates the target quantiles after applying perturbations. Defaults to True.
    - min_dist_all (float, optional): Minimum distance constraint applied globally between consecutive samples. Defaults to 0.
    - min_dist (float, optional): Minimum distance constraint applied between quantiles and the next value. Defaults to 0.

    Returns:
    - samples (numpy.ndarray): The processed and perturbed array of samples.
    - target (numpy.ndarray): The values corresponding to the specified quantiles.
    """
    
    np.random.seed(seed=fixed_seed)
    if args['method'] == 'norm':
        samples = np.random.normal(args['mean'], args['sigma'], size=n)
    elif args['method'] == 'cauchy':
        samples = args['mean'] + args['sigma'] * np.random.standard_cauchy(size=n)
    elif args['method'] == 'lognorm':
        samples = np.random.lognormal(args['mean'], args['sigma'], size=n)
    elif args['method'] == 'uniform':
        samples = np.random.uniform(low=args['a'], high=args['b'], size=n)
    elif args['method'] == 'uniform2':
        min_dist = args['min_dist']
        samples1 = np.random.uniform(low=args['a'], high=args['b'] - 2 * min_dist, size=math.ceil(n / 2))
        samples2 = np.random.uniform(low=args['b'] - min_dist, high=args['b'], size=math.floor(n / 2))
        samples = np.concatenate((samples1, samples2))
        assert len(samples) == n

    elif args['method'] == 'truncated_normal':
        a, b = args['a'], args['b']
        mean, sigma = args['mean'], args['sigma']
        lower, upper = (a - mean) / sigma, (b - mean) / sigma
        samples = truncnorm.rvs(lower, upper, loc=mean, scale=sigma, size=n, random_state=fixed_seed)
    
    
    samples = np.sort(samples)
    quantile_indices = [math.ceil(qn * n) - 1 for qn in q]
    target = np.array([samples[idx] for idx in quantile_indices])
    

    # we add a minimum distance between quantile and the next value (if min_dist = 0 | this can be used to illustrate the theoretical results of lemma)
    for idx in quantile_indices:

        j = idx + 1

        while j < n and samples[j] - samples[idx] < min_dist:
            samples[j] = samples[idx] + min_dist
            j+=1

    for i in range(1, len(samples)):
        if samples[i] - samples[i - 1] < min_dist_all:
            samples[i] = samples[i - 1] + min_dist_all


    if gauss_mech_std != 0:
        np.random.seed(seed=changing_seed)
        perturbation = np.random.normal(loc=0, scale=gauss_mech_std, size=n)
        perturbed_samples = samples + perturbation
        samples = np.sort(perturbed_samples)



    if perturbed_target:
        target = np.array([samples[idx] for idx in quantile_indices])

    samples = np.clip(samples, args['a'], args['b'])

    return samples, target
