from approx_quantiles.DP_AQ.algorithms.approximate_quantiles_algo import approximate_quantiles_algo
from utils.generate_samples import generate_samples
import numpy as np


def run_seed_experiment(seed,  args, n, hyperparameters_exp, num_q, hyperparameters_ubexp, hyperparameters_cexp, hyperparameters_gexp, a, b, epsilon, point_estimator, gap_q = None, distribution = None, hyperparameters_rqm=None, perturbed_target=True, measurement_error=0, eval_method='avg', min_dist_all = 0, min_dist = 0, eval_ubexp=False):

    """
    Runs a single experiment for a given random seed, evaluating quantile mechanisms.

    Parameters:
    - seed (int): Seed for reproducibility of the experiment.
    - args (dict): A dictionary specifying the distribution and its parameters (see `generate_samples` for details).
    - n (int): The number of samples to generate.
    - hyperparameters_exp (list): Hyperparameters for the standard exponential mechanism.
    - num_q (int): Number of quantiles to evaluate.
    - hyperparameters_ubexp (list): Hyperparameters for the unbiased exponential mechanism.
    - hyperparameters_cexp (list): Hyperparameters for the centered exponential mechanism.
    - hyperparameters_gexp (list): Hyperparameters for the Gaussian exponential mechanism.
    - a (float): Lower bound of the sampling range.
    - b (float): Upper bound of the sampling range.
    - epsilon (float): Privacy parameter for the exponential mechanisms.
    - point_estimator (str): The estimator method used in the quantile mechanisms.
    - gap_q (float, optional): Only needed if the data distribution is uniform or truncated normal, Gap adjustment for quantile evaluations in certain mechanisms. Defaults to None.
    - distribution (str, optional): Distribution type for adjustments (if needed). Defaults to None.
    - hyperparameters_rqm (list, optional): Hyperparameters for the randomized quantile mechanism. Defaults to None.
    - perturbed_target (bool, optional): If True, perturbs the target quantiles after applying noise. Defaults to True.
    - measurement_error (float, optional): Standard deviation of Gaussian noise added to the samples. Defaults to 0.
    - eval_method (str, optional): Evaluation metric ('avg' for mean absolute error, 'max' for maximum absolute error). Defaults to 'avg'.
    - min_dist_all (float, optional): Minimum global distance constraint between consecutive samples. Defaults to 0.
    - min_dist (float, optional): Minimum distance constraint between quantiles and their neighbors. Defaults to 0.
    - eval_ubexp (bool, optional): If True, evaluates the unbiased exponential mechanism. Defaults to False.

    Returns:
    - results (dict): A dictionary containing MAE results for each mechanism and the quantiles used in the experiment.
    """

    np.random.seed(seed=seed)
    results = {}
    q = np.sort(np.random.uniform(0.1, 0.9, num_q))
    samples, quantiles = generate_samples(args=args, n=n, fixed_seed=0, gauss_mech_std=measurement_error, changing_seed=seed, q=q, perturbed_target=perturbed_target, min_dist_all=min_dist_all, min_dist=min_dist)

    mae_cexp = {hyperparam[0]: 0 for hyperparam in hyperparameters_cexp}
    mae_exp = {hyperparam[0]: 0 for hyperparam in hyperparameters_exp}
    mae_ubexp = {hyperparam[0]: 0 for hyperparam in hyperparameters_ubexp}
    mae_gexp = {hyperparam[0]: 0 for hyperparam in hyperparameters_gexp}

    method = 'exp' 
    estimator_exp = approximate_quantiles_algo(target=quantiles, seed=seed, samples=samples, array=samples, quantiles=q, bounds=(a, b), epsilon=epsilon, method=method, point_estimator=point_estimator)
    
    mae_exp[hyperparameters_exp[0][0]] = np.max(np.abs(np.sort(estimator_exp) - quantiles)) if eval_method == 'max' else np.mean(np.abs(estimator_exp - quantiles))

    if eval_ubexp:
        # Unbiased Exponential Mechanism
        method = 'unbiased'
        sigma = None
        if 'sigma' in args:
            sigma = args['sigma']
        estimator_ubexp,  = approximate_quantiles_algo(sigma=sigma,target=quantiles, gap_q = gap_q, distribution = distribution, seed=seed, samples=samples, array=samples, quantiles=q, bounds=(a, b), epsilon=epsilon, method=method, point_estimator=point_estimator)
        mae_ubexp[hyperparameters_ubexp[0][0]] = np.max(np.abs(np.sort(estimator_ubexp) - quantiles)) if eval_method == 'max' else np.mean(np.abs(estimator_ubexp - quantiles))

    # Centered Exponential Mechanism
    method = 'centered'
    for hyperparam in hyperparameters_cexp:
        attenuation, eps_mu, omega, bl_cexp = hyperparam[1], hyperparam[2], hyperparam[3], hyperparam[4]
        if bl_cexp:
            mae_cexp[hyperparam[0]] = mae_exp[hyperparameters_exp[0][0]] 
            continue
        else:
            estimator_cexp = approximate_quantiles_algo(target=quantiles, seed=seed, samples=samples, array=samples, quantiles=q, bounds=(a, b), epsilon=epsilon, method=method, attenuation=attenuation, eps_mu=eps_mu, omega=omega, point_estimator=point_estimator)
            mae_cexp[hyperparam[0]] = np.max(np.abs(np.sort(estimator_cexp) - quantiles)) if eval_method == 'max' else np.mean(np.abs(estimator_cexp - quantiles))

    # Gaussian Exponential Mechanism
    method = 'gaussian'
    for hyperparam in hyperparameters_gexp:
        attenuation, eps_mu, omega, sigma_gexp, bl_gexp = hyperparam[1], hyperparam[2], hyperparam[3], hyperparam[4], hyperparam[5]
            
        if bl_gexp:
            mae_gexp[hyperparam[0]] = mae_exp[hyperparameters_exp[0][0]] 
            continue
        else:
            estimator_gexp = approximate_quantiles_algo(seed=seed, target=quantiles, samples=samples, array=samples, quantiles=q, bounds=(a, b), epsilon=epsilon, method=method, attenuation=attenuation, eps_mu=eps_mu, omega=omega, sigma_gexp=sigma_gexp, point_estimator=point_estimator)
            mae_gexp[hyperparam[0]] = np.max(np.abs(np.sort(estimator_gexp) - quantiles)) if eval_method == 'max' else np.mean(np.abs(estimator_gexp - quantiles))

    # Randomized Quantile Mechanism (if provided)
    if hyperparameters_rqm:
        mae_rqm = {hyperparam[0]: 0 for hyperparam in hyperparameters_rqm}
        
        for hyperparam in hyperparameters_rqm:
            attenuation, eps_mu, omega, delta = hyperparam[1], hyperparam[2], hyperparam[3], hyperparam[4]
            method = 'randomized_data'
            estimator_rqm = approximate_quantiles_algo(perturbed_target=perturbed_target,seed=seed, samples=samples, array=samples, quantiles=q, bounds=(a, b), epsilon=epsilon, method=method, attenuation=attenuation, eps_mu=eps_mu, omega=omega, delta=delta, measurement_error=measurement_error, point_estimator=point_estimator, target=quantiles)
            mae_rqm_val = np.abs(np.sort() - quantiles)
            print((mae_rqm_val, estimator_rqm, quantiles))
            mae_rqm[hyperparam[0]] = np.max(mae_rqm_val) if eval_method == 'max' else np.mean(mae_rqm_val)

        results['mae_rqm'] = mae_rqm
    results['mae_exp'] = mae_exp

    if eval_ubexp:
        results['mae_ubexp'] = mae_ubexp
    results['mae_cexp'] = mae_cexp
    results['mae_gexp'] = mae_gexp

    results['quantile'] = quantiles

    return results
