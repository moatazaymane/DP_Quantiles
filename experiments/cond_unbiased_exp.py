import numpy as np
from experiments.experiment import run_seed_experiment
import matplotlib.pyplot as plt
import warnings
from experiments.config import prepare_dict
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    n = 1001
    a, b = -1000, 80000
    num_seeds = 200
    num_q = 1
    epsilon_values = np.arange(1, 11, 1)
    measurement_error = 0 # if it is non zero data will be perturbed with gaussian noise and the mechanism rqm (definition ) will be evaluated
    plot = False
    min_dist = 100 # additional distance created between the sample quantile and the value to its right

    point_estimator = True
    args_norm = {'mean': 30000, 'sigma': 8000, 'method': 'norm', 'a': a, 'b': b}
    args_lognorm = {'mean': 10, 'sigma': .4, 'method': 'lognorm', 'a': a, 'b': b}
    args_uniform = {'a': 0, 'b': 20000, 'method': 'uniform'}
    args_cauchy = {'mean': 30000, 'sigma': 8000, 'method': 'cauchy', 'a': a, 'b': b}

    mean_maes_gexp = []
    mean_maes_exp = []
    mean_maes_cexp = []
    mean_maes_rqm = []  

    std_maes_gexp = []
    std_maes_exp = []
    std_maes_cexp = []
    std_maes_rqm = [] 

    epsilon_hyperparams = prepare_dict(n=n, num_q = num_q, pertubed_data = measurement_error != 0)
    averaged_results = {epsilon: {'mae_cexp': {}, 'mae_exp': {}, 'mae_gexp': {}, 'mae_rqm': {}} for epsilon in epsilon_hyperparams}


    for epsilon in epsilon_values:
        if epsilon!=5:
            continue
        hyperparams = epsilon_hyperparams[epsilon]
        averaged_mae_cexp = {hyperparam[0]: 0 for hyperparam in hyperparams['hyperparameters_cexp']}
        averaged_mae_exp = {hyperparam[0]: 0 for hyperparam in hyperparams['hyperparameters_exp']}
        averaged_mae_gexp = {hyperparam[0]: 0 for hyperparam in hyperparams['hyperparameters_gexp']}
        averaged_mae_rqm = {hyperparam[0]: 0 for hyperparam in hyperparams['hyperparameters_rqm']}  

        maes_gexp = []
        maes_exp = []
        maes_cexp = []
        maes_ubexp = []
        maes_rqm = []
        
        for seed in range(1, num_seeds + 1):
            if measurement_error:
                result = run_seed_experiment(seed=seed, n=n, args=args_lognorm, 
                    hyperparameters_exp=hyperparams['hyperparameters_exp'], 
                    hyperparameters_cexp=hyperparams['hyperparameters_cexp'],                    hyperparameters_ubexp=hyperparams['hyperparameters_ubexp'], 
 
                    hyperparameters_gexp=hyperparams['hyperparameters_gexp'],
                    hyperparameters_rqm=hyperparams['hyperparameters_rqm'], perturbed_target=True,
                    a=a, b=b, epsilon=epsilon, point_estimator=point_estimator, measurement_error=measurement_error, num_q = num_q)
            else:
                result = run_seed_experiment(seed=seed, n=n, args=args_uniform,
                    hyperparameters_exp=hyperparams['hyperparameters_exp'], 
                    hyperparameters_ubexp=hyperparams['hyperparameters_ubexp'], 
                    hyperparameters_cexp=hyperparams['hyperparameters_cexp'], 
                    hyperparameters_gexp=hyperparams['hyperparameters_gexp'], 
                    min_dist= min_dist,
                    a=a, b=b, epsilon=epsilon, point_estimator=point_estimator, num_q = num_q)

            for hyperparam in hyperparams['hyperparameters_cexp']:
                averaged_mae_cexp[hyperparam[0]] += result['mae_cexp'][hyperparam[0]]
            for hyperparam in hyperparams['hyperparameters_exp']:
                averaged_mae_exp[hyperparam[0]] += result['mae_exp'][hyperparam[0]]

   
            for hyperparam in hyperparams['hyperparameters_gexp']:
                averaged_mae_gexp[hyperparam[0]] += result['mae_gexp'][hyperparam[0]]

            if measurement_error:
                for hyperparam in hyperparams['hyperparameters_rqm']: 
                    averaged_mae_rqm[hyperparam[0]] += result['mae_rqm'][hyperparam[0]]

            maes_exp.append(result['mae_exp'][hyperparam[0]])
            maes_gexp.append(result['mae_gexp'][hyperparam[0]])
            maes_cexp.append(result['mae_cexp'][hyperparam[0]])

            if measurement_error: 
                maes_rqm.append(result['mae_rqm'][hyperparam[0]])  

        maes_exp = np.array(maes_exp)


        maes_cexp = np.array(maes_cexp)
        maes_gexp = np.array(maes_gexp)
        maes_rqm = np.array(maes_rqm) 

        mean_maes_exp.append(np.mean(maes_exp))
        mean_maes_gexp.append(np.mean(maes_gexp))
        mean_maes_cexp.append(np.mean(maes_cexp))

        if measurement_error: 
            mean_maes_rqm.append(np.mean(maes_rqm))

        std_maes_cexp.append(np.std(maes_cexp))
        std_maes_gexp.append(np.std(maes_gexp))
        std_maes_exp.append(np.std(maes_exp))

        if measurement_error: 
            std_maes_rqm.append(np.std(maes_rqm))

        print(f"EPSILON {epsilon}")
        print(f"CEXP - EPSILON = {epsilon} - Mean: {np.mean(maes_cexp):.4f}, Std Dev: {np.std(maes_cexp):.4f}")
        print(f"GEXP - EPSILON = {epsilon} - Mean: {np.mean(maes_gexp):.4f}, Std Dev: {np.std(maes_gexp):.4f}")
        print(f"EXP - EPSILON = {epsilon} - Mean: {np.mean(maes_exp):.4f}, Std Dev: {np.std(maes_exp):.4f}")

        if measurement_error: 
            print(f"RQM - EPSILON = {epsilon} - Mean: {np.mean(maes_rqm):.4f}, Std Dev: {np.std(maes_rqm):.4f}")  

    if plot:
        
        
        epsilons = np.array(list(epsilon_hyperparams.keys()))

        plt.plot(epsilons, mean_maes_exp,  label='EXP', color='green')
        plt.fill_between(epsilons, 
                        [m - s for m, s in zip(mean_maes_exp, std_maes_exp)], 
                        [m + s for m, s in zip(mean_maes_exp, std_maes_exp)], alpha=0.05, color='green')

        plt.plot(epsilons, mean_maes_gexp,  label='GEXP', color='orange')
        plt.fill_between(epsilons, 
                        [m - s for m, s in zip(mean_maes_gexp, std_maes_gexp)], 
                        [m + s for m, s in zip(mean_maes_gexp, std_maes_gexp)], alpha=0.05, color='orange')

        plt.plot(epsilons, mean_maes_cexp,  label='CEXP', color='blue')
        plt.fill_between(epsilons, 
                        [m - s for m, s in zip(mean_maes_cexp, std_maes_cexp)], 
                        [m + s for m, s in zip(mean_maes_cexp, std_maes_cexp)], alpha=0.05, color='blue')

        if measurement_error:
            plt.plot(epsilons, mean_maes_rqm,  label='RQM', color='purple')
            plt.fill_between(epsilons, 
                            [m - s for m, s in zip(mean_maes_rqm, std_maes_rqm)], 
                            [m + s for m, s in zip(mean_maes_rqm, std_maes_rqm)], alpha=0.3, color='purple')

        plt.legend(loc='best')
        plt.xlabel(r"$\epsilon$")
        plt.ylabel(r'$MAE \, (x_{(\lceil qn \rceil + 1)} - x_{(\lceil qn \rceil)} \geq 100)$')
        plt.yscale('log')
        #plt.savefig('figures/cond_unbiased_lognormal_many_min_dist.png', dpi=300)
        plt.show()
