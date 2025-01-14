import numpy as np
from experiments.experiment import run_seed_experiment
import matplotlib.pyplot as plt
import warnings
from experiments.config import prepare_dict
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    n = 10001
    a, b = -500, 500
    gap_q = .109
    num_seeds = 50
    num_q = 1
    epsilon_values = np.array([4])  # Fixed value for epsilon
    plot = False
    min_dist = 0  # additional distance created between the sample quantile and the value to its right
    eval_ubexp = True
    distribution = 'uniform'
    plot = True

    point_estimator = True

    if distribution == 'uniform':
        args = args_uniform = {'a': a, 'b': b, 'method': 'uniform'}

    elif distribution == 'truncated_normal':
        args = {'a': a, 'b': b, 'method': 'truncated_normal', 'mean': 0, 'sigma': 5}

    # Prepare lists to collect MAEs for each seed
    maes_exp_all_seeds = []
    maes_ubexp_all_seeds = []

    epsilon_hyperparams = prepare_dict(n=n, num_q=num_q, pertubed_data=False)

    # Collect MAEs for each seed at fixed epsilon
    for epsilon in epsilon_values:
        hyperparams = epsilon_hyperparams[epsilon]

        maes_exp = []
        maes_ubexp = []

        for seed in range(1, num_seeds + 1):
            result = run_seed_experiment(seed=seed, n=n, args=args,
                    hyperparameters_exp=hyperparams['hyperparameters_exp'], 
                    hyperparameters_ubexp=hyperparams['hyperparameters_ubexp'], 
                    hyperparameters_cexp=hyperparams['hyperparameters_cexp'], 
                    hyperparameters_gexp=hyperparams['hyperparameters_gexp'], 
                    min_dist=min_dist,
                    a=a, b=b, epsilon=epsilon, point_estimator=point_estimator, num_q=num_q, eval_ubexp=eval_ubexp, gap_q=gap_q, distribution=distribution)

            maes_exp.append(result['mae_exp'][list(result['mae_exp'].keys())[0]])
            if eval_ubexp:
                maes_ubexp.append(result['mae_ubexp'][list(result['mae_ubexp'].keys())[0]])

        maes_exp_all_seeds.append(maes_exp)
        maes_ubexp_all_seeds.append(maes_ubexp)

    if plot:
        # Create the boxplot for both EXP and UBEXP as a function of the seeds
        fig, ax = plt.subplots(figsize=(8, 6))

        # Boxplot data: mae_exp and mae_ubexp for each seed
        ax.boxplot([maes_exp_all_seeds[0], maes_ubexp_all_seeds[0]], vert=True, patch_artist=True, 
                   boxprops=dict(facecolor='green', color='green'), 
                   whiskerprops=dict(color='green'), 
                   flierprops=None, 
                   capprops=dict(color='green'), 
                   medianprops=dict(color='green', linewidth=2))
        
        ax.set_xticklabels(['EXP', 'UBEXP'])
        ax.set_ylabel(r'$MAE$', fontsize=14)

        #plt.title('Boxplot of MAE for EXP and UBEXP as a Function of Seeds', fontsize=16)
        plt.yscale('log')
        plt.savefig('figures/boxplot_ubexp.png', dpi=300)

        plt.show()
