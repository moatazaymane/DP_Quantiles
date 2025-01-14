import numpy as np 

def prepare_dict(n, num_q, pertubed_data = True):

    if  pertubed_data:
        epsilon_hyperparams = {
            1: {
                'hyperparameters_exp': [('1',)],
                'hyperparameters_ubexp': [('1',)], 
                'hyperparameters_cexp': [("1", 0, 0, 1e-200, True)],
                'hyperparameters_gexp': [("1", 0, 0, 1e-200, 1e-200, True)],
                'hyperparameters_rqm': [("1", 0, 0, 25, 1 / n)]
            },
            2: {
                'hyperparameters_exp': [('1',)],
                'hyperparameters_ubexp': [('1',)],  
                'hyperparameters_cexp': [("1", 0, 0, 1e-200, True)],
                'hyperparameters_gexp': [("1", 0, 0, 1e-200, 1e-200, True)],
                'hyperparameters_rqm': [("1", 0, 0, 25, 1 / n)]
            },
            3: {
                'hyperparameters_exp': [('1',)],
                'hyperparameters_ubexp': [('1',)],  
                'hyperparameters_cexp': [("1", 0, 0, 1e-200, True)],
                'hyperparameters_gexp': [("1", 0, 0, 1e-200, 1e-200, True)],
                'hyperparameters_rqm': [("1", 0, 0, 25, 1 / n)]
            },
            4: {
                'hyperparameters_exp': [('1',)],
                'hyperparameters_ubexp': [('1',)],  
                'hyperparameters_cexp': [("1", 0, 0, 1e-200, True)],
                'hyperparameters_gexp': [("1", 0, 0, 1e-200, 1e-200, True)],
                'hyperparameters_rqm': [("1", 0, 0.2, 25, 1 / n)]
            },
            5: {
                'hyperparameters_exp': [('1',)],
                'hyperparameters_ubexp': [('1',)],  
                'hyperparameters_cexp': [("1", 0, 0, 1e-200, True)],
                'hyperparameters_gexp': [("1", 0, 0, 1e-200, 1e-200, True)],
                'hyperparameters_rqm': [("1", 0, 0, 25, 1 / n)]
            },
            6: {
                'hyperparameters_exp': [('1',)],
                'hyperparameters_ubexp': [('1',)],  
                'hyperparameters_cexp': [("1", 0,0, 1, False)],
                'hyperparameters_gexp': [("1", 0, 0.1, 1, 3, False)],
                'hyperparameters_rqm': [("1", 0, 0.4, 40, .054)]
            },
            7: {
                'hyperparameters_exp': [('1',)],
                'hyperparameters_ubexp': [('1',)],  
                'hyperparameters_cexp': [("1", 0,0, 1, False)],
                'hyperparameters_gexp': [("1", 0, 0.1, 1, 3, False)],
                'hyperparameters_rqm': [("1", 0, 0.9, 40, .054)]
            },
            8: {
                'hyperparameters_exp': [('1',)],
                'hyperparameters_ubexp': [('1',)],  
                'hyperparameters_cexp': [("1", 0,6, 10, False)],
                'hyperparameters_gexp': [("1", 0, 0.1, 1, 3, False)],
                'hyperparameters_rqm': [("1", 0, 1.2, 40, .054)]
            },
            9: {
                'hyperparameters_exp': [('1',)],
                'hyperparameters_ubexp': [('1',)],  
                'hyperparameters_cexp': [("1", 0,0, 1, False)],
                'hyperparameters_gexp': [("1", 0, 0.1, 1, 3, False)],
                'hyperparameters_rqm': [("1", 0, 0.4, 40, .054)]
            },
            10: {
                'hyperparameters_exp': [('1',)],
                'hyperparameters_ubexp': [('1',)],  
                'hyperparameters_cexp': [("1", 0,0, 1, False)],
                'hyperparameters_gexp': [("1", 0, 0.1, 1, 3, False)],
                'hyperparameters_rqm': [("1", 0, 0.4, 40, .054)]
            }
        }
        return epsilon_hyperparams
    
    elif num_q == 1:

         epsilon_hyperparams = {
        1: {
            'hyperparameters_exp': [('1',)],
            'hyperparameters_ubexp': [('1',)], 
            'hyperparameters_cexp': [("1", 0, 0, 1e-200, True)],
            'hyperparameters_gexp': [("1", 0, 0, 1e-200, 1e-200, True)],
            'hyperparameters_rqm': [("1", 0, 0, 25, 1 / n)]
        },
        2: {
            'hyperparameters_exp': [('1',)],
            'hyperparameters_ubexp': [('1',)],  
            'hyperparameters_cexp': [("1", 0,0.15, 10, False)],
            'hyperparameters_gexp': [("1", 0, 0, 1e-200, 1e-200, True)],
            'hyperparameters_rqm': [("1", 0, 0.2, 10, .054)]
        },
        3: {
            'hyperparameters_exp': [('1',)],
            'hyperparameters_ubexp': [('1',)],  
            'hyperparameters_cexp': [("1", 0,1.1, 12, False)],
            'hyperparameters_gexp': [("1", 0, 0, 1e-200, 1e-200, True)],
            'hyperparameters_rqm': [("1", 0, 0.2, 10, .054)]
        },
        4: {
            'hyperparameters_exp': [('1',)],
            'hyperparameters_ubexp': [('1',)],  
            'hyperparameters_cexp': [("1", 0,1.9, 10, False)],
            'hyperparameters_gexp': [("1", 0, 0, 1e-200, 1e-200, True)],
            'hyperparameters_rqm': [("1", 0, 0.2, 10, .054)]
        },
        5: {
            'hyperparameters_exp': [('1',)],
            'hyperparameters_ubexp': [('1',)],  
            'hyperparameters_cexp': [("1", 0,3, 5, False)],
            'hyperparameters_gexp': [("1", 0, 3, 5, 20, False)],
            'hyperparameters_rqm': [("1", 0, 0.4, 10, .054)]
        },
        6: {
            'hyperparameters_exp': [('1',)],
            'hyperparameters_ubexp': [('1',)],  
            'hyperparameters_cexp': [("1", 0,4, 5, False)],
            'hyperparameters_gexp': [("1", 0, 4, 5, 20, False)],
            'hyperparameters_rqm': [("1", 0, 0.4, 10, .054)]
        },
        7: {
            'hyperparameters_exp': [('1',)],
            'hyperparameters_ubexp': [('1',)],  
            'hyperparameters_cexp': [("1", 0,5, 5, False)],
            'hyperparameters_gexp': [("1", 0, 5, 5, 20, False)],
            'hyperparameters_rqm': [("1", 0, 0.4, 10, .054)]
        },
        8: {
            'hyperparameters_exp': [('1',)],
            'hyperparameters_ubexp': [('1',)],  
            'hyperparameters_cexp': [("1", 0,6, 5, False)],
            'hyperparameters_gexp': [("1", 0, 6, 5, 20, False)],
            'hyperparameters_rqm': [("1", 0, 0.4, 10, .054)]
        },
        9: {
            'hyperparameters_exp': [('1',)],
            'hyperparameters_ubexp': [('1',)],  
            'hyperparameters_cexp': [("1", 0,7, 5, False)],
            'hyperparameters_gexp': [("1", 0, 7, 5, 20, False)],
            'hyperparameters_rqm': [("1", 0, 0.4, 10, .054)]
        },
        10: {
            'hyperparameters_exp': [('1',)],
            'hyperparameters_ubexp': [('1',)],  
            'hyperparameters_cexp': [("1", 0,8, 5, False)],
            'hyperparameters_gexp': [("1", 0, 8, 5, 20, False)],
            'hyperparameters_rqm': [("1", 0, 0.4, 10, .054)]
        }
    }
         return epsilon_hyperparams
    
    else: 

            epsilon_hyperparams = {
        1: {
            'hyperparameters_exp': [('1',)],
            'hyperparameters_ubexp': [('1',)],  # Added hyperparameters_ubexp
            'hyperparameters_cexp': [("1", 0, 0, 1e-200, True)],
            'hyperparameters_gexp': [("1", 0, 0, 1e-200, 1e-200, True)],
            'hyperparameters_rqm': [("1", 0, 0, 25, 1 / n)]
        },
        2: {
            'hyperparameters_exp': [('1',)],
            'hyperparameters_ubexp': [('1',)],  # Added hyperparameters_ubexp
            'hyperparameters_cexp': [("1", 0,0.01, 5, False)],
            'hyperparameters_gexp': [("1", 0, 0, 1e-200, 1e-200, True)],
            'hyperparameters_rqm': [("1", 0, 0, 25, 1 / n)]
        },
        3: {
            'hyperparameters_exp': [('1',)],
            'hyperparameters_ubexp': [('1',)],  # Added hyperparameters_ubexp
            'hyperparameters_cexp': [("1", 0,0.05, 5, False)],
            'hyperparameters_gexp': [("1", 0, 0, 1e-200, 1e-200, True)],
            'hyperparameters_rqm': [("1", 0, 0, 25, 1 / n)]
        },
        4: {
            'hyperparameters_exp': [('1',)],
            'hyperparameters_ubexp': [('1',)],  # Added hyperparameters_ubexp
            'hyperparameters_cexp': [("1", 0,0.15, 5, False)],
            'hyperparameters_gexp': [("1", 0, 0, 1e-200, 1e-200, True)],
            'hyperparameters_rqm': [("1", 0, 0, 25, 1 / n)]
        },
        5: {
            'hyperparameters_exp': [('1',)],
            'hyperparameters_ubexp': [('1',)],  # Added hyperparameters_ubexp
            'hyperparameters_cexp': [("1", 0,0.25, 5, False)],
            'hyperparameters_gexp': [("1", 0, 0, 1e-200, 1e-200, True)],
            'hyperparameters_rqm': [("1", 0, 0, 25, 1 / n)]
        },
        6: {
            'hyperparameters_exp': [('1',)],
            'hyperparameters_ubexp': [('1',)],  # Added hyperparameters_ubexp
            'hyperparameters_cexp': [("1", 0,.6, 5, False)],
            'hyperparameters_gexp': [("1", 0, 0.1, 1, 3, False)],
            'hyperparameters_rqm': [("1", 0, 1.5, 25, 1 / n)]
        },
        7: {
            'hyperparameters_exp': [('1',)],
            'hyperparameters_ubexp': [('1',)],  # Added hyperparameters_ubexp
            'hyperparameters_cexp': [("1", 0,.6, 5, False)],
            'hyperparameters_gexp': [("1", 0, 0.3, 1, 3, False)],
            'hyperparameters_rqm': [("1", 0, 1, 400, 1 / n)]
        },
        8: {
            'hyperparameters_exp': [('1',)],
            'hyperparameters_ubexp': [('1',)],  # Added hyperparameters_ubexp
            'hyperparameters_cexp': [("1", 0, 1.9, 10, False)],
            'hyperparameters_gexp': [("1", 0, 0.5, 1, 3, False)],
            'hyperparameters_rqm': [("1", 0, 0, 10, 10, 1 / n)]
        },
        9: {
            'hyperparameters_exp': [('1',)],
            'hyperparameters_ubexp': [('1',)],  # Added hyperparameters_ubexp
            'hyperparameters_cexp': [("1", 0, .5, 10, False)],
            'hyperparameters_gexp': [("1", 0, .6, 1, 3, False)],
            'hyperparameters_rqm': [("1", 0, 1, 10, 10, 1 / n)]
        },
                10: {
            'hyperparameters_exp': [('1',)],
            'hyperparameters_ubexp': [('1',)],  # Added hyperparameters_ubexp
            'hyperparameters_cexp': [("1", 0, .6, 10, False)],
            'hyperparameters_gexp': [("1", 0, 0.8, 1, 3, False)],
            'hyperparameters_rqm': [("1", 0, 0, 10, 10, 1 / n)]
        }
    }
            
            return epsilon_hyperparams
