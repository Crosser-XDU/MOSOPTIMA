from Run_Simulation import Run_Simulation
import numpy
import argparse

if __name__ == "__main__":

    parse = argparse.ArgumentParser(description='Parameters') 
    parse.add_argument('--num_simulation', default = 3000, type=int)
    parse.add_argument('--num_epoch', default = 20, type=int)
    parse.add_argument('--num_arm', default = 5, type=int)
    parse.add_argument('--version', default = 'batch', type=str)
    parse.add_argument('--num_d',default = 5,type=int)

    parse.add_argument('--base_mean', default = 10, type=float)
    parse.add_argument('--Delta_min', default = 0.05, type=float)
    parse.add_argument('--bayesian_mean', default=numpy.array([0,0,0,0,0]), type=numpy.ndarray)
    parse.add_argument('--bayesian_var', default=numpy.array([1,1,1,1,1]), type=numpy.ndarray)

    parse.add_argument('--reward_type', default='gauss', type=str) 
    parse.add_argument('--total_user_each_arm', default = 2000, type=int)

    parse.add_argument('--Exp', default='AB', type=str) # AA, AB
    
    parse.add_argument('--test_type', default='sequential', type=str) # sequential, fixed
    parse.add_argument('--stop_rule', default='SPRT', type=str) # SPRT, T-test, PSI_Rule, esSR, sf_SPRT
    parse.add_argument('--MAB_alg', default='Elimination', type=str) # Elimination, TS, UCB, PSI, US, EG, AB, esSR
    parse.add_argument('--Elimination_type', default = 0, type=int) # default 0
    parse.add_argument('--TS_fixed_ratio', default = 0.2, type=float)  # 0 (EG), 1 (US)
    parse.add_argument('--epsilon', default = 0.1, type=float)  

    parse.add_argument('--variance_list', nargs='+', type=float)
    parse.add_argument('--pareto_set_size', default = 3, type=int)

    parse.add_argument('--alpha', default = 0.05, type=int)
    parse.add_argument('--beta', default = 0.5, type=int)

    parse.add_argument('--Dimensional_variance', default = 1, type=int) # 0 (same variance), 1 (different variance)

    stop_rule = ['SPRT']
    Eli_algo = ['TS']
    T_test_algo = ['AB']
    PSI_algo = ['PSI']
    esSR_algo = ['esSR']
    sf_SPRT_algo = ['Elimination']
    Algorithm = [Eli_algo]
    var_all = [0.05, 0.1, 0.2, 0.5, 1, 2, 4] 
    num_arm_all = [5, 10, 15, 20]
    num_d_all = [20] #, 5, 10, 15
    pareto_set_ratio = [0.2, 0.4, 0.6, 1.0]
    delta_min = [0.05]

    choice_all = [var_all, num_arm_all, num_d_all, pareto_set_ratio, delta_min]

    for stop_rule_index in range(len(stop_rule)):
        parse.set_defaults(stop_rule = stop_rule[stop_rule_index])
        for algo in Algorithm[stop_rule_index]:
            parse.set_defaults(MAB_alg = algo)
            if algo == 'PSI':
                parse.set_defaults(num_epoch = 100)
            for choice in [2]:
                for x in choice_all[choice]:
                    if choice == 0:
                        variance = numpy.array([x for i in range(num_d_all[1])])
                        mean = numpy.array([0 for i in range(num_d_all[1])])
                        parse.set_defaults(bayesian_var = variance)
                        parse.set_defaults(bayesian_mean = mean)
                        parse.set_defaults(num_d = num_d_all[1])
                        parse.set_defaults(num_arm = num_arm_all[1])
                        args = parse.parse_args() 
                        Run_Simulation(args)
                    elif choice == 1:
                        variance = numpy.array([var_all[3] for i in range(num_d_all[1])])
                        mean = numpy.array([0 for i in range(num_d_all[1])])
                        parse.set_defaults(bayesian_var = variance)
                        parse.set_defaults(bayesian_mean = mean)
                        parse.set_defaults(num_d = num_d_all[1])
                        parse.set_defaults(num_arm = x)
                        args = parse.parse_args() 
                        Run_Simulation(args)
                    elif choice == 2:
                        variance = numpy.array([var_all[3] for i in range(x)])
                        mean = numpy.array([0 for i in range(x)])
                        parse.set_defaults(bayesian_var = variance)
                        parse.set_defaults(bayesian_mean = mean)
                        parse.set_defaults(num_d = x)
                        parse.set_defaults(num_arm = num_arm_all[1])
                        args = parse.parse_args() 
                        Run_Simulation(args)
                    elif choice == 3:
                        variance = numpy.array([var_all[3] for i in range(num_d_all[1])])
                        mean = numpy.array([0 for i in range(num_d_all[1])])
                        parse.set_defaults(bayesian_var = variance)
                        parse.set_defaults(bayesian_mean = mean)
                        parse.set_defaults(num_d = num_d_all[1])
                        parse.set_defaults(num_arm = num_arm_all[1])
                        parse.set_defaults(pareto_size = x)
                        args = parse.parse_args()
                        Run_Simulation(args)
                    elif choice == 4:
                        variance = numpy.array([var_all[3] for i in range(num_d_all[1])])
                        mean = numpy.array([0 for i in range(num_d_all[0])])
                        parse.set_defaults(bayesian_var = variance)
                        parse.set_defaults(bayesian_mean = mean)
                        parse.set_defaults(num_d = num_d_all[0])
                        parse.set_defaults(num_arm = num_arm_all[0])
                        parse.set_defaults(Delta_min = x)
                        args = parse.parse_args()
                        Run_Simulation(args)