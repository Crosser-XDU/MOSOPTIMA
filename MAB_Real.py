from Run_Real import Run_Real
import argparse
from RealArm import *
from multiprocessing import Pool

num_process = 16

if __name__ == "__main__":

    parse = argparse.ArgumentParser(description='Parameters')
    parse.add_argument('--num_epoch', default = 20, type=int)
    parse.add_argument('--version', default = 'batch', type=str)
    parse.add_argument('--num_run', default=100, type=int)

    parse.add_argument('--Delta_min', default = 0.05, type=float)

    parse.add_argument('--reward_type', default = 'gauss', type=str)
    parse.add_argument('--total_user_each_arm', default = 2000, type=int)

    parse.add_argument('--Exp', default='AB', type=str) # AA, AB

    parse.add_argument('--test_type', default ='sequential', type=str) # sequential, fixed
    parse.add_argument('--stop_rule', default ='SPRT', type=str) # SPRT, T-test, PSI_Rule, esSR, sf_SPRT
    parse.add_argument('--MAB_alg', default ='Elimination', type=str) # Elimination, TS, UCB, PSI, US, EG, AB, esSR
    parse.add_argument('--TS_fixed_ratio', default = 0.2, type=float)  # 0 (EG), 1 (US)
    parse.add_argument('--epsilon', default = 0.1, type=float)

    parse.add_argument('--alpha', default = 0.05, type=float)
    parse.add_argument('--beta', default = 0.5, type=float)

    # parse.add_argument('--fname', default = 'test_real_data.csv', type=str)
    parse.add_argument('--meta_id', default = 0, type=int)

    parse.add_argument('--objectives', default = None, type=list)

    stop_rule = ['SPRT', 'T-test', 'PSI_Rule', 'esSR']
    Eli_algo = ['Elimination', 'UCB', 'TS', 'US', 'EG']
    T_test_algo = ['AB']
    PSI_algo = ['PSI']
    esSR_algo = ['esSR']
    #sf_SPRT_algo = ['Elimination']
    Algorithm = [Eli_algo, T_test_algo, PSI_algo, esSR_algo]

    meta_id = parse.parse_args().meta_id
    reward_type = parse.parse_args().reward_type
    objectives = parse.parse_args().objectives
    arms = RealArms(meta_id=meta_id, reward_type=reward_type)

    pool = Pool(processes=num_process)

    for stop_rule_index in range(len(stop_rule)):
        parse.set_defaults(stop_rule = stop_rule[stop_rule_index])
        print("Stop Rule: ", stop_rule[stop_rule_index])
        for algo in Algorithm[stop_rule_index]:
            print("Algorithm: ", algo)
            parse.set_defaults(MAB_alg = algo)
            parse.set_defaults(num_epoch = 20)
            if algo == 'PSI':
                parse.set_defaults(num_epoch = 100)
            Run_Real(parse.parse_args(), arms)
            # tmp = pool.apply_async(Run_Real, args=(parse.parse_args(), arms))

    # pool.close()
    # pool.join()
            # Run_Real(parse.parse_args(), arms)
