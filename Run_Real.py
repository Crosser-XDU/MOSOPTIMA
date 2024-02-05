import numpy as np
from Arm import *
from RealArm import *
from Environment import *
from datetime import datetime
from tqdm import tqdm
import random
import pdb
import math
def Run_Real(args, arms):
    filename="./Exp_multiArm/"

    #set parameters
    num_epoch = args.num_epoch
    Delta_min = args.Delta_min
    num_run = args.num_run

    #pareto_set_ratio = args.pareto_set_ratio
    total_user_each_arm = args.total_user_each_arm

    epsilon = args.epsilon # EG
    fixed_ratio = args.TS_fixed_ratio # TS

    stop_rule = args.stop_rule # SPRT, t-test, PSI_Rule, esSR
    MAB_alg = args.MAB_alg # Elimination, TS, UCB, PSI, US, EG, AB
    reward_type = args.reward_type # gauss
    test_type = args.test_type  # sequential
    Exp = args.Exp # AA, AB
    alpha, beta = args.alpha, args.beta

    # fname = args.fname
    objectives = args.objectives

    #create real arms
    num_arm = arms.num_arm
    num_d = arms.num_dimension
    bayesian_mean = np.array(arms.gap_mean)
    bayesian_variance = np.array(arms.gap_var)
    variance_set = arms.vars
    mean_set = arms.means
    num_user = int((num_arm*total_user_each_arm)/num_epoch)
    if objectives != None:
        num_d = len(objectives)
        bayesian_mean = np.array(arms.gap_mean[objectives])
        bayesian_variance = np.array(arms.gap_var[objectives])
        variance_set = arms.vars[:, objectives]
        mean_set = arms.means[:, objectives]
    arm_set = arms.getarms(objectives, total_user_each_arm)

    #create file for saving results
    filename = filename + '_K'+str(num_arm) + '_D'+str(num_d) + '_Reward_type'+reward_type + 'alpha'+str(alpha)+'beta'+str(beta) + '_'+ MAB_alg
    filename += str(datetime.now())
    filename = filename.replace(":", "")
    f = open(filename, "w")

    sample_epochs = [] #save the sample epochs for success simulation
    total_sample_epochs = [] #save the sample epochs for every simulation
    Regret_all = [] #save the regret for every simulation

    Accept_H0 = 0
    Accept_H1_and_FindOpt = 0
    Find_truely_opts = 0
    Accept_H1_butNotOpt = 0

    #record the ratio of type I error and type II error
    Type_I_error_count = 0
    Type_II_error_count = 0

    final_chosen_prob = [1/num_arm for i in range(num_arm)]#list

    #initialization of esSR and sf_SPRT algorithm
    num_f = 1
    if stop_rule == 'esSR':
        num_f = 100*num_d
    scalarization_function = np.random.uniform(0, 1, (num_f,num_d))#sample from uniform distribution
    scalarization_function = scalarization_function/np.sum(scalarization_function, axis=1, keepdims=True)

    #store the flag of every arm for SPRT algorithm(1 for dominate, 0 for be dominated, -1 for not sure)
    Flag = -np.ones((num_arm, num_arm))#arm-arm ndarray
    Flag_set = -np.ones((num_arm, num_arm, num_d))#dimension-dimension ndarray

    #create environment for simulation
    use_sf = True if stop_rule == 'sf_SPRT' else False

    #get the truly optimal set
    A_star = compute_optimal_set(arm_set)
    print("A_star:", A_star)

    if stop_rule == 'sf_SPRT':
        A_star = compute_optimal_set_sf(arm_set, scalarization_function)

    #compute the delta value of every arm for computing the intananeous regret
    Delta_Parote = compute_delta_value(A_star, arm_set, sf=scalarization_function, stop_rule=stop_rule)#list
    print("Delta_Parote:", Delta_Parote)


    f.write("Real mean: " + str(mean_set) + "\n")
    f.write("Real variance: " + str(variance_set) + "\n")
    f.write("Bayesian mean: " + str(bayesian_mean) + "\n")
    f.write("Bayesian variance: " + str(bayesian_variance) + "\n")

    with tqdm(total=num_run) as pbar:
        pbar.set_description('Simulation:')

        for run in range(num_run):
            #initialization for this simulation
            Regret = 0
            arm_set = arms.getarms(objectives, total_user_each_arm)
            env = Environment(Delta_min, bayesian_mean, bayesian_variance, arms=arm_set, num_user=num_user, final_epoch=num_epoch, stop_rule=stop_rule, use_sf=use_sf, scalarization_function=scalarization_function)

            #initialization for PSI algorithm
            A = [i for i in range(num_arm)]
            P = []

            #simulation of searching in the multi-objective space
            if stop_rule != 'esSR':

                for epoch in range(num_epoch):

                    regret_epoch = env.generate_likelihood_data(final_chosen_prob = final_chosen_prob, current_epoch = epoch, arms = arm_set, A = A, Delta = Delta_Parote)
                    if regret_epoch == None:
                        print("Error in generate likelihood data")
                        break
                    Regret += regret_epoch

                    if stop_rule == 'SPRT':
                        Flag, post_mean, post_variance = env.SPRT(alpha, beta, Flag=Flag, Flag_set=Flag_set, current_epoch=epoch, test_type= test_type, real_variance=variance_set)
                    elif stop_rule == 'sf_SPRT':
                        Flag, post_mean, post_variance = env.sf_SPRT(alpha, beta, Flag=Flag, current_epoch=epoch, test_type= test_type, real_variance=variance_set)
                    elif stop_rule == "T-test":
                        Flag = env.T_test(real_variance = variance_set, current_epoch = epoch, test_type = test_type)
                    elif stop_rule == 'PSI_Rule':
                        A, P = env.PSI(epsilon = 0.0, delta = 0.1, A = A, P = P, real_variance = variance_set)

                    Active_arm_index = [arm_id for arm_id in range(num_arm)]
                    Active_arm_index = np.delete(Active_arm_index, np.where(Flag == 1)[1]).tolist()


                    if MAB_alg == 'Elimination':
                        final_chosen_prob = env.compute_Eliminate_final_assignment(Active_arm_index)
                    elif MAB_alg =='UCB':
                        final_chosen_prob = env.compute_UCB_Parote_final_assingment()
                    elif MAB_alg == 'EG':
                        final_chosen_prob = env.compute_EG_final_assingment(epsilon)
                    elif MAB_alg == 'TS':
                        final_chosen_prob = env.compute_TS_final_assignment(post_mean, post_variance, fixed_ratio)
                    elif MAB_alg == 'PSI':
                        env.PSI_Test(A, P)
                        Active_arm_index = P

                    if env.Return_type == 0:
                        Accept_H0 += 1
                        if len(A_star) == num_arm:
                            Find_truely_opts += 1
                            sample_epochs.append(epoch+1)

                    elif env.Return_type == 1:
                        Accept_H1_and_FindOpt += 1
                        if Exp == "AB":
                            if stop_rule == 'sf_SPRT':
                                if [env.opt] == A_star:
                                    Find_truely_opts += 1
                                    sample_epochs.append(epoch+1)
                                    Active_arm_index = [env.opt]
                            else:
                                if np.sort(Active_arm_index).tolist() == A_star:
                                    Find_truely_opts += 1
                                    sample_epochs.append(epoch+1)
                                elif len(Active_arm_index) < len(A_star):
                                    Type_I_error_count += 1
                                elif len(Active_arm_index) > len(A_star):
                                    Type_II_error_count += 1
                                else:
                                    Type_I_error_count += 1
                                    Type_II_error_count += 1

                    elif env.Return_type == 2:
                        Accept_H1_butNotOpt +=1

                    if env.Return_type != -1:
                        break

            #simulation using esSR algorithm
            if stop_rule == 'esSR':

                #set sets
                Active_arm_index_fc = np.array([[arm_id for arm_id in range(num_arm)] for f_id in range(num_f)])
                Optimal_set = []
                Log_K = 1/2 + sum([1/j for j in range(2, num_arm+1)])
                budget_n = total_user_each_arm*Log_K + num_arm
                user_observation = 0

                try:
                    for round in range(1, num_arm):
                        for arm in range(num_arm):
                            if Active_arm_index_fc[np.isin(Active_arm_index_fc, test_elements=[arm])].size != 0:
                                times = math.ceil(((budget_n - num_arm)/(num_arm + 1 - round))*(1/Log_K)) - math.ceil(((budget_n - num_arm)/(num_arm + 1 - round + 1))*(1/Log_K)) if round > 1 else math.ceil(((budget_n - num_arm)/(num_arm))*(1/Log_K))
                                regret_epoch = env.generate_likelihood_data(final_chosen_prob = final_chosen_prob, current_epoch = round, arms = arm_set, A = A, Delta = Delta_Parote, arm_i = arm, times = times)
                                if not regret_epoch:
                                    raise Exception("Error in generate likelihood data")
                                user_observation += times
                        average_rewards = np.array([np.mean(arm_set[arm].rewards, axis=0) for arm in range(num_arm)])

                        for func in range(num_f):
                            function_values = np.dot(scalarization_function[func], average_rewards.T)
                            function_values[np.where(Active_arm_index_fc[func] == -1)] = float('inf')
                            delete_arm = np.argmin(function_values)
                            Active_arm_index_fc[func][delete_arm] = -1
                except Exception as e:
                    print(e)

                for arm in range(num_arm):
                    if Active_arm_index_fc[np.isin(Active_arm_index_fc, test_elements=[arm])].size != 0:
                        Optimal_set.append(arm)

                if Optimal_set == A_star:
                    Find_truely_opts += 1
                    #user observation for success find
                    sample_epochs.append(user_observation)

                Accept_H1_and_FindOpt += 1
                Active_arm_index = Optimal_set

                #user observation for every simulation
                num_user = 1
                epoch = user_observation - 1

            total_sample_epochs.append(epoch+1)
            Regret_all.append(Regret)

            f.write("\nFind truely optimal:"+str(Find_truely_opts)+"; AcceptH1andFindOpt: "+str(Accept_H1_and_FindOpt)+"; AcceptH0: "+str(Accept_H0)+"; AcceptH1butNotFindOpt: "+str(Accept_H1_butNotOpt))
            f.write("\nSum sample epochs before stop:"+str(sum(sample_epochs)) +"; Sum total sample epochs:"+str(sum(total_sample_epochs))+"; Active Arm:"+str(np.sort(Active_arm_index))+"; Optimal Arm:"+str(A_star))
            if (run+1)%100 == 0:
                if Find_truely_opts == 0:
                    f.write("\nPower="+ str(Find_truely_opts/((run+1))) + "; Averaged Sample size = "+ str(0)+ "; Averaged user observations per arm = "+ str(0) + " ;Regret = " + str(sum(Regret_all)/(sum(total_sample_epochs)*num_user)))
                else:
                    f.write("\nPower="+ str(Find_truely_opts/((run+1))) + "; Averaged Sample size = "+ str(sum(sample_epochs)/Find_truely_opts)+ "; Averaged user observations per arm = "+ str(sum(sample_epochs)*num_user/(Find_truely_opts*num_arm)) + " ;Regret = " + str(sum(Regret_all)/(sum(total_sample_epochs)*num_user)))
                if Accept_H1_and_FindOpt == 0:
                    f.write("\nType I error ratio:"+str(0)+"; Type II error count:"+str(0))
                else:
                    f.write("\nType I error ratio:"+str(Type_I_error_count / Accept_H1_and_FindOpt)+"; Type II error count:"+str(Type_II_error_count / Accept_H1_and_FindOpt))
            pbar.update(1)
            f.flush()
        f.close()
