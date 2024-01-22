import numpy as np
from Arm import *
from Environment import *
from datetime import datetime
from tqdm import tqdm
import random
import math
from functools import reduce
def Run_Simulation(args):
    filename="./Exp_multiArm/"

    num_simulations = args.num_simulation
    num_epoch = args.num_epoch
    num_arm = args.num_arm
    num_d = args.num_d
    base_mean = args.base_mean
    Delta_min = args.Delta_min

    #pareto_set_ratio = args.pareto_set_ratio
    pareto_size = args.pareto_set_size
    total_user_each_arm = args.total_user_each_arm

    epsilon = args.epsilon # EG
    fixed_ratio = args.TS_fixed_ratio # TS

    #both should be ndarray
    bayesian_mean = args.bayesian_mean
    bayesian_variance = args.bayesian_var 

    Exp = args.Exp
     
    stop_rule = args.stop_rule # SPRT, t-test, PSI_Rule, esSR
    MAB_alg = args.MAB_alg # Elimination, TS, UCB, PSI, US, EG, AB
    reward_type = args.reward_type # gauss
    test_type = args.test_type  # sequential

    alpha, beta = args.alpha, args.beta
    D_variance = args.Dimensional_variance
    if num_d > 5:
        bayesian_variance[3] = bayesian_variance[3] * D_variance
        bayesian_variance[4] = bayesian_variance[4] * D_variance
    
    num_user = int((num_arm*total_user_each_arm)/num_epoch) 
    
    #create file for saving results
    filename = filename + Exp + '_C'+str(D_variance) + '_K'+str(num_arm) + '_D'+str(num_d) + 'alpha'+str(alpha)+'beta'+str(beta)+'_'+MAB_alg + '_Pareto_set_size'+str(pareto_size) + '_BaseMean'+str(base_mean) + '_DeltaMean'+str(bayesian_mean[0])+'_DeltaVar'+str(bayesian_variance[0]) + '_Min'+str(Delta_min)
    filename += str(datetime.now())
    filename = filename.replace(":", "")
    f = open(filename, "w")

    sample_epochs = []#save the sample epochs for success simulation
    total_sample_epochs = []#save the sample epochs for every simulation
    Regret_all = []#save the regret for every simulation

    Accept_H0 = 0
    Accept_H1_and_FindOpt = 0
    Find_truely_opts = 0
    Accept_H1_butNotOpt = 0

    #record the ratio of type I error and type II error
    Type_I_error_count = 0
    Type_II_error_count = 0

    with tqdm(total = num_simulations) as pbar:
        pbar.set_description('Simulations :')
        
        for sim in range(num_simulations):
            #creat mean for every arm
            if Exp == 'AB' and (bayesian_variance>0).all() and num_arm == 2:
                mean_set = np.array([[base_mean for i in range(num_d)]])
                gap = np.random.normal(loc= bayesian_mean, scale=np.sqrt(bayesian_variance), size=num_d)
                value = mean_set[0] + gap
                while (reward_type== 'bernoulli' and value.min()<0) or (reward_type== 'bernoulli' and value.max()>1):
                    gap = np.random.normal(loc= bayesian_mean, scale=np.sqrt(bayesian_variance/2), size=num_d)
                    value = mean_set[0] + gap
                mean_set = np.append(mean_set,value.reshape((1,-1)),axis=0)

            elif Exp == 'AB' and (bayesian_variance>0).all() and num_arm > 2:    
                mean_set = np.zeros((num_arm, num_d))
                mean_set[0] = np.array([base_mean for i in range(num_d)])
                pareto_set = np. sort(np.append(np.array(random.sample(range(1, num_arm), pareto_size -1)), 0))        
                pareto_flag = 0# flag for arm in pareto set

                # create pareto set
                for arm in pareto_set[1:]:
                    pareto_flag += 1
                    gap = creat_normal_value(bayesian_mean, bayesian_variance/2, num_d)
                    value = mean_set[0] + gap
                    #make sure the arm is incomparable with the first arm and the last pareto arm
                    # or not (np.abs(value - mean_set[pareto_set[:pareto_flag]]) > 0.02).all()
                    while((gap > 0).all() or (gap < 0).all() or (value - mean_set[pareto_set[pareto_flag - 1]] < 0).all() or (value - mean_set[pareto_set[pareto_flag - 1]] > 0).all()):
                        gap = creat_normal_value(bayesian_mean, bayesian_variance/2, num_d)
                        value = mean_set[0] + gap
                    mean_set[arm] = value

                #create non-pareto set
                for arm in np.delete([i for i in range(num_arm)], pareto_set):
                    gap = creat_normal_value(bayesian_mean, bayesian_variance/2, num_d)
                    
                    '''
                    while(not (np.abs(gap) > 0.0005).all()):
                        gap = creat_normal_value(bayesian_mean, bayesian_variance/2, num_d)
                    '''
                    
                    gap = -np.abs(gap)
                    value = mean_set[pareto_set[0]] + gap 
                    mean_set[arm] = value
                
            elif Exp == 'AA':
                mean_set = np.array([[base_mean for i in range(num_d)] for i in range(num_arm)])

            elif (bayesian_variance == 0.0).all() and num_arm == 2:
                mean_set = np.array([[base_mean for i in range(num_d)]])
                mean_set = np.append(mean_set,mean_set[0] + bayesian_mean,axis = 0)

            #creat variance for every arm
            if reward_type == 'bernoulli':
                variance_set=np.array([np.multiply(mean_set[i],(1-mean_set[i])) for i in range(num_arm)])
            else:
                len_vari = len(args.variance_list)
                variance_set = np.array([[args.variance_list[i] for j in range(num_d)] for i in range(len_vari)])
                if len_vari < num_arm:
                    variance_set = np.append(variance_set,np.array([[args.variance_list[-1] for j in range(num_d)] for i in range(num_arm - len(args.variance_list))]),axis=0)
            
            
            #write the parameters into file
            f.write("\n-------------Simulations:" + str(sim))
            f.write("\nReal mean: " + str(mean_set))

            #initialization for creating environment
            arm_set = get_arm_set(num_arm, reward_type, num_d, mean_set, variance_set)#ndarray
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
            env = Environment(Delta_min, bayesian_mean, bayesian_variance, arms=arm_set, num_user=num_user, final_epoch=num_epoch, stop_rule=stop_rule, use_sf=use_sf, scalarization_function=scalarization_function)

            #get the truly optimal set
            A_star = compute_optimal_set(arm_set) if Exp == 'AB' else [i for i in range(num_arm)] 

            if stop_rule == 'sf_SPRT':
                A_star = compute_optimal_set_sf(arm_set, scalarization_function)
            
            #compute the delta value of every arm for computing the intananeous regret
            Delta_Parote = compute_delta_value(A_star, arm_set, sf=scalarization_function, stop_rule=stop_rule)#list                 
            
            #initialization for PSI algorithm
            A = [i for i in range(num_arm)]
            P = []

            #initialization for regret record in this simulation
            Regret = 0

            #simulation of searching in the multi-objective space
            if stop_rule != 'esSR':

                for epoch in range(num_epoch):

                    Regret += env.generate_likelihood_data(final_chosen_prob = final_chosen_prob, current_epoch = epoch, arms = arm_set, A = A, Delta = Delta_Parote)

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

                for round in range(1, num_arm):
                    for arm in range(num_arm):
                        if Active_arm_index_fc[np.isin(Active_arm_index_fc, test_elements=[arm])].size != 0:
                            times = math.ceil(((budget_n - num_arm)/(num_arm + 1 - round))*(1/Log_K)) - math.ceil(((budget_n - num_arm)/(num_arm + 1 - round + 1))*(1/Log_K)) if round > 1 else math.ceil(((budget_n - num_arm)/(num_arm))*(1/Log_K))
                            Regret += env.generate_likelihood_data(final_chosen_prob = final_chosen_prob, current_epoch = round, arms = arm_set, A = A, Delta = Delta_Parote, arm_i = arm, times = times)
                            user_observation += times

                    average_rewards = np.array([np.mean(arm_set[arm].rewards, axis=0) for arm in range(num_arm)])

                    for func in range(num_f):
                        function_values = np.dot(scalarization_function[func], average_rewards.T)
                        function_values[np.where(Active_arm_index_fc[func] == -1)] = float('inf')
                        delete_arm = np.argmin(function_values)
                        Active_arm_index_fc[func][delete_arm] = -1
                
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

            if (sim+1)%1000 == 0:
                f.write("\nPower="+ str(Find_truely_opts/((sim+1))) + "; Averaged Sample size = "+ str(sum(sample_epochs)/Find_truely_opts)+ "; Averaged user observations per arm = "+ str(sum(sample_epochs)*num_user/(Find_truely_opts*num_arm)) + " ;Regret = " + str(sum(Regret_all)/(sum(total_sample_epochs)*num_user)))
                f.write("\nType I error ratio:"+str(Type_I_error_count / Accept_H1_and_FindOpt)+"; Type II error count:"+str(Type_II_error_count / Accept_H1_and_FindOpt))
            pbar.update(1)

            f.flush()
        f.close()
