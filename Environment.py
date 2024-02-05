from scipy.stats import norm
import math
import numpy as np
from Parote_UCB1 import *
from Parote import *

class Environment():
    def __init__(self, Delta_min,bayesian_mean,bayesian_variance,arms,num_user,final_epoch,stop_rule,use_sf,scalarization_function):

        self.K = len(arms)
        self.D = bayesian_mean.shape[0]

        self.find_opt = False
        self.num_user = num_user
        self.arms = arms
        self.stop_rule = stop_rule
        self.final_epoch = final_epoch
        self.use_sf = use_sf
        self.bayesian_mean = bayesian_mean
        self.bayesian_variance = bayesian_variance

        if self.use_sf:
            self.scalarization_function = scalarization_function
            self.bayesian_mean = float(self.bayesian_mean @ scalarization_function.T)
            self.bayesian_variance = float(self.bayesian_variance @ (scalarization_function.T)**2)

        self.num_f = self.scalarization_function.shape[0] if self.use_sf else 0
        self.Delta_min = Delta_min

        self.opt = -1
        self.Return_type = -1

        self.SPRT_likelihood_prior_ratio = norm.cdf(0, loc=self.bayesian_mean, scale=np.sqrt(self.bayesian_variance)) / (1 - norm.cdf(self.Delta_min, loc=self.bayesian_mean, scale=np.sqrt(self.bayesian_variance)))

    def generate_likelihood_data(self, final_chosen_prob,current_epoch,arms,A,Delta,arm_i = -1,times = -1):
        regret_epoch = 0
        if self.stop_rule == 'SPRT' or self.stop_rule == 'sf_SPRT':
            if current_epoch == 0:
                for arm_id in range(self.K):
                    regret_epoch += Delta[arm_id]
                    new_reward_list = arms[arm_id].get_reward().tolist()
                    self.arms[arm_id].rewards.append( new_reward_list )#list
            else:
                arm_list = range(self.K)
                for user in range(self.num_user):
                    arm_id = np.random.choice(arm_list, p = final_chosen_prob)
                    regret_epoch += Delta[arm_id]
                    new_reward_list = arms[arm_id].get_reward().tolist()
                    self.arms[arm_id].rewards.append( new_reward_list )
        elif self.stop_rule == 'PSI_Rule':
            for arm_id in A:
                for user in range(int(self.num_user/len(A))):
                    regret_epoch += Delta[arm_id]
                    new_reward_list = arms[arm_id].get_reward().tolist()
                    self.arms[arm_id].rewards.append(new_reward_list)
        elif self.stop_rule == 'T-test':
            for arm_id in range(self.K):
                for user in range(int(self.num_user/self.K)):
                    regret_epoch += Delta[arm_id]
                    new_reward_list = arms[arm_id].get_reward().tolist()
                    self.arms[arm_id].rewards.append(new_reward_list)
        elif self.stop_rule == 'esSR':
            for select in range(times):
                regret_epoch += Delta[arm_i]
                new_reward_list = arms[arm_i].get_reward().tolist()
                self.arms[arm_i].rewards.append(new_reward_list)

        return regret_epoch

    def SPRT(self,alpha,beta,Flag,Flag_set,real_variance,test_type,current_epoch):

        post_mean = np.array([np.mean(self.arms[arm].rewards, axis=0) for arm in range(self.K)])
        post_var = real_variance / np.array([len(self.arms[arm].rewards) for arm in range(self.K)]).reshape(-1,1)

        sub_beta = beta / (self.K*(self.K-1)*self.D)
        sub_alpha = alpha / (self.K*(self.K-1))
        threshold_a, threshold_b = sub_beta / (1-sub_alpha), (1-sub_beta) / (sub_alpha)

        # Create a matrix of mean differences and total variances
        mean_dif = post_mean[:, None] - post_mean[None, :]
        total_var = post_var[:, None] + post_var[None, :]

        # Calculate the likelihood ratio means and variances
        likeli_likeli_ratio_mean = (mean_dif*self.bayesian_variance+self.bayesian_mean*total_var)/(self.bayesian_variance+total_var)
        likeli_likeli_ratio_var = (self.bayesian_variance*total_var)/(self.bayesian_variance+total_var)

        # Calculate the posterior probabilities
        postprob = norm.cdf(0, loc=likeli_likeli_ratio_mean, scale=np.sqrt(likeli_likeli_ratio_var))
        postprob_rest = 1 - norm.cdf(self.Delta_min, loc = likeli_likeli_ratio_mean, scale = np.sqrt(likeli_likeli_ratio_var))

        # Calculate the likelihood ratio
        LikeliRatio = np.divide(postprob_rest, postprob, out = 1000000*np.ones_like(postprob), where=postprob >= 1e-308)

        LogLikelihood = self.SPRT_likelihood_prior_ratio * LikeliRatio

        max_judge = (LogLikelihood >= threshold_b).astype(int)
        min_judge = (LogLikelihood <= threshold_a).astype(int)

        Flag_set = (max_judge - np.logical_not(np.logical_xor(max_judge,min_judge).astype(int)))

        #update the Flag matrix
        for arm_i in range(self.K - 1):
            for arm_j in range(arm_i + 1, self.K):

                Flag_diff = Flag_set[arm_i][arm_j] - Flag_set[arm_j][arm_i]
                Flag_max = Flag_diff.max()

                if((Flag_set[arm_i][arm_j] == 1).all()) or ((Flag_set[arm_j][arm_i] == 0).all()):
                    Flag[arm_i][arm_j] = 1
                    Flag[arm_j][arm_i] = 0
                elif((Flag_set[arm_i][arm_j] == 0).all()) or ((Flag_set[arm_j][arm_i] == 1).all()):
                    Flag[arm_i][arm_j] = 0
                    Flag[arm_j][arm_i] = 1
                elif((True in (Flag_set[arm_j][arm_i][np.where(Flag_set[arm_i][arm_j]== 1)[0]] == 0)) and (True in (Flag_set[arm_j][arm_i][np.where(Flag_set[arm_i][arm_j]== 0)[0]] == 1))):
                    Flag[arm_i][arm_j] = 0
                    Flag[arm_j][arm_i] = 0
                elif((Flag_set[arm_i][arm_j]).min() == -1 or (Flag_set[arm_j][arm_i]).min() == -1):
                    Flag[arm_i][arm_j] = -1
                    Flag[arm_j][arm_i] = -1
                else:
                    Flag[arm_i][arm_j] = Flag_max
                    Flag[arm_j][arm_i] = 1-Flag_max

        Return_type = -1
        Find_opt = True

        # judge whether the Pareto optimal set is found
        Active_arm_index = np.delete([arm_id for arm_id in range(self.K)] , np.where(Flag == 1)[1]).tolist()
        for arm_i in Active_arm_index:
                for arm_j in Active_arm_index:
                    if arm_i != arm_j:
                        if  Flag[arm_i][arm_j] != 0 or Flag[arm_j][arm_i] != 0:
                            Find_opt = False
                            break

        if Find_opt == True:
            if len(Active_arm_index) == self.K:
                Return_type = 0
            else:
                Return_type = 1

        if test_type == "sequential":
            self.Return_type = Return_type
            self.find_opt = Find_opt

        elif test_type == "fixed" and current_epoch == self.final_epoch -1:
            self.Return_type = Return_type
            self.find_opt = Find_opt

        return Flag, post_mean, post_var

    def sf_SPRT(self,alpha,beta,Flag,real_variance,test_type,current_epoch):

        post_mean = np.array([np.mean(self.arms[arm].rewards, axis=0) for arm in range(self.K)])
        post_var = real_variance / np.array([len(self.arms[arm].rewards) for arm in range(self.K)]).reshape(-1,1)
        post_mean = (post_mean @ self.scalarization_function.T).reshape(-1)
        post_var = (post_var @ (self.scalarization_function.T)**2).reshape(-1)

        sub_beta = beta / (self.K*(self.K-1))
        sub_alpha = alpha / (self.K*(self.K-1))
        threshold_a, threshold_b = sub_beta / (1-sub_alpha), (1-sub_beta) / (sub_alpha)

        # Create a matrix of mean differences and total variances
        mean_dif = post_mean[:, None] - post_mean[None, :]
        total_var = post_var[:, None] + post_var[None, :]

        # Calculate the likelihood ratio means and variances
        likeli_likeli_ratio_mean = (mean_dif*self.bayesian_variance+self.bayesian_mean*total_var)/(self.bayesian_variance+total_var)
        likeli_likeli_ratio_var = (self.bayesian_variance*total_var)/(self.bayesian_variance+total_var)

        # Calculate the posterior probabilities
        postprob = norm.cdf(0, loc=likeli_likeli_ratio_mean, scale=np.sqrt(likeli_likeli_ratio_var))
        postprob_rest = 1 - norm.cdf(self.Delta_min, loc = likeli_likeli_ratio_mean, scale = np.sqrt(likeli_likeli_ratio_var))

        # Calculate the likelihood ratio
        LikeliRatio = np.divide(postprob_rest, postprob, out = 1000000*np.ones_like(postprob), where=postprob >= 1e-308)

        LogLikelihood = self.SPRT_likelihood_prior_ratio * LikeliRatio

        max_judge = (LogLikelihood >= threshold_b).astype(int)
        min_judge = (LogLikelihood <= threshold_a).astype(int)

        Flag = (max_judge - np.logical_not(np.logical_xor(max_judge,min_judge).astype(int)))
        Flag[np.diag_indices(self.K)] = -1

        Return_type = -1
        Find_opt = False
        Opt = -1

        for arm_i in range(self.K):
                find_opt = False
                if max(Flag[arm_i])>=1 and np.sum(Flag[arm_i]==-1)==1 and Flag[arm_i][arm_i]==-1:
                    find_opt = True
                    for arm_j in range(self.K):
                        if arm_j!=arm_i and Flag[arm_i][arm_j]==0 and Flag[arm_j][arm_i]!=0:
                            find_opt = False
                            break

                if find_opt == True:
                    Return_type = 1
                    Find_opt=True
                    Opt = arm_i
                    break

        items = []
        for arm_i in range(self.K):
                for arm_j in range(self.K):
                    if arm_i != arm_j:
                        items.append(Flag[arm_i][arm_j])


        if Return_type != 1 and min(items)>=0:
                if max(items)<=0:
                    Return_type = 0
                elif max(items)==1:
                    Return_type = 2

        if test_type == "sequential":
            self.Return_type = Return_type
            self.find_opt = Find_opt
            self.opt = Opt

        elif test_type == "fixed" and current_epoch==self.final_epoch -1:
            self.Return_type = Return_type
            self.find_opt = Find_opt
            self.opt = Opt

        return Flag, post_mean, post_var


    def PSI(self, epsilon, delta,  A, P, real_variance):

        post_mean = np.zeros(shape = (self.K, self.D))
        beta  = np.zeros(shape = (self.K, self.D))
        Upper_Bound = 0.05
        for arm_id in range(self.K):
            n_i = len(self.arms[arm_id].rewards)
            post_mean[arm_id] = np.sum(np.array(self.arms[arm_id].rewards), axis=0, keepdims=False) / n_i
            beta[arm_id] = np.sqrt((2/n_i) * math.log(self.K*self.D*n_i/delta) * ((real_variance[arm_id]/n_i) + Upper_Bound * math.sqrt((4/n_i) * math.log(self.K*self.D*n_i/delta))))

        A_1 = []
        P_1 = []
        P_2 = []

        for arm_i in A:
            flag = True
            for arm_j in A:
                m_ij_index = np.argmin(post_mean[arm_j] - post_mean[arm_i])
                m_ij = np.clip(post_mean[arm_j][m_ij_index] - post_mean[arm_i][m_ij_index], a_min = 0, a_max = float('inf'))
                if m_ij > math.sqrt(beta[arm_i][m_ij_index]**2 + beta[arm_j][m_ij_index]**2):
                    flag = False
                    break
            if flag:
                A_1.append(arm_i)

        for arm_i in A_1:
            flag = True
            for arm_j in A_1:
                if arm_j != arm_i:
                    M_ij_index = np.argmax(post_mean[arm_i] + epsilon - post_mean[arm_j])
                    M_ij = np.clip(post_mean[arm_i][M_ij_index] + epsilon - post_mean[arm_j][M_ij_index], a_min = 0, a_max = float('inf'))
                    if M_ij < math.sqrt(beta[arm_i][M_ij_index]**2 + beta[arm_j][M_ij_index]**2):
                        flag = False
                        break
            if flag:
                P_1.append(arm_i)

        for arm_j in P_1:
            flag = True
            for arm_i in np.setdiff1d(A_1, P_1):
                M_ij_index = np.argmax(post_mean[arm_i] + epsilon - post_mean[arm_j])
                M_ij = np.clip(post_mean[arm_i][M_ij_index] + epsilon - post_mean[arm_j][M_ij_index], a_min = 0, a_max = float('inf'))
                if M_ij <= math.sqrt(beta[arm_i][M_ij_index]**2 + beta[arm_j][M_ij_index]**2):
                    flag = False
                    break
            if flag:
                P_2.append(arm_j)

        A = np.setdiff1d(A_1, P_2)
        P.extend(P_2)

        return A,P

    def T_test(self, real_variance, current_epoch, test_type):
        # multi-arm
        Flag = -np.ones((self.K,self.K))#ndarray
        if current_epoch == self.final_epoch-1:

            alpha = 0.05/(self.K*(self.K-1))
            post_mean = np.array([np.mean(self.arms[arm].rewards, axis=0) for arm in range(self.K)])
            post_var = real_variance / np.array([len(self.arms[arm].rewards) for arm in range(self.K)]).reshape(-1,1)

            for arm_i in range(self.K):
                for arm_j in range(arm_i+1, self.K):
                    standard_error = np.sqrt(post_var[arm_i] + post_var[arm_j])

                    p_value = 1 - norm.cdf(abs(post_mean[arm_i] - post_mean[arm_j]) / standard_error)

                    greater_judge = (post_mean[arm_i] >= post_mean[arm_j]).astype(int) * (p_value < alpha).astype(int)
                    less_judge = (post_mean[arm_i] < post_mean[arm_j]).astype(int) * (p_value < alpha).astype(int)

                    Flag_ij = (greater_judge - np.logical_not(np.logical_xor(greater_judge, less_judge).astype(int)))

                    if Flag[arm_i][arm_j] == -1 or Flag[arm_j][arm_i] == -1:

                        if((Flag_ij == 1).all()):
                            Flag[arm_i][arm_j] = 1
                            Flag[arm_j][arm_i] = 0
                        elif((Flag_ij == 0).all()):
                            Flag[arm_i][arm_j] = 0
                            Flag[arm_j][arm_i] = 1
                        elif((0 in Flag_ij) and (1 in Flag_ij)):
                            Flag[arm_i][arm_j] = 0
                            Flag[arm_j][arm_i] = 0
                        elif((Flag_ij).min() == -1):
                            Flag[arm_i][arm_j] = -1
                            Flag[arm_j][arm_i] = -1


            Return_type = -1
            Find_opt = True

            Pareto_arm_index = np.delete([arm_id for arm_id in range(self.K)] , np.where(Flag == 1)[1]).tolist()
            for arm_i in Pareto_arm_index:
                    for arm_j in Pareto_arm_index:
                        if arm_i != arm_j:
                            if Flag[arm_i][arm_j] != 0 or Flag[arm_j][arm_i] != 0:
                                Find_opt = False
                                break

            if Find_opt == True:
                if len(Pareto_arm_index) == self.K:
                    Return_type = 0
                else:
                    Return_type = 1

            if test_type == "sequential":
                self.Return_type = Return_type
                self.find_opt = Find_opt

            elif test_type == "fixed" and current_epoch == self.final_epoch -1:
                self.Return_type = Return_type
                self.find_opt = Find_opt

        return Flag

    def PSI_Test(self, A, P):
        if A.size == 0:
            if np.sort(P).tolist == [i for i in range(self.K)]:
                self.Return_type = 0
            else:
                self.Return_type = 1

    def compute_Eliminate_final_assignment(self,Active_arm_index):
        chosen_prob = [0 for arm in self.arms]
        for arm_i in range(self.K):
            chosen_prob[arm_i] = 1/len(Active_arm_index) if arm_i in Active_arm_index else 0

        return chosen_prob

    def compute_UCB_Parote_final_assingment(self):
        Pareto_Optimal_Set = get_UCB_Pareto_Optimal_Set(self.D,self.K,self.arms)
        chosen_prob = [0 for arm in self.arms]
        for arm_i in range(self.K):
            chosen_prob[arm_i] = 1/len(Pareto_Optimal_Set) if arm_i in Pareto_Optimal_Set else 0
        return chosen_prob

    # def compute_EG_final_assingment(self, epsilon):
    #     Pareto_Optimal_Set  = get_Pareto_Optimal_Set(self.arms, Flag = 'EG' )
    #     chosen_prob = np.asarray([epsilon/self.K for arm in self.arms])
    #     chosen_prob[Pareto_Optimal_Set] += (1 - epsilon)/len(Pareto_Optimal_Set)
    #     return chosen_prob

    def compute_EG_final_assingment(self, epsilon):
        Pareto_Optimal_Set  = get_Pareto_Optimal_Set(self.arms, Flag = 'EG' )
        if len(Pareto_Optimal_Set) == 0:
            chosen_prob = np.asarray([1/self.K for arm in self.arms])
        else:
            chosen_prob = np.asarray([epsilon/self.K for arm in self.arms])
            chosen_prob[Pareto_Optimal_Set] += (1 - epsilon)/len(Pareto_Optimal_Set)
        return chosen_prob

    def compute_TS_final_assignment(self, post_mean, post_var ,fixed_ratio,  MC_simulation = 10000 ):
        final_chosen_prob = [0 for arm in self.arms]
        K = post_mean.shape[0]

        if fixed_ratio == 1:
            final_chosen_prob = [1/K for arm in range(K)]
        else:
            simulation_results = []
            for arm_id in range(K):
                    simulation_results.append(np.mean(np.random.normal(loc=post_mean[arm_id], scale=np.sqrt(post_var[arm_id]), size=(MC_simulation, self.D)), axis=0))
            Pareto_Optimal_Set = get_Pareto_Optimal_Set(simulation_results, Flag = 'TS')

            chosen_prob= [1/len(Pareto_Optimal_Set) if arm_id in Pareto_Optimal_Set else 0 for arm_id in range(K)]
            for arm in range(self.K):
                final_chosen_prob[arm] = chosen_prob[arm]*(1-fixed_ratio)+fixed_ratio/self.K

        return final_chosen_prob

