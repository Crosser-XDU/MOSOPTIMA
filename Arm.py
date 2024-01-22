import numpy as np
import math

def compute_mu_sigma_of_Lognormal(mean,variance):
    sigma=(np.sqrt(np.log((variance/mean**2)+1)))
    mu=(np.log(mean)-sigma**2/2)

    return mu,sigma

class Arm():
    def __init__(self,arm_id,reward_type,D,mean_list,variance_list):
        self.arm_id=arm_id
        self.reward_type=reward_type
        self.D=D
        self.mean_list=mean_list.tolist()
        self.variance_list=variance_list.tolist()
        self.rewards = []
    
    def get_reward(self):
        if self.reward_type=='bernolii':
            reward=np.random.binomial(1,self.mean_list,self.D)
        elif self.reward_type=='gauss':
            reward=creat_normal_value(self.mean_list, self.variance_list, self.D)#np.random.normal(loc = self.mean_list,scale = np.sqrt(self.variance_list), size = self.D)
        elif self.reward_type=='lognormal':
            mu = [0 for i in range(self.D)]
            sigma = [0 for i in range(self.D)]
            for i in range(len(self.mean_list)):
                mu[i],sigma[i]=compute_mu_sigma_of_Lognormal(self.mean_list[i],self.variance_list[i])
            reward=np.random.lognormal(mu,sigma,self.D)
        return reward


def get_arm_set(num_arm,reward_type,D,mean_set,variance_set):
    #return a list of arms
    Set=[]
    for i in range(num_arm):
        Set.append(Arm(i,reward_type,D,mean_list=mean_set[i],variance_list=variance_set[i]))
    return Set

def compute_optimal_set(arm_set):
    #compute the optimal set
    A_star = []
    num_arm = len(arm_set)
    for k in range(num_arm):
        mu = np.asarray(arm_set[k].mean_list)
        is_optimal = True
        i = 0
        while i < num_arm and is_optimal:
            if np.max(mu - np.asarray(arm_set[i].mean_list))<=0 and i != k:
                is_optimal = False
            i += 1
        if is_optimal:
            A_star.append(k)
    return A_star


def compute_optimal_set_sf(arm_set, sf):
    #compute the optimal arm for every scalarization function
    actual_rewards = np.array([arm_set[arm].mean_list for arm in range(len(arm_set))])
    function_values = np.dot(actual_rewards, sf.T)
    return np.argmax(function_values, axis=0).tolist()


def compute_delta_value(A_star, arm_set, sf = -1, stop_rule = -1):
    #compute delta value for each arm
    Delta_Parote = []
    num_arm = len(arm_set)
    if stop_rule != 'sf_SPRT':
        for i in range(num_arm):
            max = np.array([np.clip(np.min(np.array(arm_set[j].mean_list) - np.array(arm_set[i].mean_list)), a_min = 0, a_max = float('inf')) for j in A_star]).max()
            Delta_Parote.append(max)
    else:
        for i in range(num_arm):
                for j in A_star:
                    Delta_Parote.append(float(np.dot(np.array(arm_set[j].mean_list), sf.T) - np.dot(np.array(arm_set[i].mean_list), sf.T)))
    return Delta_Parote

def creat_normal_value(mean_list, variance_list, num_d):
    #creat normal value for each arm
    normal_value = []
    for i in range(num_d):
        normal_value.append(np.random.normal(loc = mean_list[i], scale = np.sqrt(variance_list[i])))
    return np.array(normal_value)