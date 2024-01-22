import numpy as np
def get_UCB_Pareto_Optimal_Set(num_d,num_k,arm_set):
    arms_counter = []
    Pareto_set = []
    all_counter = 0
    post_mean = []
    for i in range(num_k):
        total_data=len(arm_set[i].rewards)
        all_counter += total_data
        arms_counter.append(total_data)
        post_mean.append(np.sum(np.array(arm_set[i].rewards),axis=0,keepdims=False)/total_data)
    mu = [post_mean[k]+np.sqrt((2/arms_counter[k])*np.log( all_counter*(num_d*num_k)**0.25)) for k in range(num_k)]
    for i in range(num_k):
            optimal = True
            l = 0
            while optimal and l<num_k:
                if np.min(mu[l]-mu[i]) >= 0 and l!=i:
                    optimal = False
                    break
                l += 1
            if optimal:
                Pareto_set.append(i)

    return Pareto_set
        