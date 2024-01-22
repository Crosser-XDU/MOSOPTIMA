import numpy as np
def get_Pareto_Optimal_Set(arms, Flag):
    num_K = len(arms)
    Pareto_set = []
    if Flag == 'EG':
        arms_counter = []
        all_counter = 0
        post_mean = []
        for i in range(num_K):
            total_data=len(arms[i].rewards)
            all_counter += total_data
            arms_counter.append(total_data)
            post_mean.append(np.sum(np.array(arms[i].rewards),axis=0,keepdims=False)/total_data)
        for i in range(num_K):
                optimal = True
                l = 0
                while optimal and l<num_K:
                    if np.min(post_mean[l]-post_mean[i]) >= 0 and l!=i:
                        optimal = False
                        break
                    l += 1
                if optimal:
                    Pareto_set.append(i)
    elif Flag == 'TS':
         for i in range(num_K):
                optimal = True
                l = 0
                while optimal and l<num_K:
                    if np.min(arms[l]-arms[i]) >= 0 and l!=i:
                        optimal = False
                        break
                    l += 1
                if optimal:
                    Pareto_set.append(i)
    return Pareto_set