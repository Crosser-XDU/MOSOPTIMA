from Arm import Arm
import os
import numpy as np
import pandas as pd
import pdb
import time
from tqdm import tqdm
from real_data.meta_data import expt_metadatas
from multiprocessing import Pool

def aggregate_numerator(df, metricid2dimen):
    df = df.reset_index(drop=True)  # Reset index
    numerator = [0]*len(metricid2dimen)
    for i, metric_id in enumerate(df['metric_id']):
        numerator[metricid2dimen[metric_id]] = df['numerator'][i]
    return pd.DataFrame({
        "numerator": [numerator]
    })

def pop_std(x):
    return x.std(ddof=0)

class RealArm(Arm):
    def __init__(self,arm_id, reward_type, D, mean_list, variance_list, real_rewards):
        super().__init__(arm_id,reward_type,D,mean_list, variance_list)
        self.rewards_index = -1
        self.real_rewards = real_rewards
    def get_reward(self):
        if self.rewards_index >= len(self.real_rewards) - 1:
            return None
        self.rewards_index += 1
        return self.real_rewards[self.rewards_index]

def processing_data(armid, df1, metricid2dimen):
    d0 = str(armid)
    d1 = df1.groupby('uin').apply(lambda x: aggregate_numerator(x, metricid2dimen))
    d2 = 0
    return [d0, d1, d2]
pool_res = []

class RealArms():
    def __init__(self, meta_id, reward_type) -> None:
        self.meta_id = meta_id
        self.reward_type = reward_type

        if meta_id >= len(expt_metadatas):
            raise ValueError("Invalid meta_id")
        expt_meta = expt_metadatas[meta_id]
        data = pd.read_csv(os.path.join("real_data", expt_meta['filename']))
        # data['metric'] = data['numerator'] / data['denominator']
        # tmp = data.groupby(['metric_id', 'groupid']).agg({'metric': ['mean', 'std', pop_std, 'min', 'median', 'max'], 'uin': ['count', 'nunique']})
        # print(tmp)
        data = data[((data['metric_id'].isin(expt_meta['include_metric_ids'])) & (data['groupname'].isin(expt_meta['include_group_names'])))]

        print("Loading data...")
        self.df = data
        print("Data loaded.")
        self.df['numerator'] = self.df['numerator'] / self.df['denominator'].where(self.df['denominator'] != 0, 1)
        tmp = self.df.groupby(['exptid', 'groupid']).count().reset_index()
        gids, gid2uv = [], {}
        for i, row in tmp.iterrows():
            gids.append(row['groupid'])
            gid2uv[str(row['groupid'])] = row['uin']
        sgids = sorted(gids)
        control_gids, treat_gids = sgids[:1], sgids[1:]

        armid2gids = {}
        armid2gids["0"] = control_gids
        for i, gid in enumerate(treat_gids):
            armid2gids[str(i + 1)] = [gid]

        gid2armid = dict()
        for gid in control_gids:
                gid2armid[str(gid)] = 0
        for i, gid in enumerate(sorted(treat_gids)):
                gid2armid[str(gid)] = i + 1

        self.num_arm = 1 + len(treat_gids)
        self.df["armid"] = self.df['groupid'].apply(
                lambda x: gid2armid[str(x)])

        unique_metric_ids = self.df['metric_id'].unique()
        metricid2dimen = {id: position for position, id in enumerate(unique_metric_ids)}

        print("data_processing...")
        armid2df, armid2nextidx = dict(), dict()
        pool = Pool(processes=16)
        for armid, df1 in self.df.groupby("armid"):
            tmp = pool.apply_async(processing_data, args=(armid, df1, metricid2dimen))
            pool_res.append(tmp)
        pool.close()
        pool.join()

        for item in pool_res:
            d0, d1, d2 = item.get()
            armid2df[d0] = d1
            armid2nextidx[d0] = d2
        # armid2df[str(armid)] = df1.groupby('uin').apply(lambda x: aggregate_numerator(x, metricid2dimen))
        # armid2nextidx[str(armid)] = 0

        armid2mean, armid2var, armid2cnt = dict(), dict(), dict()
        for armid, df in armid2df.items():
            armid2cnt[armid] = df['numerator'].count()
        self.mincnt = min(armid2cnt.values())

        control_mean = np.mean(armid2df["0"]["numerator"][0:self.mincnt].tolist(), axis=0)
        self.num_dimension = len(control_mean)

        for armid in armid2df.keys():
            df = armid2df[armid][0:self.mincnt]
            armid2mean[armid] = np.mean([item for item in df["numerator"]], axis=0).tolist()
            armid2var[armid] = np.var([item for item in df["numerator"]], axis=0, ddof=1).tolist()
            armid2cnt[armid] = df['numerator'].count()

        self.means = np.array(list(armid2mean.values()))
        self.vars = np.array(list(armid2var.values()))
        self.cnts = np.array(list(armid2cnt.values()))
        max_i, max_j = 0, 1
        if self.reward_type == "bernoulli":
                gaps, max_gap = [], 0
                for i in range(len(self.means)):
                    for j in range(i + 1, len(self.means)):
                        cur_gap = abs(self.means[i] - self.means[j])
                        gaps.append(cur_gap)
                        if (cur_gap > max_gap).all():
                            max_i, max_j = i, j
                            max_gap = cur_gap
        armidtrans = {
            "0": max_i,
            "1": max_j
        }
        if self.reward_type == "bernoulli":
            self.gap_mean,  self.gap_var = abs(
             self.means[max_i] - self.means[max_j]), np.array([0.05 for i in range(self.num_dimension)])
        else:
            gaps = []
            for i in range(len(self.means)):
                for j in range(i + 1, len(self.means)):
                    gaps.append(abs(self.means[i] - self.means[j]))
            self.gap_mean = np.mean(gaps, axis=0)
            self.gap_var = np.var(gaps, axis=0)

        self.arms_rewards = []
        for arms in armid2df.values():
            self.arms_rewards.append(np.array(list(arms["numerator"])))

    def getarms(self, objectives, total_user_each_arm):
        arms = []
        if objectives != None:
            indices = np.random.choice(self.mincnt, size=total_user_each_arm*self.num_arm, replace=False)
            for i in range(self.num_arm):
                arms.append(RealArm(i, self.reward_type, len(objectives), self.means[i][objectives], self.vars[i][objectives], self.arms_rewards[i][np.ix_(indices, objectives)]))
            return arms
        else:
            indices = np.random.choice(self.mincnt, size=total_user_each_arm, replace=False)
            for i in range(self.num_arm):
                arms.append(RealArm(i, self.reward_type, self.num_dimension, self.means[i], self.vars[i], self.arms_rewards[i][indices, :]))
            return arms
