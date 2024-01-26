from Arm import Arm
import numpy as np
import pandas as pd
import time
def aggregate_numerator(df):
    return pd.DataFrame({
        "\"numerator\"": [df["\"numerator\""].tolist()],
    })
class RealArm(Arm):
    def __init__(self,arm_id, reward_type, D, mean_list, variance_list, real_rewards):
        super().__init__(arm_id,reward_type,D,mean_list,variance_list)
        self.rewards_index = -1
        self.real_rewards = real_rewards
    def get_reward(self):
        if self.rewards_index >= len(self.real_rewards) - 1:
            return None
        self.rewards_index += 1
        return self.real_rewards[self.rewards_index]
    
class RealArms():
    def __init__(self, fname, reward_type) -> None:
        self.fname = fname
        self.reward_type = reward_type

        self.df = pd.read_csv(self.fname)
        self.df['"numerator"'] = self.df['"numerator"'] / self.df['"denominator"']
        tmp = self.df.groupby(['"exptid"', '"groupid"']).count().reset_index()
        gids, gid2uv = [], {}
        for i, row in tmp.iterrows(): 
            gids.append(row['"groupid"'])
            gid2uv[str(row['"groupid"'])] = row['"uin"']
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
        self.df["armid"] = self.df['"groupid"'].apply(
                lambda x: gid2armid[str(x)])
        armid2df, armid2nextidx = dict(), dict()
        for armid, df1 in self.df.groupby("armid"):
            armid2df[str(armid)] = df1.groupby('"uin"').apply(aggregate_numerator).reset_index()
            armid2nextidx[str(armid)] = 0

        control_mean = np.mean([item for item in armid2df["0"]["\"numerator\""]], axis=0).tolist()
        self.num_dimension = len(control_mean)

        armid2mean, armid2var, armid2cnt = dict(), dict(), dict()
        for armid, df in armid2df.items():
            armid2cnt[armid] = df['"numerator"'].count()
        mincnt = min(armid2cnt.values()) 
        for armid in armid2df.keys():
            df = armid2df[armid][0:mincnt]
            armid2df[armid] = df.sample(frac=1, random_state=int(time.time())).reset_index(drop=True)
            armid2mean[armid] = np.mean([item for item in df["\"numerator\""]], axis=0).tolist()
            armid2var[armid] = np.var([item for item in df["\"numerator\""]], axis=0).tolist()
            armid2cnt[armid] = df['"numerator"'].count()

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
            self.arms_rewards.append(np.array(list(arms["\"numerator\""])))

    def getarms(self):
        arms = []
        for i in range(self.num_arm):
            arms.append(RealArm(i, self.reward_type, self.num_dimension, self.means[i], self.vars[i], self.arms_rewards[i]))
        return arms


    