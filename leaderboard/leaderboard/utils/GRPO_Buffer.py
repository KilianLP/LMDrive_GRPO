import torch
import numpy as np


class GRPOBuffer:
    def __init__(self, n_try):
        self.n_try = n_try
        
        self.buffer_list = []
        for _ in range(self.n_try):
            self.buffer_list.append(SingleGRPOBuffer())

        self.adv_buf = []

    def store(self, obs, act, rew, logp, pred_done, pred_done_logp, done, i_try):
        self.buffer_list[i_try].store(obs, act, rew, logp, pred_done, pred_done_logp, done)

    def compute(self):
        data = []
        final_reward = []
        done_idxs = []

        for _ in range(self.n_try):
            data.append(self.buffer_list.get())
            final_reward.append(data[-1]["final_reward"])
            done_idxs.append(data[-1]["done_idxs"])
        
        mean_reward = np.mean(np.array(final_reward), dim=0)
        std_reward = np.std(np.array(final_reward), dim=0)
        
        adv = (final_reward - mean_reward) / std_reward
        
        done_idxs = [0] + done_idxs
        for i in range(len(data[run]["adv"])):
            data[i]["adv"] = []
            for run in range(1,len(done_idxs)):
                data[i]["adv"] += adv[run]*np.ones(done_idxs[run]-done_idxs[run-1])

        return data

    def get(self):
        data = self.compute()
        torch_data = dict()

        torch_data["obs"] = []
        torch_data["act"] = []
        torch_data["logp"] = []
        torch_data["done"] = []
        torch_data["pred_done"] = []
        torch_data["pred_done_logp"] = []
        torch_data["adv"] = []


        for run in data:
            torch_data["obs"] += run["obs"]
            torch_data["act"] += run["act"]
            torch_data["logp"] += run["logp"]
            torch_data["done"] += run["done"]
            torch_data["pred_done"] += run["pred_done"]
            torch_data["pred_done_logp"] += run["pred_done_logp"]
            torch_data["adv"] += run["adv"]

        torch_data["obs"] = torch.tensor(torch_data["obs"], dtype=torch.float32)
        torch_data["act"] = torch.tensor(torch_data["act"], dtype=torch.float32)
        torch_data["logp"] = torch.tensor(torch_data["logp"], dtype=torch.float32)
        torch_data["done"] = torch.tensor(torch_data["done"], dtype=torch.float32)
        torch_data["pred_done"] = torch.tensor(torch_data["pred_done"], dtype=torch.float32)
        torch_data["pred_done_logp"] = torch.tensor(torch_data["pred_done_logp"], dtype=torch.float32)
        torch_data["adv"] = torch.tensor(torch_data["adv"], dtype=torch.float32)

        return torch_data



class SingleGRPOBuffer:
    def __init__(self):
        self.obs_buf = []
        self.act_buf = []
        self.rew_buf = []
        self.logp_buf = []
        self.done_buf = []
        self.pred_done_buf = []
        self.pred_done_logp_buf = []

        self.done_idxs = None
        self.final_reward = None

    def store(self, obs, act, rew, logp, pred_done, pred_done_logp, done):
        self.obs_buf.append(obs)
        self.act_buf.append(act)
        self.rew_buf.append(rew)
        self.logp_buf.append(logp)
        self.done_buf.append(done)  
        self.pred_done_buf.append(pred_done)
        self.pred_done_logp_buf.append(pred_done_logp)

    def compute(self):
        self.done_idxs = np.where(self.done_buf)[0]
        self.final_reward = self.rew_buf[self.done_idxs]

    def get(self):

        self.compute()

        data = dict()
        data["obs"] = self.obs_buf
        data["act"] = self.act_buf
        data["logp"] = self.logp_buf
        data["rew"] = self.rew_buf
        data["done"] = self.done_buf
        data["pred_done"] = self.pred_done_buf
        data["pred_done_logp"] = self.pred_done_logp_buf
        data["final_reward"] = self.final_reward
        data["done_idxs"] = self.done_idxs

        self.clear()
        return data

    def clear(self):
        self.__init__()
