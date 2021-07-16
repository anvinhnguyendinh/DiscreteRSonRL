import torch
import numpy as np

from math import ceil
from scipy.stats import norm, binom_test
from statsmodels.stats.proportion import proportion_confint

from .agent import Agent
from .model import OnlyObsSingleActionModel


class Smooth(Agent):
    ABSTAIN = -1
    MASTER  = 311295

    def __init__(self, base_classifier, num_classes, sigma, scaler, batch_size=None, action_space=None, continuous=False, is_hat=False): #
        super().__init__(OnlyObsSingleActionModel(base_classifier.policy, num_classes=num_classes, scaler=scaler))
        self.base_model = base_classifier
        self.base_classifier = self.model
        self.range = range(max(2, num_classes))
        self.num_classes = max(2, num_classes) #
        self.scaler = scaler
        self.sigma = sigma
        self.seeders = []

        self.mode = 0
        self.is_hat = is_hat

        if action_space is None:
            self.action_space = range(self.num_classes)
            self.incr = 1
        else:
            self.incr = (action_space[1] - action_space[0])  / (self.num_classes - 1)
            self.action_space = np.arange( action_space[0], action_space[1] + 1e-7, self.incr, dtype=action_space[2])
        self.map_to_acsp = lambda x: tuple(map(lambda a: self.action_space[a], x))
        self.batch_size  = batch_size
        self.current_master_seed = 0
        self.continuous = continuous

    def certify(self, x, n0, n, alpha, batch_size, seeder): #
        # draw samples of f(x + epsilon)
        counts_selection = self._sample_noise(x, n0, batch_size, seeder)
        # use these samples to take a guess at the top class
        cAHat = counts_selection.argmax().item()
        # draw more samples of f(x + epsilon)
        counts_estimation = self._sample_noise(x, n, batch_size, seeder)
        # use these samples to estimate a lower bound on pA
        nA = counts_estimation[cAHat].item()
        pABar = self._lower_confidence_bound(nA, n, alpha)
        if pABar < 0.5:
            return counts_estimation.argmax().item() if self.is_hat else Smooth.ABSTAIN, 0.0 #
        else:
            radius = self.sigma * norm.ppf(pABar)
            return cAHat, radius

    def predict(self, x, n, alpha, batch_size):
        counts = self._sample_noise(x, n, batch_size)
        top2 = counts.argsort()[::-1][:2]
        count1 = counts[top2[0]]
        count2 = counts[top2[1]]
        if binom_test(count1, count1 + count2, p=0.5) > alpha:
            return Smooth.ABSTAIN
        else:
            return top2[0]

    def _sample_noise(self, x, num, batch_size, seeder): #
        with torch.no_grad():
            counts = np.zeros(self.num_classes, dtype=int)
            batch_x = torch.FloatTensor(x / self.scaler).to(self.device)
            for _ in range(ceil(num / batch_size)):
                this_batch_size = min(batch_size, num)
                num -= this_batch_size

                batch = batch_x.repeat((this_batch_size, 1)) #
                # noise = torch.randn_like(batch, device=self.device) * self.sigma
                noise = torch.FloatTensor(seeder.standard_normal( batch.size()) * self.sigma).to(self.device)
                if not self.continuous:
                    predictions = self.base_classifier(batch + noise).argmax(1)
                else:
                    predictions = self.base_classifier(batch + noise).squeeze()
                    predictions = torch.clip(torch.round((predictions - self.action_space[0]) / self.incr), 0, self.num_classes -1).type(torch.int)
                counts += self._count_arr(predictions.cpu().numpy(), self.num_classes)
            return counts

    def _count_arr(self, arr, length):
        counts = np.zeros(length, dtype=int)
        for idx in arr:
            counts[idx] += 1
        return counts

    def _lower_confidence_bound(self, NA, N, alpha):
        return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[0]


    def master_seed(self, seed, num):
        del self.seeders
        ms = Smooth.MASTER * seed
        self.current_master_seed = ms
        self.seeders = [np.random.default_rng(ms + i) for i in range(num)]

    def certify_mode(self, x, seeder_num, step=0, n0=100, n=1000, alpha=0.01): # 0.001
        a,  r = self.certify(x, n0, n, alpha, self.batch_size, self.seeders[seeder_num])
        b,  _ = self.base_model.predict(x[0], deterministic=True)
        b = int(np.round((b - self.action_space[0]) / self.incr))

        if r != 0:
            return self.action_space[a], r, self.map_to_acsp((a, a, b))
        return  Smooth.ABSTAIN, r, self.map_to_acsp((a if self.is_hat  else self.seeders[seeder_num].choice(self.range), b, b))


class SmoothMultiA(Smooth):
    ABSTAIN = -1

    def __init__(self, base_classifier, num_classes, sigma, scaler, env_like, steps=2, batch_size=None, action_space=None, continuous=False): #
        super().__init__(base_classifier, num_classes, sigma, scaler, batch_size, action_space, continuous)
        self.steps = steps
        self.env_like = env_like
        self.actual_classes = num_classes ** steps
        self.first_divisor = num_classes ** (steps - 1)

        self.saved_radius = np.zeros((2, ))
        self.saved_action = np.zeros((2, steps) , dtype=action_space[2])
        self.saved_batch  = None

    def certify_hat(self, x, n0, n, alpha, batch_size, seeder):
        counts_selection = self._sample_noise(x, n0, batch_size, seeder)
        cAHat = counts_selection.argmax().item()
        counts_estimation = self._sample_noise(x, n, batch_size, seeder)
        nA = counts_estimation[cAHat].item()
        pABar = self._lower_confidence_bound(nA, n, alpha)
        if pABar < 0.5:
            return counts_estimation.argmax().item(), 0.0
        else:
            radius = self.sigma * norm.ppf(pABar)
            return cAHat, radius

    def _sample_noise(self, x, num, batch_size, seeder):
        with torch.no_grad():
            scaler = self.model.scaler
            counts = np.zeros(self.actual_classes, dtype=int)
            batch_x = torch.FloatTensor(x / self.scaler).to(self.device)
            for _ in range(ceil(num / batch_size)):
                this_batch_size = min(batch_size, num)
                num -= this_batch_size

                batch = batch_x.repeat((this_batch_size, 1))
                final_preds = torch.zeros(this_batch_size).to(self.device)
                size_tensor = batch.size()

                for i in range(self.steps):
                    noise = torch.FloatTensor(seeder.standard_normal( size_tensor) * self.sigma).to(self.device)
                    final_preds = final_preds * self.num_classes

                    if not self.continuous:
                        predictions = self.base_classifier(batch + noise).argmax(1) if i == 0 else self.base_classifier(batch).argmax(1)
                        preds = predictions.detach().clone()
                        predictions = preds * self.incr  + self.action_space[0]
                    else:
                        predictions = self.base_classifier(batch + noise).squeeze() if i == 0 else self.base_classifier(batch).squeeze()
                        predictions = torch.clip(predictions, self.action_space[0], self.action_space[-1]) ##
                        preds = torch.round((predictions - self.action_space[0]) / self.incr)
                        # predictions = preds * self.incr  + self.action_space[0]
                    # if i == 0: # and self.saved_batch is not None:
                        # batch = self.env_like(self.saved_batch, predictions)[0]  / scaler
                        # batch = self.env_like((batch + noise) * scaler, predictions)[0]  / scaler # batch * scaler
                    if i + 1  < self.steps:
                        batch = self.env_like(batch * scaler, predictions)[0]  / scaler
                    final_preds = final_preds + preds

                counts += self._count_arr(final_preds.type(torch.int).cpu().numpy(), self.actual_classes)
            return counts

    def certify_mode(self, x, seeder_num, step=0, n0=100, n=1000, alpha=0.01):
        b,  _ = self.base_model.predict(x[0], deterministic=True)
        b = int(np.round((b - self.action_space[0]) / self.incr))
        senum = seeder_num % 2
        # n0 *= self.steps
        # n  *= self.steps

        if step == 0:
            saved_actions, self.saved_radius[senum] = self.certify(x, n0, n, alpha, self.batch_size, self.seeders[seeder_num])
            for i in range(self.steps):
                self.saved_action[senum, - i - 1 ]  = saved_actions % self.num_classes
                saved_actions //= self.num_classes
            # self.saved_batch = torch.FloatTensor(x).to(self.device).repeat((self.batch_size, 1)) if senum == 0 else None

        r  = self.saved_radius[senum]
        a  = int(self.saved_action[senum, step])

        if r != 0: # a != SmoothMultiA.ABSTAIN
            return self.action_space[a], r, self.map_to_acsp((a, a, b))
        return  SmoothMultiA.ABSTAIN, r,  self.map_to_acsp((a if self.is_hat else self.seeders[seeder_num].choice(self.range), b, b))


class SmoothReward(SmoothMultiA):
    ABSTAIN = -1

    def __init__(self, base_classifier, num_classes, sigma, scaler, env_like, steps=2, batch_size=None, action_space=None, continuous=False, reward_space=None): #
        super().__init__(base_classifier, int(continuous) , sigma , scaler, env_like , steps, batch_size, action_space , not continuous)
        self.range_batch = range(batch_size)
        self.cur_actions = None
        self.saved_n = 0

        self.reward_classes = num_classes
        if reward_space is None:
            self.reward_space = range(num_classes)
            self.reward_incr = 1
        else:
            self.reward_incr = (reward_space[1] - reward_space[0])  / (num_classes - 1)
            self.reward_space = np.arange(reward_space[0], reward_space[1] + 1e-7 , self.reward_incr, dtype=reward_space[2])

    def _sample_noise(self, x, num, batch_size, seeder):
        with torch.no_grad():
            scaler = self.model.scaler
            saved_flag = self.saved_n <= num
            counts = np.zeros(self.reward_classes, dtype=int)
            batch_x = torch.FloatTensor(x / self.scaler).to(self.device)
            for _ in range(ceil(num / batch_size)):
                this_batch_size = min(batch_size, num)
                num -= this_batch_size

                batch = batch_x.repeat((this_batch_size, 1))
                noise = torch.FloatTensor(seeder.standard_normal(batch.size()) * self.sigma).to(self.device)

                if not self.continuous:
                    actions = self.base_classifier(batch + noise).argmax(1)
                    rctions = actions * self.incr +  self.action_space[0] #
                else:
                    actions = self.base_classifier(batch + noise).squeeze()
                    actions = torch.clip(actions, self.action_space[0], self.action_space[-1])
                    rctions = actions

                preds = self.env_like(batch * scaler , rctions)[1] # batch * scaler 
                predictions = torch.round((preds   - self.reward_space[0]) / self.reward_incr).type(torch.int).cpu().numpy()

                if saved_flag:
                    self._count_actions(actions.cpu().numpy(), predictions)
                counts += self._count_arr(predictions, self.reward_classes)
            return counts

    def _count_actions(self, acts, preds):
        if not self.continuous:
            for i in self.range_batch:
                self.cur_actions[preds[i], acts[i]] += 1
        else:
            for i in self.range_batch:
                self.cur_actions[preds[i], 0] += acts[i]
                self.cur_actions[preds[i], 1] +=  1

    def certify_mode(self, x, seeder_num, step=0, n0=100, n=1000, alpha=0.01):
        if not self.continuous:
            self.cur_actions = np.zeros((self.reward_classes, self.num_classes))
        else:
            self.cur_actions = np.zeros((self.reward_classes, 2))
        self.saved_n = n

        c,  r = self.certify(x, n0, n, alpha, self.batch_size,  self.seeders[seeder_num])
        b,  _ = self.base_model.predict(x[0], deterministic=True)

        if not self.continuous:
            b = int(np.round((b - self.action_space[0]) / self.incr))
            a = SmoothReward.ABSTAIN if r == 0 else int(self.cur_actions[c].argmax())
            if r != 0:
                return self.action_space[a], r, self.map_to_acsp((a, a, b)),self.reward_space[c]
            return  a, r, self.map_to_acsp((self.seeders[seeder_num].choice(self.range),  b,  b)), c

        cur, b = lambda x: self.cur_actions[x, 0] / self.cur_actions[x, 1], float(b)
        a = SmoothReward.ABSTAIN if r == 0  else cur(c)

        if r != 0:
            return a, r, (a, a,  b), self.reward_space[c]
        return  a, r, (self.cur_actions[:, 0].sum() / self.cur_actions[:, 1].sum(), b, b), - c


class SmoothMultiR(SmoothReward):
    ABSTAIN = -1

    def __init__(self, base_classifier, num_classes, sigma, scaler, env_like, steps=2, batch_size=None , action_space=None , continuous=False, reward_space=None): #
        super().__init__(base_classifier, num_classes, sigma, scaler, env_like, steps, batch_size, action_space, continuous, reward_space)

        self.saved = np.zeros((2, steps))
        self.saved_reward = np.zeros((2,), dtype=np.int)
        self.reward_space *= steps
        self.reward_incr  *= steps

    def _sample_noise(self, x, num, batch_size, seeder):
        with torch.no_grad():
            scaler = self.model.scaler
            saved_flag = self.saved_n <= num
            counts = np.zeros(self.reward_classes, dtype=int)
            batch_x = torch.FloatTensor(x / self.scaler).to(self.device)
            for _ in range(ceil(num / batch_size)):
                this_batch_size = min(batch_size, num)
                num -= this_batch_size

                batch = batch_x.repeat((this_batch_size, 1))
                final_preds = torch.zeros(this_batch_size).to(self.device)
                size_tensor = batch.size()

                if not self.continuous:
                    final_acts = torch.zeros(this_batch_size, dtype=torch.int).to(self.device)
                else:
                    final_acts = torch.zeros(this_batch_size, self.steps).to(self.device)

                for i in range(self.steps):
                    noise = torch.FloatTensor(seeder.standard_normal( size_tensor) * self.sigma).to(self.device)

                    if not self.continuous:
                        final_acts = final_acts * self.num_classes
                        actions = self.base_classifier(batch + noise).argmax(1) if i == 0 else self.base_classifier(batch).argmax(1)
                        final_acts = final_acts + actions
                        actions = actions * self.incr +  self.action_space[0]
                    else:
                        actions = self.base_classifier(batch + noise).squeeze() if i == 0 else self.base_classifier(batch).squeeze()
                        actions = torch.clip(actions, self.action_space[0], self.action_space[-1])
                        final_acts[:, i] = actions.detach().clone()

                    batch, preds, _ = self.env_like((batch + noise) * scaler, actions) if i == 0 else self.env_like(batch  * scaler , actions) # batch  * scaler 
                    final_preds = final_preds +  preds
                    batch = batch   /  scaler

                predictions = torch.round((final_preds - self.reward_space[0]) / self.reward_incr).type(torch.int).cpu().numpy()

                if saved_flag:
                    self._count_actions(final_acts.cpu().numpy() , predictions)
                counts += self._count_arr(predictions, self.reward_classes)
            return counts

    def certify_mode(self, x, seeder_num, step=0, n0=100, n=1000, alpha=0.01):
        b,  _ = self.base_model.predict(x[0], deterministic=True)
        senum = seeder_num % 2
        # n0 *= self.steps
        # n  *= self.steps
        self.saved_n = n

        if not self.continuous:
            self.cur_actions = np.zeros((self.reward_classes, self.actual_classes))
        else:
            self.cur_actions = np.zeros((self.reward_classes, 2,  self.steps))

        if step == 0:
            c, r = self.certify(x, n0, n, alpha, self.batch_size, self.seeders[seeder_num])
            self.saved_reward[senum] = c
            self.saved_radius[senum] = r

            if not self.continuous:
                a = SmoothReward.ABSTAIN if r == 0 else int(self.cur_actions[c].argmax())
                for i in range(self.steps):
                    self.saved_action[senum, - i - 1] = a % self.num_classes
                    a //=  self.num_classes
            else:
                a = SmoothReward.ABSTAIN if r == 0 else self.cur_actions[c, 0] / self.cur_actions[c, 1]
                self.saved[senum] = self.cur_actions[:, 0].sum(0) / self.cur_actions[:, 1].sum(0)
                self.saved_action[senum] =  a

        c  = self.saved_reward[senum]
        r  = self.saved_radius[senum]
        a  = self.saved_action[senum, step]

        if not self.continuous:
            b = int(np.round((b - self.action_space[0]) / self.incr))
            if r != 0:
                return self.action_space[a], r, self.map_to_acsp((a, a, b)), self.reward_space[c]
            return  SmoothMultiR.ABSTAIN, r,  self.map_to_acsp((self.seeders[seeder_num].choice(self.range), b, b)), c

        b = float(b)
        if r != 0:
            return a, r, (a, a, b), self.reward_space[c]
        return  a, r, (self.saved[senum, step], b, b), - c
