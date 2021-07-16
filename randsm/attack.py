import torch
import numpy as np
from advertorch.attacks import LinfPGDAttack, L2PGDAttack

from .agent import Agent
from .model import OnlyObsSingleActionModel

# SCAL = np.array([4.8, 4.8   , 0.418, 4.8   ]) #
# LIMI = np.array([2.4, np.inf, 0.209, np.inf])


class LLLLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inp, target):
        return - (inp ** 2).mean() # - ((inp - torch.clip(target, -2, 2)) ** 2).mean()

class LogitLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inp, target):
        # - torch.max(inp, dim=1)[0].mean()
        return - torch.gather(inp, 1, target).mean()


class SimpleAttack(Agent):
    PRE_DICT = {'ori': True, 'per': False, True: True, False: False}
    POS_DICT = {False: 0, True: 1, 0: 0, 1: 1, 2: 2, 3: 0, 4: 1, 5: 2}

    def __init__(self, model, pre_flag='ori', pos_flag=0, eps=0.05, nb_iter=81, eps_iter=0.01, rand_init=False, clip=None, scaler=None, num_classes=2, batch_size=50, loss=None): #
        super().__init__(OnlyObsSingleActionModel(model.policy, num_classes=num_classes, scaler=scaler, batch_size=batch_size))
        self.label_flag = True

        self.loss = loss
        if not loss:
            # self.label_flag = False
            self.loss = LogitLoss()
        else:
            self.loss = LLLLoss()

        self.pre_flag = SimpleAttack.PRE_DICT[pre_flag]
        self.pos_flag = SimpleAttack.POS_DICT[pos_flag]

        self.eps = eps
        if nb_iter > 10: #
            self.nb_iter = nb_iter % 10
            nb_iter = nb_iter // 10
        else:
            self.nb_iter = nb_iter

        self.adversary = L2PGDAttack(self.model, loss_fn=self.loss, eps=eps, nb_iter=nb_iter,
            eps_iter=eps_iter, rand_init=rand_init, clip_min=0.0, clip_max=1.0, targeted=False)

        adv_clip = clip.copy()
        adv_clip[clip < np.inf] = 0.5 # == + eps
        self.adversary.clip_min = - torch.FloatTensor(adv_clip).to(self.device) if not np.all(clip < np.inf) else -0.5
        self.adversary.clip_max =   torch.FloatTensor(adv_clip).to(self.device) if not np.all(clip < np.inf) else  0.5

        self.scaler = scaler
        self.clip_max = clip
        self.clip_min = - clip
        self.perturbation = None
        self.pre_perturbation = None
        # self.real_model = model

    def pre_perturb(self, img):
        if self.pre_perturbation is None:
            return img
        return np.clip(img + self.pre_perturbation, self.clip_min, self.clip_max)

    def preprocess (self, obs):
        return obs if self.pre_flag      else self.pre_perturb(obs)

    def posuprocess(self, obs):
        return obs if self.pos_flag <= 1 else self.pre_perturb(obs)

    def postprocess(self, obs, adv):
        if self.pos_flag % 3 == 0:
            return self.pre_perturb(obs)
        if self.pos_flag % 3 == 1:
            return adv
        return obs

    def attack(self, obs,    count):
        if count == 0:
            self.pre_perturbation = None
        if (count + 1) % self.nb_iter:
            return 0, self.posuprocess(obs)

        rgb = self.preprocess(obs)

        with torch.no_grad():
            # rgb = rgb * np.sqrt(np.array([0.31303893, 0.12230581, 3.50250079]) + 1e-8) + np.array([0.75143003, 0.00285801, 0.0622901])
            _rgb = torch.FloatTensor( rgb / self.scaler).to(self.device)

            model_pred = self.model.predict(_rgb)
            # print(model_pred,self.model.true_forward(rgb/self.scaler))
            # model_pred = torch.max(self.model(_rgb),1)[1].unsqueeze(1)
            # a = self.model.true_forward(rgb / self.scaler)[0]
            # b = model_pred.detach().cpu().numpy()[0, 0]
            # c = self.real_model.predict(rgb, deterministic=True)[0][0]
            # if a != b or a != c or b != c:
            #     print(a, b, c, None)

        label = model_pred.detach() if self.label_flag else torch.FloatTensor([0])
        adv_img = self.adversary.perturb(_rgb,  label)

        adv_img = adv_img.detach().cpu().numpy() *self.scaler
        ret_img = self.postprocess(obs, adv_img)
        self.pre_perturbation = adv_img - rgb
        # self.perturbation = ret_img - obs #
        return 1, ret_img