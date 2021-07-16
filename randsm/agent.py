import torch
from stable_baselines3.common.utils import set_random_seed


class Agent(object):
    def __init__(self, model):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.model.eval()


def set_randsm_seed(seed=0, env=None):
    set_random_seed(seed)
    try:
        env.seed(seed)
        env.action_space.seed(seed)
    except:
        pass