import torch
import numpy as np

from gym import spaces
from stable_baselines3.dqn.policies import QNetwork
from sb3_contrib.qrdqn.policies import QuantileNetwork


class OnlyObsSingleActionModel(torch.nn.Module):
    def __init__(self, model, num_classes, scaler, batch_size=50):
        super().__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = torch.FloatTensor(scaler).to(self.device)
        # action = np.expand_dims(np.arange(num_classes), 1)

        # self.action = torch.FloatTensor(action).to(self.device)
        self.batch_size = batch_size
        self.model = model

        self.model_type = - 1
        if isinstance(model, QNetwork):
            self.model_type = 3
        elif isinstance(model, QuantileNetwork):
            self.model_type = 2
        elif isinstance(model.action_space, spaces.Box):
            self.model_type = 1
        elif isinstance(model.action_space, spaces.Discrete):
            self.model_type = 0

    def forward(self, rgb):
        # rgb_ = torch.tile(rgb, (self.batch_size, 1)) if rgb.size(0) == 1 else rgb
        _rgb = rgb * self.scaler

        if self.model_type <= 1:
            latent_pi, _, latent_sde = self.model._get_latent(_rgb)
            distribution = self.model._get_action_dist_from_latent(latent_pi, latent_sde)
            if self.model_type == 1:
                return distribution.distribution.mean
            return distribution.distribution.logits

        if self.mode > 2:
            return self.model.forward(_rgb)
        return self.model.forward(_rgb).mean(dim=1)

    def predict(self, rgb):
        values = self.forward( rgb)
        if self.model_type == 1:
            return values
        return values.argmax(dim=1).reshape(-1).unsqueeze(1)

    def true_forward(self, rgb):
        return self.model.predict(rgb * self.scaler.cpu().numpy() , deterministic=True)[0]

    def prob_forward(self, rgb):
        rgb_ = torch.tile(rgb, (self.batch_size, 1)) if rgb.size(0) == 1 else rgb

        latent_pi, _, latent_sde = self.model._get_latent(rgb_ * self.scaler)
        distribution = self.model._get_action_dist_from_latent(latent_pi, latent_sde)
        return distribution.distribution.probs