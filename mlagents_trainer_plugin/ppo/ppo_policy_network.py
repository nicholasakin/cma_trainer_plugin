# policy_network.py
import torch
import torch.nn as nn
from torch.distributions import Categorical

class ActorCriticNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_units=128, num_layers=2):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_units))
            layers.append(nn.ReLU())
            in_dim = hidden_units
        self.shared = nn.Sequential(*layers)
        # Policy head
        self.policy_logits = nn.Linear(hidden_units, action_dim)
        # Value head
        self.value_head = nn.Linear(hidden_units, 1)

    def forward(self, x):
        """
        Given a batch of observations x (shape [N, obs_dim]),
        returns a Categorical distribution and a value tensor [N, 1].
        """
        x = self.shared(x)
        logits = self.policy_logits(x)
        dist = Categorical(logits=logits)
        value = self.value_head(x)
        return dist, value

    def evaluate(self, obs, actions):
        """
        Used in the optimizer: recompute log-probs & values for a batch.
        """
        dist, value = self.forward(obs)
        return dist, value
