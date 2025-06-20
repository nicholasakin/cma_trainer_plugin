import torch
import torch.nn.functional as F

def compute_gae(rewards, values, dones, gamma, lambd):
    """
    Generalized Advantage Estimation (GAE).
    rewards: [T]
    values: [T+1]
    dones:   [T]
    returns, advs: [T]
    """
    T = len(rewards)
    advs = torch.zeros(T, dtype=torch.float32)
    lastgaelam = 0
    for t in reversed(range(T)):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * values[t+1] * mask - values[t]
        advs[t] = lastgaelam = delta + gamma * lambd * mask * lastgaelam
    returns = advs + values[:-1]
    return returns, advs

class PPOCustomOptimizer:
    """Custom PPO optimizer implementing clipped surrogate loss."""
    def __init__(self, policy, settings):
        self.policy = policy
        hp = settings.hyperparameters
        self.clip_epsilon = getattr(hp, "epsilon", 0.2)
        self.entropy_coef = getattr(hp, "entropy_coef", 0.01)
        self.value_loss_coef = getattr(hp, "value_loss_coef", 0.5)
        self.gamma = getattr(settings, 'reward_signals').get('extrinsic').gamma
        self.lambd = hp.lambd
        self.num_epoch = hp.num_epoch
        self.optimizer = torch.optim.Adam(
            self.policy.network.parameters(),
            lr=hp.learning_rate,
            eps=1e-5
        )

    def update(self, rollout_buffer):
        """
        rollout_buffer should be a dict containing:
          'obs', 'actions', 'log_probs', 'rewards', 'dones', 'values'
        """
        obs = rollout_buffer['obs']            # [N, obs_dim]
        actions = rollout_buffer['actions']    # [N, action_dim] or [N]
        old_log_probs = rollout_buffer['log_probs']  # [N]
        rewards = rollout_buffer['rewards']    # [N]
        dones = rollout_buffer['dones']        # [N]
        values = rollout_buffer['values']      # [N+1]

        # Compute returns and advantages via GAE
        returns, advantages = compute_gae(
            rewards, values, dones,
            self.gamma, self.lambd
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Repeat for multiple epochs
        for _ in range(self.num_epoch):
            dist, value_preds = self.policy.evaluate(obs, actions)
            new_log_probs = dist.log_prob(actions)
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon,
                                 1.0 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(value_preds.squeeze(-1), returns)
            entropy_loss = -dist.entropy().mean()

            loss = policy_loss + self.value_loss_coef * value_loss + self.entropy_coef * entropy_loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.policy.network.parameters(),
                max_norm=0.5
            )
            self.optimizer.step()