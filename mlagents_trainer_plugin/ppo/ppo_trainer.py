import torch
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from torch.utils.tensorboard import SummaryWriter
from ppo_optimizer import PPOCustomOptimizer
from ppo_policy_network import ActorCriticNetwork

class PPOTrainer:
    def __init__(self, config_yaml: str, env_path: str):
        # Load YAML config yourself or pass in a dict
        import yaml
        settings = yaml.safe_load(open(config_yaml))["behaviors"]["3DBall"]
        self.hp = settings['hyperparameters']
        rc = settings['reward_signals']['extrinsic']
        self.gamma = rc['gamma']
        # Build environment with no graphics for speed
        channel = EngineConfigurationChannel()
        channel.set_configuration_parameters(time_scale=20.0)
        self.env = UnityEnvironment(
            file_name=env_path,
            side_channels=[channel]
        )
        self.env.reset()
        spec = list(self.env.behavior_specs.values())[0]
        obs_shape = spec.observation_shapes[0][0]
        action_size = spec.action_spec.discrete_branches[0]
        # Build policy network
        self.policy = ActorCriticNetwork(obs_shape, action_size)
        self.optimizer = PPOCustomOptimizer(self.policy, types.SimpleNamespace(hyperparameters=self.hp, reward_signals=types.SimpleNamespace(extrinsic=types.SimpleNamespace(gamma=self.gamma))),)
        self.time_horizon = settings['time_horizon']
        self.max_steps = settings['max_steps']
        self.writer = SummaryWriter(log_dir="./ppo_logs")

    def collect_rollout(self):
        spec = list(self.env.behavior_specs.values())[0]
        decision_steps, terminal_steps = self.env.get_steps(list(self.env.behavior_specs.keys())[0])
        obs_list, action_list, logp_list, reward_list, done_list, value_list = [], [], [], [], [], []
        step=0
        while step < self.time_horizon:
            # Get current obs
            decision_steps, terminal_steps = self.env.get_steps(list(self.env.behavior_specs.keys())[0])
            obs = torch.from_numpy(decision_steps.obs[0]).float()
            dist, value = self.policy.forward(obs)
            action = dist.sample()
            logp = dist.log_prob(action)
            # Step the environment
            action_tuple = spec.action_spec.empty_action(n_agents=len(decision_steps))
            action_tuple.add_discrete(action.cpu().numpy())
            self.env.set_actions(list(self.env.behavior_specs.keys())[0], action_tuple)
            self.env.step()
            # Record
            obs_list.append(obs)
            action_list.append(action)
            logp_list.append(logp)
            value_list.append(value)
            # Rewards & dones
            next_decision, next_terminal = self.env.get_steps(list(self.env.behavior_specs.keys())[0])
            reward = torch.tensor([r.reward for r in next_decision + next_terminal], dtype=torch.float32)
            done = torch.tensor([1 if ts else 0 for ts in next_terminal], dtype=torch.float32)
            reward_list.append(reward)
            done_list.append(done)
            step += 1
        # After horizon, get value for last state
        _, last_value = self.policy.forward(obs_list[-1])
        value_list.append(last_value)
        # Stack tensors
        rollout = {
            'obs': torch.vstack(obs_list),
            'actions': torch.cat(action_list),
            'log_probs': torch.cat(logp_list),
            'rewards': torch.cat(reward_list),
            'dones': torch.cat(done_list),
            'values': torch.cat(value_list),
        }
        return rollout

    def train(self):
        total_steps = 0
        while total_steps < self.max_steps:
            rollout = self.collect_rollout()
            self.optimizer.update(rollout)
            total_steps += len(rollout['rewards'])
            # Logging
            if total_steps % self.hp['summary_freq'] == 0:
                self.writer.add_scalar('reward/mean', rollout['rewards'].mean().item(), total_steps)
        # Save final model
        torch.save(self.policy.state_dict(), "ppo_final.pt")

if __name__ == '__main__':
    import sys
    trainer = PPOTrainer(sys.argv[1], sys.argv[2])
    trainer.train()