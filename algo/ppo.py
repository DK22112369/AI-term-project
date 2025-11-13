"""Simple PPO agent skeleton for continuous actions.

This is a minimal, easy-to-read PPO skeleton intended for development and
smoke-testing. The update() method is a placeholder and should be expanded
for production training.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes=(64, 64)):
        super().__init__()
        layers = []
        last = state_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            last = h
        layers.append(nn.Linear(last, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_sizes=(64, 64)):
        super().__init__()
        layers = []
        last = state_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            last = h
        layers.append(nn.Linear(last, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class PPOAgent:
    def __init__(self, state_dim, action_dim, config):
        self.device = config.device
        self.policy_net = PolicyNetwork(state_dim, action_dim).to(self.device)
        # log_std parameterized as independent per action dim
        self.log_std = nn.Parameter(torch.zeros(action_dim, dtype=torch.float32, device=self.device))
        self.value_net = ValueNetwork(state_dim).to(self.device)

        self.policy_optimizer = torch.optim.Adam(list(self.policy_net.parameters()) + [self.log_std], lr=config.policy_lr)
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=config.value_lr)

        self.clip_param = config.clip_param
        self.gamma = config.gamma
        self.gae_lambda = config.gae_lambda
        self.ppo_epochs = config.ppo_epochs
        self.batch_size = config.batch_size
        # loss coeffs
        self.entropy_coef = getattr(config, "entropy_coef", 0.01)
        self.value_loss_coef = getattr(config, "value_loss_coef", 0.5)

    def select_action(self, state):
        """Given a numpy state, return (action, log_prob)."""
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        mean = self.policy_net(state_t)
        std = torch.exp(self.log_std)
        dist = Normal(mean, std)
        action = dist.sample()
        action_clipped = torch.clamp(action, -1.0, 1.0)
        log_prob = dist.log_prob(action).sum(dim=-1)

        return action_clipped.detach().cpu().numpy().ravel(), float(log_prob.detach().cpu().numpy().ravel())

    def compute_gae(self, rewards, values, dones, next_value):
        advantages = []
        gae = 0
        values = np.append(values, next_value)
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step + 1] * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[step]) * gae
            advantages.insert(0, gae)
        return np.array(advantages, dtype=np.float32)

    def update(self, rollout_buffer):
        """Perform PPO update using data in rollout_buffer.

        rollout_buffer should provide lists: states, actions, log_probs, rewards,
        dones, next_states. This implementation flattens the buffer and runs
        multiple PPO epochs with minibatches.
        """
        # convert to tensors
        states = torch.tensor(np.asarray(rollout_buffer.states, dtype=np.float32), device=self.device)
        actions = torch.tensor(np.asarray(rollout_buffer.actions, dtype=np.float32), device=self.device)
        old_log_probs = torch.tensor(np.asarray(rollout_buffer.log_probs, dtype=np.float32), device=self.device)
        rewards = np.asarray(rollout_buffer.rewards, dtype=np.float32)
        dones = np.asarray(rollout_buffer.dones, dtype=np.float32)

        # compute values and next_value for GAE
        with torch.no_grad():
            values = self.value_net(states).cpu().numpy()

        # next value: use last next_state if available
        if len(rollout_buffer.next_states) > 0:
            last_next = np.asarray(rollout_buffer.next_states[-1], dtype=np.float32)
            with torch.no_grad():
                next_value = float(self.value_net(torch.tensor(last_next, dtype=torch.float32, device=self.device).unsqueeze(0)).cpu().numpy().ravel()[0])
        else:
            next_value = 0.0

        # GAE advantages
        advantages = self.compute_gae(rewards, values, dones, next_value)
        returns = advantages + values

        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        advantages_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        # normalize advantages
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        dataset_size = states.shape[0]
        batch_size = min(self.batch_size, dataset_size)

        for epoch in range(self.ppo_epochs):
            perm = np.random.permutation(dataset_size)
            for start in range(0, dataset_size, batch_size):
                idx = perm[start:start + batch_size]
                b_states = states[idx]
                b_actions = actions[idx]
                b_old_logp = old_log_probs[idx]
                b_adv = advantages_t[idx]
                b_ret = returns_t[idx]

                # new policy distribution
                mean = self.policy_net(b_states)
                std = torch.exp(self.log_std)
                dist = Normal(mean, std)
                new_logp = dist.log_prob(b_actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()

                ratio = torch.exp(new_logp - b_old_logp)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * b_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # value loss
                value_pred = self.value_net(b_states)
                value_loss = nn.MSELoss()(value_pred, b_ret)

                # update policy
                self.policy_optimizer.zero_grad()
                (policy_loss - self.entropy_coef * entropy).backward()
                self.policy_optimizer.step()

                # update value
                self.value_optimizer.zero_grad()
                (self.value_loss_coef * value_loss).backward()
                self.value_optimizer.step()

