"""Soft actor-critic on small MLPs, for a learner that runs on the CPU beside the arm.

Observations are flat vectors (the base policy's features and the joints);
actions are in ``[-1, 1]`` and each method maps them to what it changes (a
residual in degrees, or flow noise). The actor's mean head starts at zero, so the
first actions are the base policy's own.
"""

from __future__ import annotations

import copy
import threading
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MIN, LOG_STD_MAX = -5.0, 1.0


@dataclass
class SACConfig:
    obs_dim: int
    action_dim: int
    hidden: tuple[int, ...] = (256, 256)
    gamma: float = 0.97  # per decision; at 3 decisions/s a 15 s episode is 45 of them
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    init_alpha: float = 0.05
    target_entropy: float | None = None  # default -action_dim
    batch_size: int = 128
    buffer_size: int = 200_000


def _mlp(inputs: int, hidden: tuple[int, ...], outputs: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    for width in hidden:
        layers += [nn.Linear(inputs, width), nn.LayerNorm(width), nn.ReLU()]
        inputs = width
    layers.append(nn.Linear(inputs, outputs))
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, config: SACConfig) -> None:
        super().__init__()
        self.net = _mlp(config.obs_dim, config.hidden, 2 * config.action_dim)
        last = self.net[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        with torch.no_grad():  # start with a small spread: log_std = -1.5
            last.bias[config.action_dim:] = -1.5

    def forward(self, obs: torch.Tensor, *, deterministic: bool = False, std_scale: float = 1.0):
        mean, log_std = self.net(obs).chunk(2, dim=-1)
        log_std = log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp() * std_scale
        if deterministic or std_scale == 0:
            return torch.tanh(mean), None
        noise = torch.randn_like(mean)
        pre = mean + std * noise
        action = torch.tanh(pre)
        log_prob = (
            (-0.5 * noise.pow(2) - log_std - 0.5 * np.log(2 * np.pi)).sum(-1)
            - torch.log1p(-action.pow(2) + 1e-6).sum(-1)
        )
        return action, log_prob


class Critics(nn.Module):
    def __init__(self, config: SACConfig) -> None:
        super().__init__()
        self.q1 = _mlp(config.obs_dim + config.action_dim, config.hidden, 1)
        self.q2 = _mlp(config.obs_dim + config.action_dim, config.hidden, 1)

    def forward(self, obs: torch.Tensor, action: torch.Tensor):
        x = torch.cat((obs, action), dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


class ReplayBuffer:
    def __init__(self, obs_dim: int, action_dim: int, size: int) -> None:
        self.obs = np.zeros((size, obs_dim), np.float32)
        self.next_obs = np.zeros((size, obs_dim), np.float32)
        self.action = np.zeros((size, action_dim), np.float32)
        self.reward = np.zeros(size, np.float32)
        self.done = np.zeros(size, np.float32)
        self.size, self.capacity, self._i = 0, size, 0
        self._lock = threading.Lock()

    def add(self, obs, action, reward, next_obs, done) -> None:
        with self._lock:
            i = self._i
            self.obs[i], self.action[i], self.reward[i] = obs, action, reward
            self.next_obs[i], self.done[i] = next_obs, float(done)
            self._i = (i + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def sample(self, batch: int, rng: np.random.Generator) -> dict[str, torch.Tensor]:
        with self._lock:
            index = rng.integers(0, self.size, batch)
            arrays = {
                "obs": self.obs[index], "action": self.action[index], "reward": self.reward[index],
                "next_obs": self.next_obs[index], "done": self.done[index],
            }
        return {k: torch.from_numpy(v.copy()) for k, v in arrays.items()}

    def state_dict(self) -> dict[str, np.ndarray]:
        n = self.size
        return {"obs": self.obs[:n], "next_obs": self.next_obs[:n], "action": self.action[:n],
                "reward": self.reward[:n], "done": self.done[:n]}

    def load_state_dict(self, state: dict[str, np.ndarray]) -> None:
        for name in ("obs", "next_obs", "action", "reward", "done"):
            values = state[name][-self.capacity:]
            getattr(self, name)[: len(values)] = values
        self.size = len(state["reward"][-self.capacity:])
        self._i = self.size % self.capacity


class SAC:
    """Learner state; ``policy`` is a copy the actor thread reads, refreshed by
    ``publish``."""

    def __init__(self, config: SACConfig, seed: int = 0) -> None:
        torch.manual_seed(seed)
        self.config = config
        self.actor = Actor(config)
        self.critics = Critics(config)
        self.target = copy.deepcopy(self.critics).requires_grad_(False)
        self.log_alpha = torch.tensor(np.log(config.init_alpha), dtype=torch.float32, requires_grad=True)
        self.target_entropy = (
            -float(config.action_dim) if config.target_entropy is None else config.target_entropy
        )
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critics.parameters(), lr=config.critic_lr)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=config.alpha_lr)
        self.buffer = ReplayBuffer(config.obs_dim, config.action_dim, config.buffer_size)
        self.rng = np.random.default_rng(seed)
        self.steps = 0
        self._policy_lock = threading.Lock()
        self.policy = copy.deepcopy(self.actor).requires_grad_(False).eval()

    def act(self, obs: np.ndarray, *, std_scale: float = 1.0) -> np.ndarray:
        """An action from the published policy; ``std_scale`` 0 is its mean."""
        with self._policy_lock, torch.no_grad():
            action, _ = self.policy(
                torch.as_tensor(obs, dtype=torch.float32)[None], std_scale=std_scale,
                deterministic=std_scale == 0,
            )
        return action[0].numpy()

    def publish(self) -> None:
        with self._policy_lock:
            self.policy.load_state_dict(self.actor.state_dict())

    def update(self) -> dict[str, float]:
        c = self.config
        batch = self.buffer.sample(c.batch_size, self.rng)
        alpha = self.log_alpha.exp().detach()
        with torch.no_grad():
            next_action, next_logp = self.actor(batch["next_obs"])
            q1_t, q2_t = self.target(batch["next_obs"], next_action)
            target = batch["reward"] + c.gamma * (1 - batch["done"]) * (
                torch.min(q1_t, q2_t) - alpha * next_logp
            )
        q1, q2 = self.critics(batch["obs"], batch["action"])
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        action, logp = self.actor(batch["obs"])
        q = torch.min(*self.critics(batch["obs"], action))
        actor_loss = (alpha * logp - q).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        with torch.no_grad():
            for p, p_t in zip(self.critics.parameters(), self.target.parameters()):
                p_t.mul_(1 - c.tau).add_(c.tau * p)
        self.steps += 1
        return {"critic": float(critic_loss.detach()), "actor": float(actor_loss.detach()),
                "alpha": float(alpha), "q": float(q.mean().detach())}

    def state_dict(self) -> dict:
        return {
            "config": asdict(self.config), "actor": self.actor.state_dict(),
            "critics": self.critics.state_dict(), "target": self.target.state_dict(),
            "log_alpha": self.log_alpha.detach(), "steps": self.steps,
            "buffer": self.buffer.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.actor.load_state_dict(state["actor"])
        self.critics.load_state_dict(state["critics"])
        self.target.load_state_dict(state["target"])
        with torch.no_grad():
            self.log_alpha.copy_(state["log_alpha"])
        self.steps = int(state["steps"])
        self.buffer.load_state_dict(state["buffer"])
        self.publish()


class LearnerThread(threading.Thread):
    """Runs ``utd`` updates per new transition in the background and publishes
    the actor every ``publish_every`` updates."""

    def __init__(self, sac: SAC, *, utd: int = 4, warmup: int = 256, publish_every: int = 50) -> None:
        super().__init__(daemon=True)
        self.sac, self.utd, self.warmup, self.publish_every = sac, utd, warmup, publish_every
        self._owed = 0
        self._cv = threading.Condition()
        self._stop_event = threading.Event()
        self.last: dict[str, float] = {}

    def transitions_added(self, n: int = 1) -> None:
        with self._cv:
            self._owed += n * self.utd
            self._cv.notify()

    def stop(self) -> None:
        self._stop_event.set()
        with self._cv:
            self._cv.notify()

    def run(self) -> None:
        while not self._stop_event.is_set():
            with self._cv:
                while not self._stop_event.is_set() and (
                    self._owed <= 0 or self.sac.buffer.size < self.warmup
                ):
                    self._cv.wait(timeout=0.5)
                if self._stop_event.is_set():
                    return
                self._owed -= 1
            self.last = self.sac.update()
            if self.sac.steps % self.publish_every == 0:
                self.sac.publish()
