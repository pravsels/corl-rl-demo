"""The two ways the learner changes the base policy, behind one interface.

Every ``steps_per_decision`` control ticks the loop calls ``decide`` (run the
base policy, build SAC's observation) and then ``goals`` (turn SAC's action in
``[-1, 1]`` into joint goals for the next ticks). An action of zero gives the
base policy's own behaviour, which is where SAC's actor starts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from corl_rl_demo.rig import JOINTS

# Joints enter SAC's observation in units of this many degrees, and the time since
# the episode began in units of this many seconds.
STATE_SCALE = 90.0
TIME_SCALE = 15.0


def smooth(plan: np.ndarray, window: int) -> np.ndarray:
    """Moving average over ``window`` ticks, the ends held. A flow policy's chunk
    drawn from random noise zig-zags from tick to tick; at 30 Hz a 1.5 degree
    zig-zag is a 2700 deg/s^2 acceleration, well past the gate."""
    if window <= 1:
        return plan
    half = window // 2
    padded = np.concatenate((np.repeat(plan[:1], half, 0), plan, np.repeat(plan[-1:], window - 1 - half, 0)))
    kernel = np.ones(window) / window
    return np.stack([np.convolve(padded[:, j], kernel, mode="valid") for j in range(plan.shape[1])], axis=1)


@dataclass
class Decision:
    obs: np.ndarray  # SAC's observation
    state: np.ndarray  # joints, degrees
    context: Any  # what ``goals`` needs from the base policy


class ACTBase:
    """A trained LeRobot ACT checkpoint, frozen. ``plan`` returns its action
    chunk (absolute joint goals) and the mean of its transformer encoder's
    output tokens as the feature SAC sees."""

    def __init__(self, path: str | Path, device: str | None = None) -> None:
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.policies.factory import make_pre_post_processors

        from corl_rl_demo.agent.flow import default_device

        self.device = torch.device(device) if device else default_device()
        self.policy = ACTPolicy.from_pretrained(str(path)).to(self.device).eval()
        self.policy.config.device = str(self.device)
        self.pre, self.post = make_pre_post_processors(
            self.policy.config, pretrained_path=str(path),
            preprocessor_overrides={"device_processor": {"device": str(self.device)}},
        )
        self.cameras = [key.removeprefix("observation.images.") for key in self.policy.config.image_features]
        self.feature_dim = int(self.policy.config.dim_model)
        self._feature: torch.Tensor | None = None
        self.policy.model.encoder.register_forward_hook(self._keep_feature)

    def _keep_feature(self, _module, _inputs, output: torch.Tensor) -> None:
        self._feature = output.mean(dim=0)  # (tokens, batch, dim) -> (batch, dim)

    @torch.no_grad()
    def plan(self, robot_obs: dict[str, Any], state: np.ndarray, task: str) -> tuple[np.ndarray, np.ndarray]:
        from lerobot.policies.utils import prepare_observation_for_inference

        observation = {"observation.state": state.astype(np.float32)}
        for camera in self.cameras:
            observation[f"observation.images.{camera}"] = robot_obs[camera]
        batch = self.pre(prepare_observation_for_inference(observation, self.device, task, None))
        actions = self.post(self.policy.predict_action_chunk(batch))
        return self._feature[0].float().cpu().numpy(), actions[0].float().cpu().numpy()


class ResidualACT:
    """ACT acts; SAC adds a bounded offset of up to ``scale_deg`` degrees per
    joint to the ticks it executes."""

    name = "residual-act"

    def __init__(self, base: ACTBase, *, steps_per_decision: int, scale_deg: float = 5.0,
                 task: str = "Press the green button") -> None:
        self.base, self.k, self.scale, self.task = base, steps_per_decision, scale_deg, task
        self.smooth_ticks = 5
        self.obs_dim = base.feature_dim + 2 * len(JOINTS) + 1
        self.action_dim = len(JOINTS)

    def decide(self, robot_obs: dict[str, Any], state: np.ndarray, elapsed_s: float = 0.0) -> Decision:
        feature, chunk = self.base.plan(robot_obs, state, self.task)
        if chunk.shape[0] < self.k:
            raise ValueError(f"ACT predicts {chunk.shape[0]} steps, fewer than {self.k} per decision")
        obs = np.concatenate((feature, state / STATE_SCALE, (chunk[0] - state) / 10.0,
                              [elapsed_s / TIME_SCALE])).astype(np.float32)
        return Decision(obs, state, chunk)

    def goals(self, decision: Decision, action: np.ndarray) -> np.ndarray:
        plan = smooth(decision.context, self.smooth_ticks) + self.scale * np.asarray(action)[None, :]
        self.tail = plan[self.k:]  # what this plan does next, for blending into the next one
        return plan[: self.k]


class DSRLFlow:
    """The flow policy acts from a noise SAC chooses: ``noise_steps`` rows of
    noise in ``[-bound, bound]``, the last repeated over the chunk."""

    name = "dsrl-flow"

    def __init__(self, flow: Any, *, steps_per_decision: int, noise_steps: int = 5,
                 bound: float = 1.5) -> None:
        horizon = flow.config.horizon
        if not 1 <= noise_steps <= horizon or steps_per_decision > horizon:
            raise ValueError("noise_steps and steps_per_decision must fit in the flow's horizon")
        self.flow, self.k, self.noise_steps, self.bound = flow, steps_per_decision, noise_steps, bound
        # For running the base policy alone, ignoring SAC: "random" draws a fresh
        # N(0, I) noise every plan, as the flow policy does on its own; "episode"
        # draws one per episode and keeps it, so the plans stay in one mode (the demos
        # pass the same poses going up and coming back down).
        self.random_noise: str | bool = False
        self._episode_noise: np.ndarray | None = None
        self.smooth_ticks = 5
        self._rng = np.random.default_rng()
        self.obs_dim = flow.config.feature_dim + len(JOINTS) + 1
        self.action_dim = noise_steps * len(JOINTS)

    def decide(self, robot_obs: dict[str, Any], state: np.ndarray, elapsed_s: float = 0.0) -> Decision:
        images = {name: robot_obs[name] for name in self.flow.config.cameras}
        feature = self.flow.feature(images, state, elapsed_s)
        obs = np.concatenate((feature[0].float().cpu().numpy(), state / STATE_SCALE,
                              [elapsed_s / TIME_SCALE])).astype(np.float32)
        return Decision(obs, state, feature)

    def noise(self, action: np.ndarray) -> np.ndarray:
        rows = np.asarray(action, dtype=np.float32).reshape(self.noise_steps, len(JOINTS)) * self.bound
        tail = np.repeat(rows[-1:], self.flow.config.horizon - self.noise_steps, axis=0)
        return np.concatenate((rows, tail))

    def begin_episode(self) -> None:
        self._episode_noise = None

    def goals(self, decision: Decision, action: np.ndarray) -> np.ndarray:
        shape = (self.flow.config.horizon, len(JOINTS))
        if self.random_noise == "episode":
            if self._episode_noise is None:
                self._episode_noise = self._rng.standard_normal(shape).astype(np.float32)
            noise = self._episode_noise
        elif self.random_noise:
            noise = self._rng.standard_normal(shape).astype(np.float32)
        else:
            noise = self.noise(action)
        plan = smooth(self.flow.chunk(decision.context, decision.state, noise), self.smooth_ticks)
        self.tail = plan[self.k:]  # what this plan does next, for blending into the next one
        return plan[: self.k]


class DiffusionBase:
    """A trained LeRobot Diffusion Policy checkpoint, frozen, with its observation
    history (``n_obs_steps`` frames, one control tick apart) kept here.

    ``push`` adds one tick's observation; ``plan`` runs the denoiser from a given
    noise and returns the executed part of the chunk (``n_action_steps`` absolute
    joint goals), what follows it, and the encoder's conditioning vector.
    """

    def __init__(self, path: str | Path, device: str | None = None) -> None:
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
        from lerobot.policies.factory import make_pre_post_processors

        from corl_rl_demo.agent.flow import default_device

        self.device = torch.device(device) if device else default_device()
        self.policy = DiffusionPolicy.from_pretrained(str(path)).to(self.device).eval()
        self.policy.config.device = str(self.device)
        self.pre, self.post = make_pre_post_processors(
            self.policy.config, pretrained_path=str(path),
            preprocessor_overrides={"device_processor": {"device": str(self.device)}},
        )
        config = self.policy.config
        self.image_keys = list(config.image_features)
        self.cameras = [key.removeprefix("observation.images.") for key in self.image_keys]
        self.horizon, self.n_obs_steps, self.n_action_steps = config.horizon, config.n_obs_steps, config.n_action_steps
        self.action_dim = config.action_feature.shape[0]
        self.policy.reset()

    def reset(self) -> None:
        self.policy.reset()

    @torch.no_grad()
    def push(self, robot_obs: dict[str, Any], state: np.ndarray, task: str) -> None:
        from lerobot.policies.utils import populate_queues, prepare_observation_for_inference
        from lerobot.utils.constants import OBS_IMAGES

        observation = {"observation.state": state.astype(np.float32)}
        for camera in self.cameras:
            observation[f"observation.images.{camera}"] = robot_obs[camera]
        batch = self.pre(prepare_observation_for_inference(observation, self.device, task, None))
        batch = dict(batch)
        batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.image_keys], dim=-4)
        self.policy._queues = populate_queues(self.policy._queues, batch)

    def _history(self) -> dict[str, torch.Tensor]:
        from lerobot.utils.constants import OBS_IMAGES, OBS_STATE

        return {key: torch.stack(list(self.policy._queues[key]), dim=1) for key in (OBS_STATE, OBS_IMAGES)}

    @torch.no_grad()
    def condition(self) -> torch.Tensor:
        """The encoder's conditioning vector for the current history, ``(1, D)``."""
        return self.policy.diffusion._prepare_global_conditioning(self._history())

    @torch.no_grad()
    def plan(self, condition: torch.Tensor, noise: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        sample = self.policy.diffusion.conditional_sample(
            1, global_cond=condition,
            noise=torch.as_tensor(noise, dtype=torch.float32, device=self.device)[None],
        )
        start = self.n_obs_steps - 1
        actions = self.post(sample[:, start:])[0].float().cpu().numpy()
        return actions[: self.n_action_steps], actions[self.n_action_steps:]


class DSRLDiffusion:
    """The Diffusion Policy acts from a noise SAC chooses (``noise_steps`` rows in
    ``[-bound, bound]``, the last repeated over the horizon), or, run alone, from
    random noise as it does natively. It plans every ``n_action_steps`` ticks and
    sees every tick's observation in between (``observe_tick``)."""

    name = "dsrl-diffusion"

    def __init__(self, base: DiffusionBase, *, noise_steps: int = 4, bound: float = 1.5,
                 task: str = "Press the green button") -> None:
        self.base, self.noise_steps, self.bound, self.task = base, noise_steps, bound, task
        self.k = base.n_action_steps
        self.action_dim = noise_steps * base.action_dim
        self.smooth_ticks = 5
        self.random_noise: str | bool = False
        self._episode_noise: np.ndarray | None = None
        self._rng = np.random.default_rng()
        self.tail = np.zeros((0, base.action_dim))
        base.push(_blank_observation(base), np.zeros(base.action_dim), task)
        self.obs_dim = int(base.condition().numel()) + base.action_dim + 1
        base.reset()

    def begin_episode(self) -> None:
        self.base.reset()
        self._episode_noise = None

    def observe_tick(self, robot_obs: dict[str, Any], state: np.ndarray) -> None:
        self.base.push(robot_obs, state, self.task)

    def decide(self, robot_obs: dict[str, Any], state: np.ndarray, elapsed_s: float = 0.0) -> Decision:
        self.base.push(robot_obs, state, self.task)
        condition = self.base.condition()
        obs = np.concatenate((condition[0].float().cpu().numpy(), state / STATE_SCALE,
                              [elapsed_s / TIME_SCALE])).astype(np.float32)
        return Decision(obs, state, condition)

    def noise(self, action: np.ndarray) -> np.ndarray:
        shape = (self.base.horizon, self.base.action_dim)
        if self.random_noise == "episode":
            if self._episode_noise is None:
                self._episode_noise = self._rng.standard_normal(shape).astype(np.float32)
            return self._episode_noise
        if self.random_noise:
            return self._rng.standard_normal(shape).astype(np.float32)
        rows = np.asarray(action, dtype=np.float32).reshape(self.noise_steps, self.base.action_dim) * self.bound
        return np.concatenate((rows, np.repeat(rows[-1:], shape[0] - self.noise_steps, axis=0)))

    def goals(self, decision: Decision, action: np.ndarray) -> np.ndarray:
        executed, tail = self.base.plan(decision.context, self.noise(action))
        plan = smooth(np.concatenate((executed, tail)), self.smooth_ticks)
        self.tail = plan[len(executed):]
        return plan[: len(executed)]


def _blank_observation(base: DiffusionBase) -> dict[str, Any]:
    """A black frame per camera, for sizing SAC's observation before the arm is read."""
    shape = base.policy.config.image_features[base.image_keys[0]].shape  # (C, H, W)
    return {camera: np.zeros((shape[1], shape[2], 3), np.uint8) for camera in base.cameras}
