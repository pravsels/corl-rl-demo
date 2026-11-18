"""A small flow-matching policy: the base that DSRL steers.

The cameras, the joints and the time since the episode began go through a
shared ResNet-18 and a projection into a 512-D feature (what SAC sees). The
time is there because the demos wait at rest for about 3 s before moving: the
images and joints are the same all through that wait, so only the clock tells
the policy when to go. A flow head turns a ``(horizon, joints)`` noise into an
action chunk in ``flow_steps`` Euler steps. Actions are trained relative to the
current joints and normalized per chunk step. Training reads a dataset recorded
with ``lerobot-5motor record``:

    uv run rl-agent flow-train --dataset <repo id> --output outputs/flow
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class FlowConfig:
    cameras: tuple[str, ...] = ("top",)
    action_dim: int = 5
    image_size: int = 128
    horizon: int = 30
    feature_dim: int = 512
    hidden_dim: int = 512
    flow_steps: int = 10
    pretrained_encoder: bool = True
    # Seconds since the episode began enter the feature as t / time_scale_s,
    # capped at 2; 0 leaves time out (checkpoints from before it existed).
    time_scale_s: float = 10.0

    @classmethod
    def from_json(cls, text: str) -> FlowConfig:
        data = json.loads(text)
        data["cameras"] = tuple(data["cameras"])
        data.setdefault("time_scale_s", 0.0)
        return cls(**data)

    @property
    def time_input(self) -> bool:
        return self.time_scale_s > 0


def letterbox(image: np.ndarray, size: int) -> np.ndarray:
    """Resize an HWC uint8 image to ``size`` x ``size`` keeping its aspect ratio,
    on black (OpenPI's resize_with_pad)."""
    from PIL import Image

    if image.shape[:2] == (size, size):
        return image
    pil = Image.fromarray(image)
    width, height = pil.size
    ratio = max(width / size, height / size)
    new_w, new_h = int(width / ratio), int(height / ratio)
    canvas = Image.new(pil.mode, (size, size), 0)
    canvas.paste(pil.resize((new_w, new_h), Image.BILINEAR), ((size - new_w) // 2, (size - new_h) // 2))
    return np.asarray(canvas)


def _time_embedding(t: torch.Tensor, dim: int = 64) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(1000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / half)
    angles = t[:, None] * freqs[None] * 1000.0
    return torch.cat((angles.sin(), angles.cos()), dim=-1)


class _AdaLNBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width, elementwise_affine=False)
        self.modulation = nn.Linear(width, 3 * width)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)
        self.mlp = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, width))

    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, shift, gate = self.modulation(cond).chunk(3, dim=-1)
        return h + (1 + gate) * self.mlp(self.norm(h) * (1 + scale) + shift)


class FlowNet(nn.Module):
    def __init__(self, config: FlowConfig) -> None:
        super().__init__()
        import torchvision

        self.config = config
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if config.pretrained_encoder else None
        trunk = torchvision.models.resnet18(weights=weights)
        trunk.fc = nn.Identity()
        self.encoder = trunk
        self.camera_embedding = nn.Parameter(torch.zeros(len(config.cameras), 512))
        self.feature = nn.Sequential(
            nn.Linear(512 * len(config.cameras) + config.action_dim + int(config.time_input),
                      config.feature_dim),
            nn.LayerNorm(config.feature_dim),
        )
        chunk = config.horizon * config.action_dim
        width = config.hidden_dim
        self.cond = nn.Sequential(nn.Linear(config.feature_dim + 64, width), nn.SiLU())
        self.x_in = nn.Linear(chunk, width)
        self.blocks = nn.ModuleList(_AdaLNBlock(width) for _ in range(3))
        self.x_out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, chunk))
        nn.init.zeros_(self.x_out[1].weight)
        nn.init.zeros_(self.x_out[1].bias)
        self.register_buffer("image_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def embed(self, images: list[torch.Tensor], state: torch.Tensor,
              elapsed_s: torch.Tensor | None = None) -> torch.Tensor:
        """One ``(B, 3, H, W)`` image in [0, 1] per camera, the normalized joints
        and the ``(B,)`` seconds since the episode began."""
        features = [
            self.encoder((image - self.image_mean) / self.image_std) + self.camera_embedding[i]
            for i, image in enumerate(images)
        ]
        inputs = [*features, state]
        if self.config.time_input:
            if elapsed_s is None:
                raise ValueError("this policy needs the time since the episode began")
            inputs.append((elapsed_s / self.config.time_scale_s).clamp(max=2.0)[:, None].to(state))
        return self.feature(torch.cat(inputs, dim=-1))

    def velocity(self, feature: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        cond = self.cond(torch.cat((feature, _time_embedding(t)), dim=-1))
        h = self.x_in(x_t.flatten(1))
        for block in self.blocks:
            h = block(h, cond)
        return self.x_out(h).view_as(x_t)

    def loss(self, feature: torch.Tensor, actions: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        """π0's flow matching: ``x_t = t·noise + (1-t)·actions``, target ``noise - actions``."""
        noise = torch.randn_like(actions)
        t = torch.distributions.Beta(1.5, 1.0).sample((actions.shape[0],)).to(actions) * 0.999 + 0.001
        x_t = t[:, None, None] * noise + (1 - t[:, None, None]) * actions
        error = (self.velocity(feature, x_t, t) - (noise - actions)) ** 2
        keep = (~pad).to(error.dtype)[..., None]
        return (error * keep).sum() / (keep.sum() * actions.shape[-1]).clamp_min(1.0)

    @torch.no_grad()
    def sample(self, feature: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        x, dt = noise, -1.0 / self.config.flow_steps
        for step in range(self.config.flow_steps):
            t = torch.full((x.shape[0],), 1.0 + step * dt, device=x.device, dtype=x.dtype)
            x = x + dt * self.velocity(feature, x, t)
        return x


def load_weights(net: FlowNet, state: dict[str, torch.Tensor]) -> None:
    """Load ``state``, also from a checkpoint without the time input: its column
    starts at zero, so the policy first acts exactly as before."""
    key = "feature.0.weight"
    old, new = state[key], net.state_dict()[key]
    if old.shape != new.shape:
        if old.shape[1] + 1 != new.shape[1]:
            raise ValueError(f"cannot load {key} of shape {tuple(old.shape)} into {tuple(new.shape)}")
        state = dict(state)
        state[key] = torch.cat((old, torch.zeros(old.shape[0], 1, dtype=old.dtype)), dim=1)
    net.load_state_dict(state)


def chunk_targets(states: np.ndarray, actions: np.ndarray, episodes: np.ndarray, horizon: int):
    """Each frame's next ``horizon`` actions relative to its joints (the last one
    repeated past the episode's end), and which steps are that padding."""
    n = len(states)
    chunks = np.empty((n, horizon, actions.shape[1]), np.float32)
    pad = np.zeros((n, horizon), bool)
    for episode in np.unique(episodes):
        index = np.flatnonzero(episodes == episode)
        start, end = index[0], index[-1] + 1
        for i in range(start, end):
            take = np.arange(i, i + horizon)
            pad[i] = take >= end
            chunks[i] = actions[np.minimum(take, end - 1)] - states[i]
    return chunks, pad


def active_frames(states: np.ndarray, episodes: np.ndarray, *, threshold_deg: float = 3.0,
                  lead: int = 6) -> np.ndarray:
    """Frames to train on: each episode from ``lead`` frames before any joint first
    moves ``threshold_deg`` from where it started. The demos wait about 3 s before
    moving; a policy trained on that wait sees the same image at rest and learns to
    stay there."""
    keep = []
    for episode in np.unique(episodes):
        index = np.flatnonzero(episodes == episode)
        moved = np.flatnonzero(np.abs(states[index] - states[index[0]]).max(1) > threshold_deg)
        start = max(0, (moved[0] if len(moved) else 0) - lead)
        keep.append(index[start:])
    return np.concatenate(keep)


def norm_stats(states: np.ndarray, chunks: np.ndarray, pad: np.ndarray) -> dict[str, list]:
    valid = chunks[~pad[:, 0]]
    floor = 1e-2  # a joint the demos never move
    return {
        "state_mean": states.mean(0).tolist(),
        "state_std": np.maximum(states.std(0), floor).tolist(),
        "action_mean": valid.mean(0).tolist(),
        "action_std": np.maximum(valid.std(0), floor).tolist(),
    }


def default_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FlowPolicy:
    """A trained ``FlowNet`` with its normalization, on one device."""

    def __init__(self, net: FlowNet, stats: dict[str, list], device: torch.device) -> None:
        self.net = net.to(device).eval()
        self.config = net.config
        self.device = device
        t = lambda key: torch.tensor(stats[key], dtype=torch.float32, device=device)  # noqa: E731
        self.state_mean, self.state_std = t("state_mean"), t("state_std")
        self.action_mean, self.action_std = t("action_mean"), t("action_std")

    @classmethod
    def load(cls, directory: str | Path, device: str | None = None) -> FlowPolicy:
        directory = Path(directory)
        config = FlowConfig.from_json((directory / "flow_config.json").read_text())
        config.pretrained_encoder = False
        net = FlowNet(config)
        load_weights(net, torch.load(directory / "model.pt", map_location="cpu"))
        stats = json.loads((directory / "norm_stats.json").read_text())
        return cls(net, stats, torch.device(device) if device else default_device())

    def save(self, directory: str | Path) -> None:
        save(Path(directory), self.net, {
            "state_mean": self.state_mean.tolist(), "state_std": self.state_std.tolist(),
            "action_mean": self.action_mean.tolist(), "action_std": self.action_std.tolist(),
        })

    @torch.no_grad()
    def feature(self, images: dict[str, np.ndarray], state: np.ndarray, elapsed_s: float = 0.0) -> torch.Tensor:
        size = self.config.image_size
        tensors = [
            torch.from_numpy(np.array(letterbox(images[name], size))).to(self.device)
            .permute(2, 0, 1).float().div(255.0)[None]
            for name in self.config.cameras
        ]
        joints = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        elapsed = torch.tensor([elapsed_s], dtype=torch.float32, device=self.device)
        return self.net.embed(tensors, ((joints - self.state_mean) / self.state_std)[None], elapsed)

    @torch.no_grad()
    def chunk(self, feature: torch.Tensor, state: np.ndarray, noise: np.ndarray) -> np.ndarray:
        """Absolute joint goals ``(horizon, joints)`` from a ``(horizon, joints)`` noise."""
        x = self.net.sample(feature, torch.as_tensor(noise, dtype=torch.float32, device=self.device)[None])[0]
        relative = x * self.action_std + self.action_mean
        return (relative + torch.as_tensor(state, dtype=torch.float32, device=self.device)).cpu().numpy()


def save(directory: Path, net: FlowNet, stats: dict[str, list]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), directory / "model.pt")
    (directory / "flow_config.json").write_text(json.dumps(asdict(net.config), indent=2))
    (directory / "norm_stats.json").write_text(json.dumps(stats))


# -- training --------------------------------------------------------------------


@dataclass
class TrainConfig:
    dataset: str
    output: str
    root: str | None = None
    steps: int = 20_000
    batch_size: int = 32
    lr: float = 3e-4
    encoder_lr: float = 1e-4
    num_workers: int = 4
    save_every: int = 5_000
    log_every: int = 100
    trim_idle_deg: float = 0.0  # drop each demo's wait before a joint moves this far; 0 keeps it
    init: str | None = None  # a flow-train output to continue from
    # Random brightness, contrast and saturation of up to this fraction per image.
    # Without it the policy keys on a 1-2 % darkening that precedes every demo's
    # start, which the rig does not reproduce, and never leaves the start pose.
    color_jitter: float = 0.2
    keep_checkpoints: bool = False  # also keep each save_every checkpoint in <output>/step_<N>
    flow: FlowConfig = field(default_factory=FlowConfig)


class _Frames(torch.utils.data.Dataset):
    def __init__(self, dataset: Any, chunks, pad, states, stats, config: FlowConfig,
                 frames: np.ndarray | None = None, times: np.ndarray | None = None) -> None:
        self.dataset, self.chunks, self.pad, self.states, self.config = dataset, chunks, pad, states, config
        self.times = np.zeros(len(chunks), np.float32) if times is None else times
        self.frames = np.arange(len(chunks)) if frames is None else frames
        self.state_mean, self.state_std = np.asarray(stats["state_mean"]), np.asarray(stats["state_std"])
        self.action_mean, self.action_std = np.asarray(stats["action_mean"]), np.asarray(stats["action_std"])

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        i = int(self.frames[item])
        frame = self.dataset[i]
        images = []
        for name in self.config.cameras:
            chw = frame[f"observation.images.{name}"]
            hwc = (chw.permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype(np.uint8)
            images.append(torch.from_numpy(np.array(letterbox(hwc, self.config.image_size))).permute(2, 0, 1))
        return {
            "images": torch.stack(images),
            "state": torch.as_tensor((self.states[i] - self.state_mean) / self.state_std, dtype=torch.float32),
            "actions": torch.as_tensor((self.chunks[i] - self.action_mean) / self.action_std, dtype=torch.float32),
            "pad": torch.as_tensor(self.pad[i]),
            "time": torch.as_tensor(self.times[i], dtype=torch.float32),
        }


def _color_jitter(images: torch.Tensor, amount: float) -> torch.Tensor:
    """Per-image random brightness, contrast and saturation in ``[1 - amount, 1 + amount]``."""
    if amount <= 0:
        return images
    n = images.shape[0]
    factor = lambda: 1 + (torch.rand(n, 1, 1, 1, device=images.device) * 2 - 1) * amount  # noqa: E731
    images = images * factor()
    mean = images.mean(dim=(1, 2, 3), keepdim=True)
    images = (images - mean) * factor() + mean
    gray = images.mean(dim=1, keepdim=True)
    images = (images - gray) * factor() + gray
    return images.clamp(0, 1)


def _random_shift(images: torch.Tensor, pad: int = 6) -> torch.Tensor:
    n, c, h, w = images.shape
    padded = F.pad(images, (pad, pad, pad, pad), mode="replicate")
    dx = torch.randint(0, 2 * pad + 1, (n,), device=images.device)
    dy = torch.randint(0, 2 * pad + 1, (n,), device=images.device)
    rows = torch.arange(h, device=images.device)[None] + dy[:, None]
    cols = torch.arange(w, device=images.device)[None] + dx[:, None]
    return padded[torch.arange(n)[:, None, None, None], torch.arange(c)[None, :, None, None],
                  rows[:, None, :, None], cols[:, None, None, :]]


def train(cfg: TrainConfig) -> Path:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    device = default_device()
    dataset = LeRobotDataset(cfg.dataset, root=cfg.root, video_backend="pyav")
    table = dataset.hf_dataset.with_format("numpy")
    states = np.stack(table["observation.state"]).astype(np.float32)
    actions = np.stack(table["action"]).astype(np.float32)
    episodes = np.asarray(table["episode_index"])
    times = (np.asarray(table["frame_index"]) / dataset.fps).astype(np.float32)
    missing = [c for c in cfg.flow.cameras if f"observation.images.{c}" not in dataset.meta.features]
    if missing:
        raise SystemExit(f"the dataset has no cameras {missing}")
    cfg.flow.action_dim = actions.shape[1]
    chunks, pad = chunk_targets(states, actions, episodes, cfg.flow.horizon)
    frames = (active_frames(states, episodes, threshold_deg=cfg.trim_idle_deg)
              if cfg.trim_idle_deg > 0 else np.arange(len(states)))
    if cfg.init:
        stats = json.loads((Path(cfg.init) / "norm_stats.json").read_text())
    else:
        stats = norm_stats(states[frames], chunks[frames], pad[frames])
    loader = torch.utils.data.DataLoader(
        _Frames(dataset, chunks, pad, states, stats, cfg.flow, frames, times), batch_size=cfg.batch_size,
        shuffle=True, drop_last=True, num_workers=cfg.num_workers, persistent_workers=cfg.num_workers > 0,
    )
    net = FlowNet(cfg.flow)
    if cfg.init:
        load_weights(net, torch.load(Path(cfg.init) / "model.pt", map_location="cpu"))
    net = net.to(device).train()
    encoder = list(net.encoder.parameters())
    ids = {id(p) for p in encoder}
    optimizer = torch.optim.AdamW(
        [{"params": encoder, "lr": cfg.encoder_lr},
         {"params": [p for p in net.parameters() if id(p) not in ids], "lr": cfg.lr}],
        weight_decay=1e-4,
    )
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / 500) * 0.5 * (1 + math.cos(math.pi * s / cfg.steps))
    )
    output = Path(cfg.output)
    print(f"[flow] {len(np.unique(episodes))} episodes, {len(frames)} of {len(states)} frames "
          f"{'(idle starts trimmed) ' if cfg.trim_idle_deg > 0 else ''}"
          f"{'with' if cfg.flow.time_input else 'without'} time input, "
          f"{actions.shape[1]} joints, on {device}", flush=True)
    step, running, started = 0, 0.0, time.monotonic()
    while step < cfg.steps:
        for batch in loader:
            images = batch["images"].to(device).float().div(255.0)
            b, cams = images.shape[:2]
            shape = images.shape
            images = _color_jitter(_random_shift(images.flatten(0, 1)), cfg.color_jitter).view(shape)
            feature = net.embed([images[:, i] for i in range(cams)], batch["state"].to(device),
                                batch["time"].to(device))
            loss = net.loss(feature, batch["actions"].to(device), batch["pad"].to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            schedule.step()
            step += 1
            running += float(loss.detach())
            if step % cfg.log_every == 0:
                print(f"[flow] step {step}/{cfg.steps} loss {running / cfg.log_every:.4f} "
                      f"({step / (time.monotonic() - started):.1f} steps/s)", flush=True)
                running = 0.0
            if step % cfg.save_every == 0 or step == cfg.steps:
                save(output, net, stats)
                if cfg.keep_checkpoints:
                    save(output / f"step_{step}", net, stats)
            if step >= cfg.steps:
                break
    return output
