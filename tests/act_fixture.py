"""A small untrained ACT checkpoint with this rig's features, saved like a trained one."""

from pathlib import Path

import torch

from corl_rl_demo.rig import CAMERA_HEIGHT, CAMERA_WIDTH, JOINTS


def make_act_checkpoint(directory: Path, chunk_size: int = 30) -> Path:
    from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    n = len(JOINTS)
    config = ACTConfig(
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(n,)),
            "observation.images.top": PolicyFeature(type=FeatureType.VISUAL, shape=(3, CAMERA_HEIGHT, CAMERA_WIDTH)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(n,))},
        chunk_size=chunk_size, n_action_steps=chunk_size, pretrained_backbone_weights=None,
        dim_feedforward=256, n_encoder_layers=1, n_decoder_layers=1, device="cpu",
        normalization_mapping={"VISUAL": NormalizationMode.MEAN_STD, "STATE": NormalizationMode.MEAN_STD,
                               "ACTION": NormalizationMode.MEAN_STD},
    )
    stats = {
        "observation.state": {"mean": torch.zeros(n), "std": torch.full((n,), 30.0)},
        "action": {"mean": torch.full((n,), 10.0), "std": torch.full((n,), 30.0)},
        "observation.images.top": {"mean": torch.full((3, 1, 1), 0.5), "std": torch.full((3, 1, 1), 0.25)},
    }
    policy = ACTPolicy(config)
    pre, post = make_pre_post_processors(config, dataset_stats=stats)
    policy.save_pretrained(directory)
    pre.save_pretrained(directory)
    post.save_pretrained(directory)
    return directory


def make_diffusion_checkpoint(directory: Path) -> Path:
    """A small untrained Diffusion Policy with this rig's features (top camera)."""
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.policies.factory import make_pre_post_processors

    n = len(JOINTS)
    config = DiffusionConfig(
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(n,)),
            "observation.images.top": PolicyFeature(type=FeatureType.VISUAL, shape=(3, CAMERA_HEIGHT, CAMERA_WIDTH)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(n,))},
        resize_shape=(60, 80), crop_ratio=0.9, down_dims=(32, 64), noise_scheduler_type="DDIM",
        num_inference_steps=2, device="cpu",
    )
    stats = {
        "observation.state": {"min": torch.full((n,), -100.0), "max": torch.full((n,), 100.0),
                              "mean": torch.zeros(n), "std": torch.full((n,), 30.0)},
        "action": {"min": torch.full((n,), -100.0), "max": torch.full((n,), 100.0),
                   "mean": torch.zeros(n), "std": torch.full((n,), 30.0)},
        "observation.images.top": {"mean": torch.full((3, 1, 1), 0.5), "std": torch.full((3, 1, 1), 0.25),
                                   "min": torch.zeros(3, 1, 1), "max": torch.ones(3, 1, 1)},
    }
    policy = DiffusionPolicy(config)
    pre, post = make_pre_post_processors(config, dataset_stats=stats)
    policy.save_pretrained(directory)
    pre.save_pretrained(directory)
    post.save_pretrained(directory)
    return directory
