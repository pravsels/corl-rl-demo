"""LeRobot configs for the five-motor arms. The arms themselves are in ``five_motor``.

LeRobot finds an arm's class from its config: it drops ``Config`` from the class
name and looks in the module named like this one without ``config_``. So the
class names here and this file's name must stay in step with ``five_motor.py``.
"""

from dataclasses import dataclass, field
from pathlib import Path

from lerobot.robots import RobotConfig
from lerobot.robots.so_follower import SOFollowerRobotConfig
from lerobot.teleoperators import TeleoperatorConfig
from lerobot.teleoperators.so_leader import SOLeaderTeleopConfig

# Calibration is kept in the repo, not in LeRobot's per-machine cache, so that
# another machine drives the same arms without calibrating them again. This is
# the repo's folder because ``uv sync`` installs the package in place.
CALIBRATION = Path(__file__).resolve().parents[2] / "calibration"


@RobotConfig.register_subclass("so101_5motor_follower")
@dataclass
class SO101FiveMotorFollowerConfig(SOFollowerRobotConfig):
    calibration_dir: Path | None = field(default=CALIBRATION / "follower", kw_only=True)


@TeleoperatorConfig.register_subclass("so101_5motor_leader")
@dataclass
class SO101FiveMotorLeaderConfig(SOLeaderTeleopConfig):
    calibration_dir: Path | None = field(default=CALIBRATION / "leader", kw_only=True)
