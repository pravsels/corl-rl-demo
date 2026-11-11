"""LeRobot configs for the five-motor arms. The arms themselves are in ``five_motor``.

LeRobot finds an arm's class from its config: it drops ``Config`` from the class
name and looks in the module named like this one without ``config_``. So the
class names here and this file's name must stay in step with ``five_motor.py``.
"""

from dataclasses import dataclass

from lerobot.robots import RobotConfig
from lerobot.robots.so_follower import SOFollowerRobotConfig
from lerobot.teleoperators import TeleoperatorConfig
from lerobot.teleoperators.so_leader import SOLeaderTeleopConfig


@RobotConfig.register_subclass("so101_5motor_follower")
@dataclass
class SO101FiveMotorFollowerConfig(SOFollowerRobotConfig):
    pass


@TeleoperatorConfig.register_subclass("so101_5motor_leader")
@dataclass
class SO101FiveMotorLeaderConfig(SOLeaderTeleopConfig):
    pass
