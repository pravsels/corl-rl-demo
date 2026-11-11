"""SO-101 arms driven by their five joint motors, with no gripper motor.

Pressing a button needs no gripper, so the gripper is fixed shut and its motor is
left out everywhere: it is not on the bus, not calibrated, not recorded and not
commanded. Datasets and policies made with these arms have five-number states
and actions.

LeRobot's own commands only know the six-motor arms. ``lerobot-5motor`` runs
them with these two types added:

    uv run lerobot-5motor calibrate --robot.type=so101_5motor_follower ...
"""

from __future__ import annotations

import importlib
import sys

from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.robots.so_follower import SOFollower
from lerobot.teleoperators.so_leader import SOLeader

from corl_rl_demo.config_five_motor import (
    SO101FiveMotorFollowerConfig,
    SO101FiveMotorLeaderConfig,
)

# What the joints of this arm that were never a gripper report. LeRobot gives a
# gripper servo lower limits (half torque, a quarter when overloaded) and they
# stay in the servo, so one reused as a joint would be weak without this.
JOINT_LIMITS = {
    "Max_Torque_Limit": 1000,
    "Torque_Limit": 1000,  # the running copy of the line above, read at power-on
    "Protection_Current": 310,
    "Overload_Torque": 80,
}

COMMANDS = ("calibrate", "teleoperate", "record", "replay")


def _without_gripper(bus: FeetechMotorsBus) -> FeetechMotorsBus:
    motors = {name: motor for name, motor in bus.motors.items() if name != "gripper"}
    return FeetechMotorsBus(port=bus.port, motors=motors, calibration=bus.calibration)


class SO101FiveMotorFollower(SOFollower):
    config_class = SO101FiveMotorFollowerConfig
    # Also the calibration folder and the dataset's robot type, so neither is
    # mistaken for a six-motor arm's.
    name = "so101_5motor_follower"

    def __init__(self, config: SO101FiveMotorFollowerConfig):
        super().__init__(config)
        self.bus = _without_gripper(self.bus)

    def configure(self) -> None:
        super().configure()
        with self.bus.torque_disabled():
            for motor in self.bus.motors:
                for register, value in JOINT_LIMITS.items():
                    self.bus.write(register, motor, value)


class SO101FiveMotorLeader(SOLeader):
    """Reads five joints. A gripper motor on the leader, if it has one, is ignored."""

    config_class = SO101FiveMotorLeaderConfig
    name = "so101_5motor_leader"

    def __init__(self, config: SO101FiveMotorLeaderConfig):
        super().__init__(config)
        self.bus = _without_gripper(self.bus)


def main(argv: list[str] | None = None) -> None:
    """Run a LeRobot command with the five-motor arm types added."""
    args = sys.argv[1:] if argv is None else list(argv)
    if not args or args[0] not in COMMANDS:
        sys.exit(f"usage: lerobot-5motor {{{','.join(COMMANDS)}}} [the lerobot command's options]")
    command, *options = args
    # LeRobot's commands read their options from sys.argv.
    sys.argv = [f"lerobot-{command}", *options]
    importlib.import_module(f"lerobot.scripts.lerobot_{command}").main()


if __name__ == "__main__":
    main()
