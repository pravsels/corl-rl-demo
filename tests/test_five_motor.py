import sys
from contextlib import contextmanager

import pytest
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.datasets.feature_utils import combine_feature_dicts
from lerobot.datasets.pipeline_features import (
    aggregate_pipeline_dataset_features,
    create_initial_features,
)
from lerobot.processor import make_default_processors
from lerobot.robots import RobotConfig, make_robot_from_config
from lerobot.teleoperators import TeleoperatorConfig, make_teleoperator_from_config

from corl_rl_demo.config_five_motor import (
    SO101FiveMotorFollowerConfig,
    SO101FiveMotorLeaderConfig,
)
from corl_rl_demo.five_motor import SO101FiveMotorFollower, SO101FiveMotorLeader, main
from corl_rl_demo.rig import CAMERA_FPS, CAMERA_HEIGHT, CAMERA_WIDTH, JOINTS

PORT = "/dev/does-not-exist"
JOINT_KEYS = [f"{joint}.pos" for joint in JOINTS]


def follower(tmp_path, **kwargs):
    config = SO101FiveMotorFollowerConfig(port=PORT, id="test", calibration_dir=tmp_path, **kwargs)
    return SO101FiveMotorFollower(config)


def leader(tmp_path):
    config = SO101FiveMotorLeaderConfig(port=PORT, id="test", calibration_dir=tmp_path)
    return SO101FiveMotorLeader(config)


def test_follower_has_the_five_arm_joints_and_no_gripper(tmp_path):
    robot = follower(tmp_path)
    assert "gripper" not in JOINTS
    assert {name: motor.id for name, motor in robot.bus.motors.items()} == {
        "shoulder_pan": 1,
        "shoulder_lift": 2,
        "elbow_flex": 3,
        "wrist_flex": 4,
        "wrist_roll": 5,
    }
    assert tuple(robot.bus.motors) == JOINTS
    assert list(robot.action_features) == JOINT_KEYS


def test_leader_reports_the_same_five_joints(tmp_path):
    assert list(leader(tmp_path).action_features) == JOINT_KEYS


def test_a_recorded_dataset_has_five_joint_state_and_action(tmp_path):
    top = OpenCVCameraConfig(
        index_or_path=0, width=CAMERA_WIDTH, height=CAMERA_HEIGHT, fps=CAMERA_FPS
    )
    robot = follower(tmp_path, cameras={"top": top})

    # The same construction lerobot-record uses.
    teleop_action_processor, _, robot_observation_processor = make_default_processors()
    features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=True,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=True,
        ),
    )

    assert features["action"]["names"] == JOINT_KEYS
    assert features["action"]["shape"] == (5,)
    assert features["observation.state"]["names"] == JOINT_KEYS
    assert features["observation.state"]["shape"] == (5,)
    assert features["observation.images.top"]["shape"] == (CAMERA_HEIGHT, CAMERA_WIDTH, 3)


def test_lerobot_builds_both_arms_from_their_type_names(tmp_path):
    robot_config = RobotConfig.get_choice_class("so101_5motor_follower")(
        port=PORT, id="test", calibration_dir=tmp_path
    )
    teleop_config = TeleoperatorConfig.get_choice_class("so101_5motor_leader")(
        port=PORT, id="test", calibration_dir=tmp_path
    )
    assert type(make_robot_from_config(robot_config)) is SO101FiveMotorFollower
    assert type(make_teleoperator_from_config(teleop_config)) is SO101FiveMotorLeader


def test_calibration_is_kept_apart_from_a_six_motor_arm():
    assert SO101FiveMotorFollower.name == "so101_5motor_follower"
    assert SO101FiveMotorLeader.name == "so101_5motor_leader"


class RecordingBus:
    """Stands in for the motor bus: remembers each write and whether torque was off."""

    def __init__(self, motors):
        self.motors = motors
        self.torque_off = False
        self.writes = []

    @contextmanager
    def torque_disabled(self):
        self.torque_off = True
        try:
            yield
        finally:
            self.torque_off = False

    def configure_motors(self):
        pass

    def write(self, register, motor, value):
        self.writes.append((register, motor, value, self.torque_off))


def test_configure_gives_every_joint_full_joint_limits(tmp_path):
    # A gripper servo moved into a joint keeps the gripper's reduced limits.
    robot = follower(tmp_path)
    robot.bus = RecordingBus(robot.bus.motors)

    robot.configure()

    for register, wanted in {
        "Max_Torque_Limit": 1000,
        "Torque_Limit": 1000,
        "Protection_Current": 310,
        "Overload_Torque": 80,
    }.items():
        written = {m: (v, off) for r, m, v, off in robot.bus.writes if r == register}
        # Torque must be off, or the servo will not store the value.
        assert written == {joint: (wanted, True) for joint in JOINTS}, register


def test_launcher_rejects_a_command_it_does_not_wrap(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["lerobot-5motor"])
    with pytest.raises(SystemExit, match="calibrate.*teleoperate.*record.*replay"):
        main(["setup-motors"])


def test_launcher_runs_a_lerobot_command_that_knows_the_five_motor_arm(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["lerobot-5motor"])
    # LeRobot gets as far as opening the port, so it parsed the type and built the arm.
    with pytest.raises(ConnectionError, match=PORT):
        main(
            [
                "calibrate",
                "--robot.type=so101_5motor_follower",
                f"--robot.port={PORT}",
                "--robot.id=test",
                f"--robot.calibration_dir={tmp_path}",
            ]
        )
