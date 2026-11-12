import json
import sys
import threading
from pathlib import Path

import numpy as np
import pytest
import torch

from corl_rl_demo.agent.episode import RewardConfig, score
from corl_rl_demo.agent.methods import DSRLFlow, ResidualACT
from corl_rl_demo.agent.run import Keys, LoopConfig, run_episode
from corl_rl_demo.agent.safety import Limits, SafeArm, SafetyStop
from corl_rl_demo.agent.sac import SAC, ReplayBuffer, SACConfig
from corl_rl_demo.buttons import PressLatch
from corl_rl_demo.rig import JOINTS

sys.path.insert(0, str(Path(__file__).parent))
from act_fixture import make_act_checkpoint  # noqa: E402

N = len(JOINTS)


class Bus:
    def __init__(self) -> None:
        self.position = {j: 0.0 for j in JOINTS}
        self.current = {j: 40 for j in JOINTS}
        self.temperature = {j: 35 for j in JOINTS}
        self.writes = []

    def sync_read(self, name, normalize=True, num_retry=0):
        return {"Present_Position": self.position, "Present_Current": self.current,
                "Present_Temperature": self.temperature}[name]

    def sync_write(self, name, values):
        self.writes.append(dict(values))
        self.position = dict(values)

    def disable_torque(self):
        pass


class Robot:
    def __init__(self) -> None:
        self.bus = Bus()
        self.sent = []
        self.on_send = None

    def get_observation(self):
        return {**{f"{j}.pos": v for j, v in self.bus.position.items()},
                "top": np.zeros((240, 320, 3), np.uint8)}

    def send_action(self, action):
        self.sent.append(action)
        self.bus.position = {j: action[f"{j}.pos"] for j in JOINTS}
        if self.on_send:
            self.on_send(len(self.sent))
        return action


def limits(**overrides) -> Limits:
    return Limits(joint_min={j: -60.0 for j in JOINTS}, joint_max={j: 60.0 for j in JOINTS},
                  home={j: 0.0 for j in JOINTS}, **overrides)


# -- safety ----------------------------------------------------------------------


def test_goals_are_clipped_to_the_box_and_the_step_size():
    robot = Robot()
    arm = SafeArm(robot, limits(max_velocity_deg_s=1e9, max_accel_deg_s2=1e9), fps=30)
    sent = arm.send(np.array([90.0, 5.0, 0.0, 0.0, 0.0]))
    assert sent[0] == pytest.approx(10.0)  # clipped to 60, then 10 deg from 0
    assert sent[1] == pytest.approx(5.0)


def test_acceleration_holds_the_arm_until_unlocked():
    robot = Robot()
    arm = SafeArm(robot, limits(max_velocity_deg_s=1e9, max_step_deg=90), fps=30)
    arm.send(np.zeros(N))
    arm.send(np.zeros(N))
    with pytest.raises(SafetyStop, match="acceleration on shoulder_lift"):
        arm.send(np.array([0.0, 6.0, 0.0, 0.0, 0.0]))  # 6 deg in 1/30 s from rest
    assert robot.bus.writes, "held at the present position"
    with pytest.raises(SafetyStop, match="locked"):
        arm.send(np.zeros(N))
    arm.unlock()
    arm.send(np.zeros(N))


def test_speed_current_and_operator_stop():
    robot = Robot()
    arm = SafeArm(robot, limits(max_velocity_deg_s=100, max_accel_deg_s2=1e9, max_step_deg=90), fps=30)
    arm.send(np.zeros(N))
    with pytest.raises(SafetyStop, match="speed on elbow_flex"):
        arm.send(np.array([0, 0, 5.0, 0, 0]))  # 150 deg/s
    arm.unlock()
    robot.bus.current["wrist_roll"] = 0x8000 | 450
    arm.observe()
    arm.observe()  # a spike shorter than current_ticks is tolerated
    with pytest.raises(SafetyStop, match="current on wrist_roll"):
        arm.observe()
    assert arm.peak_current == 450
    robot.bus.current["wrist_roll"] = 10
    arm.unlock()
    arm.stop()
    with pytest.raises(SafetyStop, match="operator stop"):
        arm.send(np.zeros(N))


def test_limits_file_round_trip(tmp_path):
    path = tmp_path / "limits.json"
    path.write_text(json.dumps({k: {j: v for j in JOINTS} for k, v in (("min", -1), ("max", 1), ("home", 0))}))
    assert Limits.load(path, max_current=0).max_current == 0
    with pytest.raises(SystemExit):
        Limits.load(tmp_path / "missing.json")


# -- reward ----------------------------------------------------------------------


def test_reward_rules():
    cfg = RewardConfig(timeout_s=15, step_cost=0.01)
    assert score(("D2",), 3.0, cfg).label == "success" and score(("D2",), 3.0, cfg).reward == 1.0
    assert score(("D3", "D2"), 3.0, cfg).label == "wrong_button"
    assert score((), 15.0, cfg).label == "timeout"
    running = score((), 1.0, cfg)
    assert not running.done and running.reward == pytest.approx(-0.01)


# -- SAC -------------------------------------------------------------------------


def test_sac_starts_at_zero_and_learns_a_bandit():
    sac = SAC(SACConfig(obs_dim=3, action_dim=2, batch_size=64, gamma=0.0), seed=0)
    assert np.allclose(sac.act(np.zeros(3), std_scale=0.0), 0.0)
    rng = np.random.default_rng(0)
    target = np.array([0.6, -0.4])
    for _ in range(1500):
        obs = rng.normal(size=3).astype(np.float32)
        action = rng.uniform(-1, 1, 2).astype(np.float32)
        sac.buffer.add(obs, action, -np.sum((action - target) ** 2), obs, True)
    for _ in range(800):
        sac.update()
    sac.publish()
    assert np.abs(sac.act(np.zeros(3), std_scale=0.0) - target).max() < 0.15


def test_replay_buffer_round_trip():
    buffer = ReplayBuffer(2, 1, 10)
    for i in range(12):
        buffer.add([i, i], [i], float(i), [i, i], i % 2 == 0)
    copy = ReplayBuffer(2, 1, 10)
    copy.load_state_dict(buffer.state_dict())
    assert copy.size == 10 and set(copy.reward.tolist()) == set(buffer.reward.tolist())


# -- methods ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def act_dir(tmp_path_factory):
    return make_act_checkpoint(tmp_path_factory.mktemp("act"))


def test_residual_act_zero_action_is_act(act_dir):
    from corl_rl_demo.agent.methods import ACTBase

    method = ResidualACT(ACTBase(act_dir, device="cpu"), steps_per_decision=10, scale_deg=5.0)
    obs = Robot().get_observation()
    decision = method.decide(obs, np.zeros(N))
    assert decision.obs.shape == (method.obs_dim,)
    base = method.goals(decision, np.zeros(N))
    assert base.shape == (10, N)
    np.testing.assert_allclose(method.goals(decision, np.ones(N)) - base, 5.0, atol=1e-5)


class FakeFlow:
    class config:
        horizon, feature_dim, cameras = 30, 8, ("top",)

    def __init__(self):
        self.noises = []

    def feature(self, images, state, elapsed_s=0.0):
        return torch.zeros(1, 8)

    def chunk(self, feature, state, noise):
        self.noises.append(noise)
        return np.repeat(state[None], 30, axis=0) + noise.mean()


def test_dsrl_noise_rows_and_bound():
    flow = FakeFlow()
    method = DSRLFlow(flow, steps_per_decision=10, noise_steps=5, bound=1.5)
    decision = method.decide(Robot().get_observation(), np.zeros(N))
    goals = method.goals(decision, np.linspace(-1, 1, 5 * N))
    noise = flow.noises[-1]
    assert goals.shape == (10, N) and noise.shape == (30, N)
    assert np.abs(noise).max() == pytest.approx(1.5)
    np.testing.assert_allclose(noise[5:], np.repeat(noise[4:5], 25, axis=0))


# -- episode loop ----------------------------------------------------------------


class Buttons:
    def __init__(self):
        self.latch = PressLatch()

    def seconds_since_reading(self):
        return 0.0


class StillMethod:
    """Holds the joints; obs is the tick count so far."""

    obs_dim, action_dim = 2, N

    def __init__(self, k=3):
        self.k = k

    def decide(self, robot_obs, state, elapsed_s=0.0):
        from corl_rl_demo.agent.methods import Decision

        return Decision(np.array([state[0], 1.0], np.float32), state, None)

    def goals(self, decision, action):
        return np.repeat(decision.state[None], self.k, axis=0)


def _episode(robot, buttons, **reward):
    sac = SAC(SACConfig(obs_dim=2, action_dim=N), seed=0)
    arm = SafeArm(robot, limits(), fps=1000)
    result = run_episode(arm=arm, buttons=buttons, method=StillMethod(), sac=sac, learner=None,
                         reward=RewardConfig(**reward), loop=LoopConfig(fps=1000), std_scale=0.0,
                         keys=Keys())
    return result, sac.buffer


def test_green_press_ends_the_episode_as_a_success():
    robot, buttons = Robot(), Buttons()
    buttons.latch.update({"D2": False, "D3": False})
    robot.on_send = lambda n: n == 7 and buttons.latch.update({"D2": True})
    result, buffer = _episode(robot, buttons)
    assert result["success"] and result["outcome"] == "success" and result["ticks"] == 7
    assert buffer.size == 3 and buffer.done[:3].tolist() == [0, 0, 1] and buffer.reward[2] == 1.0


def test_wrong_button_and_timeout_are_failures():
    robot, buttons = Robot(), Buttons()
    robot.on_send = lambda n: n == 2 and buttons.latch.update({"D3": True})
    assert _episode(robot, buttons)[0]["outcome"] == "wrong_button"
    result, buffer = _episode(Robot(), Buttons(), timeout_s=0.0)
    assert result["outcome"] == "timeout" and buffer.done[0] == 1


def test_safety_stop_ends_the_episode_with_the_penalty():
    robot = Robot()
    robot.on_send = lambda n: n == 4 and robot.bus.current.update({"shoulder_pan": 900})
    result, buffer = _episode(robot, Buttons(), safety_penalty=2.0)
    assert result["outcome"] == "safety" and "current" in result["reason"]
    assert buffer.reward[buffer.size - 1] == -2.0 and buffer.done[buffer.size - 1] == 1


# -- flow ------------------------------------------------------------------------


def test_chunk_targets_are_relative_and_padded():
    from corl_rl_demo.agent.flow import chunk_targets

    states = np.arange(20 * N, dtype=np.float32).reshape(20, N)
    actions = states + 1
    chunks, pad = chunk_targets(states, actions, np.repeat([0, 1], 10), horizon=4)
    np.testing.assert_allclose(chunks[0, 0], 1.0)
    assert pad[8].tolist() == [False, False, True, True] and not pad[10].any()


def test_flow_trains_on_a_recorded_dataset_and_plans(tmp_path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from corl_rl_demo.agent.flow import FlowConfig, FlowPolicy, TrainConfig, train

    names = [f"{j}.pos" for j in JOINTS]
    features = {
        "observation.state": {"dtype": "float32", "shape": (N,), "names": names},
        "action": {"dtype": "float32", "shape": (N,), "names": names},
        "observation.images.top": {"dtype": "video", "shape": (48, 64, 3),
                                   "names": ["height", "width", "channels"]},
    }
    root = tmp_path / "demos"
    dataset = LeRobotDataset.create("local/demos", fps=30, features=features, root=root, use_videos=True)
    for _ in range(2):
        for i in range(20):
            state = np.full(N, float(i), np.float32)
            dataset.add_frame({"observation.state": state, "action": state + 1,
                               "observation.images.top": np.full((48, 64, 3), i * 10, np.uint8),
                               "task": "Press the green button"})
        dataset.save_episode(parallel_encoding=False)
    dataset.finalize()
    out = train(TrainConfig(dataset="local/demos", root=str(root), output=str(tmp_path / "flow"),
                            steps=3, batch_size=4, num_workers=0, log_every=1,
                            flow=FlowConfig(image_size=32, hidden_dim=64, pretrained_encoder=False)))
    policy = FlowPolicy.load(out, device="cpu")
    method = DSRLFlow(policy, steps_per_decision=10)
    decision = method.decide(Robot().get_observation(), np.zeros(N))
    assert decision.obs.shape == (method.obs_dim,) == (512 + N + 1,)
    assert method.goals(decision, np.zeros(method.action_dim)).shape == (10, N)


def test_dry_run_never_commands_the_arm():
    robot = Robot()
    arm = SafeArm(robot, limits(), fps=30, dry_run=True)
    arm.send(np.full(N, 3.0))
    arm.move_to(np.full(N, 20.0))
    assert robot.sent == [] and robot.bus.writes == [] and len(arm.dry_commands) == 1


def test_limits_from_demos():
    states = np.array([[0.0] * N, [10.0] * N, [5.0] * N, [1.0] * N])
    actions = states + 1
    limits = Limits.from_dataset(states, actions, np.array([0, 0, 1, 1]), margin_deg=2.0)
    assert limits.joint_min["shoulder_pan"] == -2.0 and limits.joint_max["shoulder_pan"] == 13.0
    assert limits.home["shoulder_pan"] == 2.5  # mean of the two start poses


def test_servo_caps_write_ram_registers():
    from corl_rl_demo.agent.safety import apply_servo_caps

    class CapBus:
        motors = {j: None for j in JOINTS}
        writes = []

        def write(self, name, motor, value, normalize=True):
            self.writes.append((name, motor, value, normalize))

    bus = CapBus()
    apply_servo_caps(bus, torque=600, speed_deg_s=90, acceleration=100)
    assert ("Torque_Limit", "shoulder_pan", 600, False) in bus.writes
    assert ("Goal_Velocity", "wrist_roll", 1024, False) in bus.writes
    assert {name for name, *_ in bus.writes} == {"Torque_Limit", "Goal_Velocity", "Acceleration"}


def test_flow_reads_the_clock_and_old_checkpoints_load_unchanged(tmp_path):
    """The demos wait at rest before moving; only the time since the episode
    began can tell the policy when to go. A checkpoint from before the time
    input loads with a zero column for it, so it first acts exactly as before."""
    from corl_rl_demo.agent.flow import FlowConfig, FlowNet, load_weights

    old = FlowNet(FlowConfig(image_size=32, hidden_dim=64, pretrained_encoder=False, time_scale_s=0.0))
    new = FlowNet(FlowConfig(image_size=32, hidden_dim=64, pretrained_encoder=False, time_scale_s=10.0))
    load_weights(new, old.state_dict())
    old.eval(), new.eval()
    images, state = [torch.rand(1, 3, 32, 32)], torch.zeros(1, N)
    with torch.no_grad():
        before = old.embed(images, state)
        at_3s = new.embed(images, state, torch.tensor([3.0]))
        torch.nn.init.normal_(new.feature[0].weight)  # once the time column is trained
        changed = new.embed(images, state, torch.tensor([3.0])) - new.embed(images, state, torch.tensor([0.0]))
    torch.testing.assert_close(before, at_3s)
    assert changed.abs().max() > 0
    with pytest.raises(ValueError, match="time since the episode began"):
        new.embed(images, state)


def test_rates_use_the_time_that_passed_between_commands(monkeypatch):
    """After the arm waits for the next plan, a jump is spread over the wait."""
    from corl_rl_demo.agent import safety

    clock = iter([0.0, 1 / 30, 2 / 30, 2 / 30 + 0.15])
    monkeypatch.setattr(safety.time, "monotonic", lambda: next(clock))
    arm = SafeArm(Robot(), limits(max_step_deg=90), fps=30)
    for _ in range(3):
        arm.gate(np.zeros(N))
    arm.gate(np.array([0, 0, 2.5, 0, 0]))  # 2.5 deg in one tick would be 5600 deg/s^2


def test_blend_fades_from_the_old_plan_into_the_new():
    from corl_rl_demo.agent.run import blend

    old_tail = np.zeros((20, N))
    new = np.full((10, N), 10.0)
    out = blend(old_tail, new, ticks=4)
    assert out[:4, 0].tolist() == pytest.approx([2.0, 4.0, 6.0, 8.0])
    assert (out[4:] == 10.0).all()
    assert blend(None, new, 4) is new


def test_smooth_removes_zig_zag_and_keeps_a_ramp():
    from corl_rl_demo.agent.methods import smooth

    ramp = np.linspace(0, 29, 30)[:, None].repeat(N, 1)
    np.testing.assert_allclose(smooth(ramp, 5)[2:-2], ramp[2:-2])
    zigzag = ramp + np.where(np.arange(30) % 2, 1.5, -1.5)[:, None]
    assert np.abs(np.diff(smooth(zigzag, 5), 2, axis=0)).max() < 0.25 * np.abs(np.diff(zigzag, 2, axis=0)).max()


def test_diffusion_base_plans_from_given_noise_and_runs_an_episode(tmp_path):
    """The official Diffusion Policy as the base: chunks of n_action_steps absolute
    goals from a noise we choose, its observation history fed every tick."""
    from act_fixture import make_diffusion_checkpoint

    from corl_rl_demo.agent.methods import DiffusionBase, DSRLDiffusion

    method = DSRLDiffusion(DiffusionBase(make_diffusion_checkpoint(tmp_path), device="cpu"))
    obs = Robot().get_observation()
    method.begin_episode()
    decision = method.decide(obs, np.zeros(N))
    assert decision.obs.shape == (method.obs_dim,)
    zero = method.goals(decision, np.zeros(method.action_dim))
    assert zero.shape == (8, N) and method.tail.shape[1] == N
    np.testing.assert_allclose(method.goals(decision, np.zeros(method.action_dim)), zero, atol=1e-4)
    assert not np.allclose(method.goals(decision, np.ones(method.action_dim)), zero)
    method.random_noise = "random"
    assert not np.allclose(method.goals(decision, np.zeros(method.action_dim)), zero)

    robot, buttons = Robot(), Buttons()
    robot.on_send = lambda n: n == 12 and buttons.latch.update({"D2": True})
    pushes = []
    original = method.observe_tick
    method.observe_tick = lambda o, s: (pushes.append(1), original(o, s))
    sac = SAC(SACConfig(obs_dim=method.obs_dim, action_dim=method.action_dim), seed=0)
    result = run_episode(arm=SafeArm(robot, limits(max_velocity_deg_s=1e9, max_accel_deg_s2=1e9, max_step_deg=90),
                                     fps=1000),
                         buttons=buttons, method=method, sac=sac, learner=None, reward=RewardConfig(),
                         loop=LoopConfig(fps=1000), std_scale=0.0, keys=Keys())
    assert result["success"] and result["ticks"] == 12
    assert len(pushes) == 7 + 3  # every tick but the first of each plan


def test_stretch_follows_the_same_path_more_slowly():
    from corl_rl_demo.agent.run import stretch

    start = np.zeros(N)
    goals = np.outer(np.arange(1, 5), np.ones(N)) * 4.0  # 4 deg per tick
    slow = stretch(goals, start, 0.5)
    assert len(slow) == 8
    np.testing.assert_allclose(slow[-1], goals[-1])
    np.testing.assert_allclose(np.diff(np.vstack((start, slow)), axis=0), 2.0)  # 2 deg per tick
    assert stretch(goals, start, 1.0) is goals
