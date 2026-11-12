"""The online RL loop on the rig, and the ``rl-agent`` command.

    uv run rl-agent limits --dataset pravsels/corl-press-green   # joint box and start pose from the demos
    uv run rl-agent flow-train --dataset pravsels/corl-press-green
    uv run rl-agent run --method dsrl-flow --policy outputs/flow --dry-run     # torque off, nothing moves
    uv run rl-agent run --method dsrl-flow --policy outputs/flow --no-learn --episodes 5
    uv run rl-agent run --method dsrl-flow --policy outputs/flow

Each episode starts from a pose drawn near home, runs until a button is pressed,
the timeout, or a safety stop, and is logged to ``<run>/episodes.jsonl`` (for the
display). SAC learns on a background thread while the arm moves; every
``eval_every``-th episode runs SAC's mean action, so the log shows the policy
itself next to the exploring one. Keys: space stops the arm, p pauses after the
episode, q quits. The learner is saved every ``save_every`` episodes and on exit;
``--resume`` continues from it.
"""

from __future__ import annotations

import argparse
import json
import select
import sys
import termios
import threading
import time
import tty
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from corl_rl_demo import rig
from corl_rl_demo.agent.episode import RewardConfig, safety_outcome, score
from corl_rl_demo.agent.safety import LIMITS_FILE, Limits, Overheat, SafeArm, SafetyStop
from corl_rl_demo.agent.sac import SAC, LearnerThread, SACConfig
from corl_rl_demo.rig import JOINTS


@dataclass
class LoopConfig:
    fps: int = rig.CAMERA_FPS
    steps_per_decision: int = 10
    blend_ticks: int = 5  # cross-fade from the previous plan into a new one
    speed_scale: float = 1.0  # < 1 follows each plan's path more slowly (0.5: half speed)
    episodes: int = 1000
    start_jitter_deg: float = 3.0
    exploration: float = 1.0  # scales SAC's std; 0 runs its mean
    eval_every: int = 10
    save_every: int = 10
    silent_module_s: float = 5.0
    seed: int = 0


class Keys:
    """space: stop the arm; p: pause after this episode; q: quit."""

    def __init__(self, arm: SafeArm | None = None) -> None:
        self.arm = arm
        self.quit = threading.Event()
        self.pause = threading.Event()
        self._saved = None

    def start(self) -> None:
        if not sys.stdin.isatty():
            return
        self._saved = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin)
        threading.Thread(target=self._read, daemon=True).start()

    def close(self) -> None:
        if self._saved is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._saved)

    def _read(self) -> None:
        while not self.quit.is_set():
            if not select.select([sys.stdin], [], [], 0.1)[0]:
                continue
            key = sys.stdin.read(1)
            if key == " " and self.arm is not None:
                self.arm.stop()
                print("\n[agent] stop: holding the arm", flush=True)
            elif key == "p":
                self.pause.set()
                print("\n[agent] pausing after this episode; Enter to go on", flush=True)
            elif key == "q":
                self.quit.set()
            elif key in "\r\n":
                self.pause.clear()


def joints(robot_obs: dict[str, Any]) -> np.ndarray:
    return np.array([float(robot_obs[f"{j}.pos"]) for j in JOINTS])


def run_episode(
    *, arm: SafeArm, buttons: Any, method: Any, sac: SAC, learner: LearnerThread | None,
    reward: RewardConfig, loop: LoopConfig, std_scale: float, keys: Keys,
) -> dict[str, Any]:
    """One episode from wherever the arm is. Adds its transitions to SAC's buffer."""
    period = 1.0 / loop.fps
    if hasattr(method, "begin_episode"):
        method.begin_episode()
    buttons.latch.begin_episode()
    started = time.monotonic()
    previous: tuple[np.ndarray, np.ndarray, float] | None = None
    decisions, ticks = 0, 0
    outcome = None
    decision = action = None
    previous_tail: np.ndarray | None = None
    last_goal: np.ndarray | None = None
    trace: list[dict[str, Any]] = []  # one entry per plan, for seeing what the arm did

    def add(obs, action, r, next_obs, done) -> None:
        sac.buffer.add(obs, action, r, next_obs, done)
        if learner is not None:
            learner.transitions_added()

    try:
        while outcome is None or not outcome.done:
            if keys.quit.is_set():
                return {"outcome": "quit", "success": False, "decisions": decisions, "ticks": ticks}
            planning = time.monotonic()
            robot_obs = arm.observe()
            decision = method.decide(robot_obs, joints(robot_obs), time.monotonic() - started)
            if previous is not None:
                add(*previous, decision.obs, False)
            action = sac.act(decision.obs, std_scale=std_scale)
            goals = blend(previous_tail, method.goals(decision, action), loop.blend_ticks)
            trace.append({
                "t": round(planning - started, 2), "plan_ms": round((time.monotonic() - planning) * 1000),
                "joints": np.round(decision.state, 1).tolist(), "plan_end": np.round(goals[-1], 1).tolist(),
            })
            previous_tail = getattr(method, "tail", None)
            goals = stretch(goals, decision.state if last_goal is None else last_goal, loop.speed_scale)
            last_goal = goals[-1]
            observe_tick = getattr(method, "observe_tick", None)
            for i, goal in enumerate(goals):
                tick = time.monotonic()
                if observe_tick is not None and i > 0:
                    # Policies with an observation history see every tick, not only the plans.
                    tick_obs = arm.observe()
                    observe_tick(tick_obs, joints(tick_obs))
                arm.send(goal)
                ticks += 1
                if buttons.latch.presses():
                    break
                time.sleep(max(0.0, period - (time.monotonic() - tick)))
            decisions += 1
            outcome = score(buttons.latch.presses(), time.monotonic() - started, reward)
            if outcome.done:
                add(decision.obs, action, outcome.reward, decision.obs, True)
            else:
                previous = (decision.obs, action, outcome.reward)
    except SafetyStop as stop:
        print(f"[agent] safety stop: {stop}", flush=True)
        outcome = safety_outcome(reward)
        if decision is not None and action is not None:
            add(decision.obs, action, outcome.reward, decision.obs, True)
        return {"outcome": "safety", "success": False, "reason": str(stop),
                "decisions": decisions + 1, "ticks": ticks, "trace": trace}
    elapsed = time.monotonic() - started
    return {
        "outcome": outcome.label, "success": outcome.label == "success",
        "time_to_press_s": round(elapsed, 2) if outcome.label == "success" else None,
        "decisions": decisions, "ticks": ticks, "trace": trace,
    }


def stretch(goals: np.ndarray, start: np.ndarray, speed_scale: float) -> np.ndarray:
    """The same path from ``start`` through ``goals``, spread over 1/speed_scale as
    many ticks, so the arm moves at ``speed_scale`` of the planned speed."""
    if speed_scale >= 1.0:
        return goals
    path = np.vstack((start, goals))
    ticks = int(round(len(goals) / speed_scale))
    where = np.linspace(0, len(goals), ticks + 1)[1:]
    return np.stack([np.interp(where, np.arange(len(path)), path[:, j]) for j in range(path.shape[1])], axis=1)


def blend(previous_tail: np.ndarray | None, goals: np.ndarray, ticks: int) -> np.ndarray:
    """Cross-fade from where the last plan was heading into the new one over the
    first ``ticks`` ticks, so switching chunks does not jerk the arm."""
    if previous_tail is None or ticks <= 0 or len(previous_tail) == 0:
        return goals
    n = min(ticks, len(previous_tail), len(goals))
    weight = (np.arange(1, n + 1) / (n + 1))[:, None]
    goals = np.array(goals, dtype=np.float64)
    goals[:n] = weight * goals[:n] + (1 - weight) * previous_tail[:n]
    return goals


def reset(arm: SafeArm, buttons: Any, loop: LoopConfig, rng: np.random.Generator) -> None:
    """Wait for cool motors and a live button module, then glide to a start pose
    drawn near home (the plan's design choice 4)."""
    while (hottest := arm.check_heat()) >= arm.limits.pause_c:
        print(f"[agent] motors at {hottest:.0f} C; cooling", flush=True)
        arm.robot.bus.disable_torque()
        time.sleep(30)
    while buttons.seconds_since_reading() > loop.silent_module_s:
        print("[agent] no reading from the button module; waiting", flush=True)
        time.sleep(2)
    home = np.array([arm.limits.home[j] for j in JOINTS])
    arm.move_to(home + rng.uniform(-loop.start_jitter_deg, loop.start_jitter_deg, len(JOINTS)))
    arm.unlock()


def build_method(name: str, policy: str, steps_per_decision: int, device: str | None) -> Any:
    from corl_rl_demo.agent.methods import ACTBase, DSRLFlow, ResidualACT

    if name == "residual-act":
        return ResidualACT(ACTBase(policy, device=device), steps_per_decision=steps_per_decision)
    if name == "dsrl-flow":
        from corl_rl_demo.agent.flow import FlowPolicy

        return DSRLFlow(FlowPolicy.load(policy, device=device), steps_per_decision=steps_per_decision)
    if name == "dsrl-diffusion":
        from corl_rl_demo.agent.methods import DiffusionBase, DSRLDiffusion

        # Plans every n_action_steps ticks, as the policy was trained to.
        return DSRLDiffusion(DiffusionBase(policy, device=device))
    raise ValueError(f"unknown method {name!r}")


def cameras_for(method: Any) -> list[str]:
    if hasattr(method, "base"):
        return list(method.base.cameras)
    return list(method.flow.config.cameras)


def limits_from_args(args: argparse.Namespace) -> Limits:
    overrides = {
        name: value for name, value in (
            ("max_velocity_deg_s", args.max_velocity), ("max_accel_deg_s2", args.max_accel),
            ("max_step_deg", args.max_step), ("max_current", args.max_current),
        ) if value is not None
    }
    return Limits.load(args.limits, **overrides)


def run(args: argparse.Namespace) -> None:
    from lerobot.cameras.opencv import OpenCVCameraConfig

    from corl_rl_demo.agent.safety import align_goals_to_present, apply_servo_caps

    from corl_rl_demo.buttons import ButtonModule
    from corl_rl_demo.config_five_motor import SO101FiveMotorFollowerConfig
    from corl_rl_demo.five_motor import SO101FiveMotorFollower

    loop = LoopConfig(steps_per_decision=args.steps_per_decision, episodes=args.episodes,
                      exploration=args.exploration, seed=args.seed, speed_scale=args.speed_scale)
    if not 0.1 <= loop.speed_scale <= 1.0:
        raise SystemExit("--speed-scale must be within 0.1..1")
    method = build_method(args.method, args.policy, loop.steps_per_decision, args.device)
    camera_index = {"top": args.top_camera, "wrist": args.wrist_camera}
    cameras = {
        name: OpenCVCameraConfig(index_or_path=camera_index[name], width=rig.CAMERA_WIDTH,
                                 height=rig.CAMERA_HEIGHT, fps=rig.CAMERA_FPS)
        for name in cameras_for(method)
    }
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    sac = SAC(SACConfig(obs_dim=method.obs_dim, action_dim=method.action_dim), seed=loop.seed)
    checkpoint = run_dir / "learner.pt"
    if args.resume:
        sac.load_state_dict(torch.load(checkpoint, weights_only=False))
        print(f"[agent] resumed at {sac.steps} updates, {sac.buffer.size} transitions", flush=True)
    (run_dir / "config.json").write_text(json.dumps(
        {"method": args.method, "policy": args.policy, "loop": asdict(loop), "sac": asdict(sac.config)}, indent=2))

    robot = SO101FiveMotorFollower(SO101FiveMotorFollowerConfig(
        port=args.arm_port, id=args.robot_id, cameras=cameras))
    align_goals_to_present(args.arm_port)
    robot.connect(calibrate=False)
    if args.dry_run:
        robot.bus.disable_torque()
        print("[agent] dry run: torque off, the arm is never commanded", flush=True)
    else:
        apply_servo_caps(robot.bus, torque=args.torque_limit, speed_deg_s=args.servo_speed,
                         acceleration=args.servo_acceleration)
    limits = limits_from_args(args)
    print(f"[agent] limits: {args.limits}; speed {limits.max_velocity_deg_s} deg/s, acceleration "
          f"{limits.max_accel_deg_s2} deg/s^2, step {limits.max_step_deg} deg, current "
          f"{limits.max_current}, pause {limits.pause_c} C, stop {limits.hot_c} C; servo torque "
          f"{args.torque_limit}/1000", flush=True)
    arm = SafeArm(robot, limits, loop.fps, dry_run=args.dry_run)
    buttons = ButtonModule.open(args.button_port)
    learning = not (args.no_learn or args.dry_run)
    if args.base_noise != "zero":
        if learning or not hasattr(method, "random_noise"):
            raise SystemExit("--base-noise runs the base alone: add --no-learn, with dsrl-flow or dsrl-diffusion")
        method.random_noise = args.base_noise
    # The first plan on the GPU takes seconds (kernels compile); do it before the arm waits on it.
    started = time.monotonic()
    warm = arm.observe()
    method.goals(method.decide(warm, joints(warm), 0.0), np.zeros(method.action_dim))
    print(f"[agent] policy warmed up in {time.monotonic() - started:.1f} s", flush=True)
    learner = LearnerThread(sac, utd=args.utd) if learning else None
    if learner is not None:
        learner.start()
    keys = Keys(arm)
    keys.start()
    rng = np.random.default_rng(loop.seed)
    log = (run_dir / "episodes.jsonl").open("a")
    recent: list[bool] = []
    try:
        for index in range(1, loop.episodes + 1):
            if keys.quit.is_set():
                break
            while keys.pause.is_set() and not keys.quit.is_set():
                time.sleep(0.1)
            reset(arm, buttons, loop, rng)
            evaluating = loop.eval_every > 0 and index % loop.eval_every == 0
            result = run_episode(
                arm=arm, buttons=buttons, method=method, sac=sac, learner=learner,
                reward=RewardConfig(timeout_s=args.episode_timeout), loop=loop, keys=keys,
                std_scale=0.0 if evaluating or not learning else loop.exploration,
            )
            result["peak_current"] = arm.peak_current
            result.update({k: round(v, 2) if isinstance(v, float) else v for k, v in arm.stats.items()})
            result["hottest_c"] = max(arm.temperatures().values())
            if result["outcome"] == "quit":
                break
            if not evaluating:
                recent = (recent + [result["success"]])[-20:]
            record = {
                "episode": index, "time": datetime.now(UTC).isoformat(), "method": args.method,
                "eval": evaluating, **result, "learner_updates": sac.steps,
                "transitions": sac.buffer.size, "learning": learning, "dry_run": args.dry_run,
                **({f"sac_{k}": v for k, v in learner.last.items()} if learner else {}),
            }
            log.write(json.dumps(record) + "\n")
            log.flush()
            print(f"[agent] episode {index}{' (eval)' if evaluating else ''}: {result['outcome']}"
                  f"{'' if result.get('time_to_press_s') is None else f' in {result['time_to_press_s']} s'}"
                  f"; last {len(recent)}: {sum(recent)}/{len(recent)}; {sac.steps} updates; "
                  f"peak current {result['peak_current']}, hottest {result['hottest_c']:.0f} C\n"
                  f"        policy asked up to {result['asked_ahead_max_deg']} deg ahead, sent "
                  f"{result['sent_ahead_max_deg']}; clipped: box {result['box_clips']}, step "
                  f"{result['step_clips']}; arm moved up to {result['moved_max_deg']} deg", flush=True)
            if learning and index % loop.save_every == 0:
                torch.save(sac.state_dict(), checkpoint)
    except Overheat as error:
        print(f"[agent] stopping: {error}", flush=True)
    finally:
        keys.close()
        if learner is not None:
            learner.stop()
            torch.save(sac.state_dict(), checkpoint)
        log.close()
        buttons.close()
        try:
            arm.unlock()
            arm.move_to(np.array([arm.limits.home[j] for j in JOINTS]))
        finally:
            robot.disconnect()  # torque off; the arm goes limp at its start pose


def setup_command(args: argparse.Namespace) -> None:
    from corl_rl_demo.agent.safety import setup
    from corl_rl_demo.config_five_motor import SO101FiveMotorFollowerConfig
    from corl_rl_demo.five_motor import SO101FiveMotorFollower

    from corl_rl_demo.agent.safety import align_goals_to_present

    robot = SO101FiveMotorFollower(SO101FiveMotorFollowerConfig(port=args.arm_port, id=args.robot_id))
    align_goals_to_present(args.arm_port)
    robot.connect(calibrate=False)
    try:
        setup(robot, Path(args.limits))
    finally:
        robot.disconnect()
    print(f"wrote {args.limits}")


def limits_command(args: argparse.Namespace) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(args.dataset, root=args.root, download_videos=False)
    table = dataset.hf_dataset.with_format("numpy")
    limits = Limits.from_dataset(
        np.stack(table["observation.state"]), np.stack(table["action"]),
        np.asarray(table["episode_index"]), margin_deg=args.margin,
    )
    limits.save(args.limits)
    for j in JOINTS:
        print(f"{j:14s} {limits.joint_min[j]:7.1f} .. {limits.joint_max[j]:6.1f}   home {limits.home[j]:6.1f}")
    print(f"wrote {args.limits}")


def _with_time(config, no_time: bool = False):
    """Continuing a checkpoint without the time input adds it, unless ``no_time``."""
    if not config.time_input and not no_time:
        config.time_scale_s = 10.0
    return config


def flow_train_command(args: argparse.Namespace) -> None:
    from corl_rl_demo.agent.flow import FlowConfig, TrainConfig, train

    train(TrainConfig(
        dataset=args.dataset, root=args.root, output=args.output, steps=args.steps,
        batch_size=args.batch_size, num_workers=args.num_workers,
        trim_idle_deg=args.trim_idle, init=args.init, color_jitter=args.color_jitter,
        save_every=args.save_every, keep_checkpoints=args.save_every < args.steps,
        flow=(_with_time(FlowConfig.from_json((Path(args.init) / "flow_config.json").read_text()), args.no_time)
              if args.init
              else FlowConfig(cameras=tuple(args.cameras.split(",")), pretrained_encoder=not args.no_pretrained,
                              time_scale_s=0.0 if args.no_time else 10.0)),
    ))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="rl-agent", description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    def arm_options(command: argparse.ArgumentParser) -> None:
        command.add_argument("--arm-port", default=rig.ARM_PORT)
        command.add_argument("--robot-id", default="demo_follower")
        command.add_argument("--limits", default=str(LIMITS_FILE))

    setup_parser = commands.add_parser("setup", help="record the joint box and the start pose by hand")
    arm_options(setup_parser)

    limits_parser = commands.add_parser("limits", help="the joint box and start pose from the demos")
    limits_parser.add_argument("--dataset", required=True)
    limits_parser.add_argument("--root", default=None)
    limits_parser.add_argument("--margin", type=float, default=3.0, help="degrees beyond the demos' range")
    limits_parser.add_argument("--limits", default=str(LIMITS_FILE))

    flow_parser = commands.add_parser("flow-train", help="train the flow base policy for dsrl-flow")
    flow_parser.add_argument("--dataset", required=True, help="the demos' repo id")
    flow_parser.add_argument("--root", default=None, help="the dataset's local folder, if not the HF cache")
    flow_parser.add_argument("--output", default="outputs/flow")
    flow_parser.add_argument("--cameras", default="top")
    flow_parser.add_argument("--steps", type=int, default=20_000)
    flow_parser.add_argument("--batch-size", type=int, default=32)
    flow_parser.add_argument("--num-workers", type=int, default=4)
    flow_parser.add_argument("--no-pretrained", action="store_true", help="skip the ImageNet ResNet-18 weights")
    flow_parser.add_argument("--trim-idle", type=float, default=0.0,
                             help="drop each demo's wait before a joint moves this many degrees (0 keeps it)")
    flow_parser.add_argument("--init", default=None, help="continue from this flow-train output")
    flow_parser.add_argument("--save-every", type=int, default=5_000,
                             help="checkpoint period; each one is kept in <output>/step_<N>")
    flow_parser.add_argument("--color-jitter", type=float, default=0.2,
                             help="random brightness, contrast and saturation (0 turns it off)")
    flow_parser.add_argument("--no-time", action="store_true", help="leave the time input out")

    run_parser = commands.add_parser("run", help="learn online on the rig")
    arm_options(run_parser)
    run_parser.add_argument("--method", choices=("residual-act", "dsrl-flow", "dsrl-diffusion"), required=True)
    run_parser.add_argument("--policy", required=True, help="ACT checkpoint dir, or flow-train output")
    run_parser.add_argument("--button-port", default=rig.BUTTON_PORT)
    run_parser.add_argument("--top-camera", type=int, default=rig.TOP_CAMERA)
    run_parser.add_argument("--wrist-camera", type=int, default=1)
    run_parser.add_argument("--run-dir", default="outputs/rl")
    run_parser.add_argument("--resume", action="store_true")
    run_parser.add_argument("--episodes", type=int, default=1000)
    run_parser.add_argument("--speed-scale", type=float, default=1.0,
                            help="follow each plan's path at this fraction of its speed (0.5: half)")
    run_parser.add_argument("--episode-timeout", type=float, default=15.0,
                            help="seconds without a press before an episode fails")
    run_parser.add_argument("--steps-per-decision", type=int, default=10)
    run_parser.add_argument("--exploration", type=float, default=1.0, help="scale of SAC's std; 0 runs its mean")
    run_parser.add_argument("--utd", type=int, default=4, help="SAC updates per transition")
    run_parser.add_argument("--device", default=None, help="base policy device (default mps)")
    run_parser.add_argument("--seed", type=int, default=0)
    run_parser.add_argument("--dry-run", action="store_true",
                            help="torque off: run the policy and the gate but never command the arm")
    run_parser.add_argument("--no-learn", action="store_true", help="run the policy without exploring or learning")
    run_parser.add_argument("--base-noise", choices=("zero", "random", "episode"), default="zero",
                            help="with --no-learn: the flow's noise; random draws N(0, I) each plan, "
                            "episode draws one per episode")
    run_parser.add_argument("--torque-limit", type=int, default=700, help="servo torque cap, of 1000")
    run_parser.add_argument("--servo-speed", type=float, default=180.0, help="servo speed cap, deg/s")
    run_parser.add_argument("--servo-acceleration", type=int, default=150,
                            help="servo acceleration, 100 steps/s^2 per unit (LeRobot uses 254)")
    run_parser.add_argument("--max-velocity", type=float, default=None, help="deg/s (default 150)")
    run_parser.add_argument("--max-accel", type=float, default=None, help="deg/s^2 (default 2500)")
    run_parser.add_argument("--max-step", type=float, default=None, help="deg per tick (default 10)")
    run_parser.add_argument("--max-current", type=int, default=None, help="counts (default 280)")

    args = parser.parse_args(argv)
    {"setup": setup_command, "limits": limits_command, "flow-train": flow_train_command,
     "run": run}[args.command](args)


if __name__ == "__main__":
    main()
