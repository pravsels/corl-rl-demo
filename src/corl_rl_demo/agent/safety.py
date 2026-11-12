"""Joint, step-size, speed and current limits between the agent and the arm.

Three layers, from the outside in:

1. Software, on every command. A goal outside the joint box is clipped, and so
   is a step of more than ``max_step_deg`` from where the arm is. A commanded
   speed or acceleration over its limit, a motor current over its limit for
   ``current_ticks`` ticks in a row, or the operator's stop holds the arm where it
   is and raises ``SafetyStop``: the episode ends as a failure with a penalty and
   the arm glides back to its start pose before the next one.
2. Heat, between episodes. Above ``pause_c`` the run rests with torque off until
   the motors cool; at ``hot_c`` it stops (``Overheat``).
3. The servos themselves (``apply_servo_caps``): a torque cap, a speed cap and an
   acceleration ramp, written to RAM registers, so a power cycle undoes them.

The joint box and home pose come from the demos (``from_dataset``: their range
plus a margin, and their mean start pose), or from a hand sweep (``setup``).
``SafeArm(dry_run=True)`` runs everything but never commands the arm.
"""

from __future__ import annotations

import json
import select
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from corl_rl_demo.rig import JOINTS

LIMITS_FILE = Path("outputs/limits.json")


class SafetyStop(RuntimeError):
    """The gate refused a command; the arm is held until ``unlock``."""


class Overheat(RuntimeError):
    """A motor is too hot to go on."""


@dataclass
class Limits:
    joint_min: dict[str, float]
    joint_max: dict[str, float]
    home: dict[str, float]
    # The demos (pravsels/corl-press-green) move at most 135 deg/s, accelerate at
    # most 2000 deg/s^2 and command at most 17 deg ahead of the arm.
    max_step_deg: float = 10.0  # per control tick, relative to the present position
    max_velocity_deg_s: float = 150.0
    max_accel_deg_s2: float = 2500.0
    max_current: int = 280  # Feetech counts (6.5 mA); the servos cut out at 310
    current_ticks: int = 3  # consecutive ticks over max_current before stopping
    hot_c: float = 50.0
    pause_c: float = 45.0  # rest between episodes until the motors cool below this

    @classmethod
    def load(cls, path: str | Path = LIMITS_FILE, **overrides: Any) -> Limits:
        path = Path(path)
        if not path.is_file():
            raise SystemExit(f"{path} does not exist; run `rl-agent setup` first")
        data = json.loads(path.read_text())
        for key in ("min", "max", "home"):
            missing = set(JOINTS).difference(data.get(key, {}))
            if missing:
                raise ValueError(f"{path}: '{key}' lacks joints {sorted(missing)}")
        return cls(
            joint_min={j: float(data["min"][j]) for j in JOINTS},
            joint_max={j: float(data["max"][j]) for j in JOINTS},
            home={j: float(data["home"][j]) for j in JOINTS},
            **overrides,
        )

    @classmethod
    def from_dataset(cls, states: np.ndarray, actions: np.ndarray, episodes: np.ndarray,
                     margin_deg: float = 3.0, **overrides: Any) -> Limits:
        """The demos' joint range plus ``margin_deg``, and their mean start pose."""
        both = np.concatenate((states, actions))
        starts = np.stack([states[episodes == e][0] for e in np.unique(episodes)])
        low, high = both.min(0) - margin_deg, both.max(0) + margin_deg
        return cls(
            joint_min=dict(zip(JOINTS, low.tolist())), joint_max=dict(zip(JOINTS, high.tolist())),
            home=dict(zip(JOINTS, starts.mean(0).tolist())), **overrides,
        )

    def save(self, path: str | Path = LIMITS_FILE) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"min": self.joint_min, "max": self.joint_max, "home": self.home}, indent=2) + "\n")

    def clip(self, goal: np.ndarray) -> np.ndarray:
        low = np.array([self.joint_min[j] for j in JOINTS])
        high = np.array([self.joint_max[j] for j in JOINTS])
        return np.clip(goal, low, high)


def current_magnitude(raw: int) -> int:
    """Feetech reports current as sign-magnitude with the sign in bit 15."""
    return int(raw) & 0x7FFF


STEPS_PER_DEG = 4096 / 360
# A read the bus drops once in a while (no status packet) is retried this often
# before the run stops.
READ_RETRIES = 3


def align_goals_to_present(port: str, motor_ids: range = range(1, len(JOINTS) + 1)) -> None:
    """Set each servo's goal to where it is, before LeRobot enables torque on
    connect: a stale goal from an earlier session would otherwise make the arm
    jump to it. Reads and writes raw registers; torque is left as it is."""
    import scservo_sdk as scs

    handler, packets = scs.PortHandler(port), scs.PacketHandler(0)
    if not handler.openPort() or not handler.setBaudRate(1_000_000):
        raise RuntimeError(f"cannot open {port}")
    try:
        for motor_id in motor_ids:
            present, result, _ = packets.read2ByteTxRx(handler, motor_id, 56)  # Present_Position
            if result != scs.COMM_SUCCESS:
                raise RuntimeError(f"motor {motor_id} did not answer on {port}")
            packets.write2ByteTxRx(handler, motor_id, 42, present)  # Goal_Position
    finally:
        handler.closePort()


def apply_servo_caps(bus: Any, *, torque: int = 700, speed_deg_s: float = 180.0,
                     acceleration: int = 150) -> None:
    """Cap each servo's torque (of 1000), speed and acceleration (100 steps/s^2
    per unit; LeRobot sets 254, the fastest). RAM registers: a power cycle resets them."""
    for motor in bus.motors:
        bus.write("Torque_Limit", motor, int(torque), normalize=False)
        bus.write("Goal_Velocity", motor, int(speed_deg_s * STEPS_PER_DEG), normalize=False)
        bus.write("Acceleration", motor, int(acceleration), normalize=False)


class SafeArm:
    """The five-motor follower behind the gate. ``robot`` is a connected
    ``SO101FiveMotorFollower`` (or anything with its ``bus``, ``get_observation``
    and ``send_action``)."""

    def __init__(self, robot: Any, limits: Limits, fps: int, *, dry_run: bool = False) -> None:
        self.robot = robot
        self.limits = limits
        self.dt = 1.0 / fps
        self.dry_run = dry_run
        self._history: list[tuple[np.ndarray, float]] = []  # (goal, time sent)
        self._locked: str | None = None
        self._stop = threading.Event()
        self._over_current = 0
        self.peak_current = 0
        self.dry_commands: list[np.ndarray] = []
        self.reset_stats()

    def reset_stats(self) -> None:
        """Per-episode record of what the policy asked for and what the gate let through."""
        self.stats = {"asked_ahead_max_deg": 0.0, "sent_ahead_max_deg": 0.0, "box_clips": 0,
                      "step_clips": 0, "moved_max_deg": 0.0}
        self._start: np.ndarray | None = None

    # reading
    def positions(self) -> np.ndarray:
        values = self.robot.bus.sync_read("Present_Position", num_retry=READ_RETRIES)
        return np.array([float(values[j]) for j in JOINTS])

    def observe(self) -> dict[str, Any]:
        self.check_current()
        return self.robot.get_observation()

    def temperatures(self) -> dict[str, float]:
        return {k: float(v) for k, v in self.robot.bus.sync_read("Present_Temperature", normalize=False, num_retry=READ_RETRIES).items()}

    def check_current(self) -> None:
        if self.limits.max_current <= 0:
            return
        raw = self.robot.bus.sync_read("Present_Current", normalize=False, num_retry=READ_RETRIES)
        current = {k: current_magnitude(v) for k, v in raw.items()}
        worst = max(current, key=current.get)
        self.peak_current = max(self.peak_current, current[worst])
        if current[worst] <= self.limits.max_current:
            self._over_current = 0
            return
        self._over_current += 1
        if self._over_current >= self.limits.current_ticks:
            self._over_current = 0
            raise self._hold(f"current on {worst}: {current[worst]} counts > {self.limits.max_current}")

    def check_heat(self) -> float:
        temps = self.temperatures()
        hottest = max(temps.values())
        if hottest >= self.limits.hot_c:
            self._hold("overheat")
            raise Overheat(f"motor at {hottest:.0f} C (limit {self.limits.hot_c:.0f} C)")
        return hottest

    # commanding
    def stop(self) -> None:
        """From another thread (the keyboard): the next command trips."""
        self._stop.set()

    def _hold(self, reason: str) -> SafetyStop:
        if self.dry_run:
            self._locked = reason
            self._history.clear()
            return SafetyStop(reason)
        try:
            present = self.positions()
            self.robot.bus.sync_write("Goal_Position", dict(zip(JOINTS, present.tolist())))
        except Exception:
            pass
        self._locked = reason
        self._history.clear()
        return SafetyStop(reason)

    def gate(self, goal: np.ndarray) -> np.ndarray:
        """The goal actually sent: clipped to the box, or ``SafetyStop``."""
        lim = self.limits
        if self._stop.is_set():
            self._stop.clear()
            raise self._hold("operator stop")
        if self._locked:
            raise SafetyStop(f"locked after {self._locked}")
        goal = np.asarray(goal, dtype=np.float64)
        if goal.shape != (len(JOINTS),) or not np.isfinite(goal).all():
            raise self._hold(f"bad goal {goal}")
        clipped = lim.clip(goal)
        if not np.allclose(clipped, goal):
            self.stats["box_clips"] += 1
        goal = clipped
        # Rates over the time that really passed between commands (never less than
        # one control tick): the arm pauses while the policy plans the next chunk,
        # and a jump after a pause is spread over the pause.
        now = time.monotonic()
        if self._history:
            last_goal, last_time = self._history[-1]
            dt = max(now - last_time, self.dt)
            velocity = (goal - last_goal) / dt
            i = int(np.argmax(np.abs(velocity)))
            if abs(velocity[i]) > lim.max_velocity_deg_s:
                raise self._hold(
                    f"speed on {JOINTS[i]}: {abs(velocity[i]):.0f} deg/s > {lim.max_velocity_deg_s:.0f}"
                )
            if len(self._history) >= 2:
                before_goal, before_time = self._history[-2]
                dt_before = max(last_time - before_time, self.dt)
                accel = (velocity - (last_goal - before_goal) / dt_before) / ((dt + dt_before) / 2)
                i = int(np.argmax(np.abs(accel)))
                if abs(accel[i]) > lim.max_accel_deg_s2:
                    raise self._hold(
                        f"acceleration on {JOINTS[i]}: {abs(accel[i]):.0f} deg/s^2 > {lim.max_accel_deg_s2:.0f}"
                    )
        self._history = [*self._history[-1:], (goal, now)]
        return goal

    def send(self, goal: np.ndarray) -> np.ndarray:
        present = self.positions()
        if self._start is None:
            self._start = present
        st = self.stats
        st["asked_ahead_max_deg"] = max(st["asked_ahead_max_deg"], float(np.abs(np.asarray(goal) - present).max()))
        st["moved_max_deg"] = max(st["moved_max_deg"], float(np.abs(present - self._start).max()))
        goal = self.gate(goal)
        # Step-size limit relative to where the arm is (LeRobot's max_relative_target).
        step = np.clip(goal - present, -self.limits.max_step_deg, self.limits.max_step_deg)
        if not np.allclose(step, goal - present):
            st["step_clips"] += 1
        sent = present + step
        st["sent_ahead_max_deg"] = max(st["sent_ahead_max_deg"], float(np.abs(step).max()))
        if self.dry_run:
            self.dry_commands.append(sent)
            return sent
        self.robot.send_action({f"{j}.pos": float(v) for j, v in zip(JOINTS, sent)})
        return sent

    # reset
    def move_to(self, target: np.ndarray, speed_deg_s: float = 30.0, rate: int = 50) -> None:
        """Glide to ``target`` at a bounded speed; the gate does not apply."""
        if self.dry_run:
            return
        target = self.limits.clip(np.asarray(target, dtype=np.float64))
        start = self.positions()
        steps = max(1, int(np.max(np.abs(target - start)) / speed_deg_s * rate))
        for k in range(1, steps + 1):
            point = start + (target - start) * k / steps
            self.robot.bus.sync_write("Goal_Position", dict(zip(JOINTS, point.tolist())))
            time.sleep(1.0 / rate)
        time.sleep(0.3)

    def unlock(self) -> None:
        self._locked = None
        self._stop.clear()
        self._history.clear()
        self._over_current = 0
        self.peak_current = 0
        self.reset_stats()


def setup(robot: Any, path: Path = LIMITS_FILE, margin_deg: float = 2.0) -> dict:
    """Record the joint box from a hand sweep with torque off, then the home pose."""
    robot.bus.disable_torque()
    input(
        "Torque is off. Press Enter, then move the arm by hand through all the space it "
        "may use while pressing the buttons. Press Enter again when done. "
    )
    low = {j: float("inf") for j in JOINTS}
    high = {j: float("-inf") for j in JOINTS}
    while not select.select([sys.stdin], [], [], 0.05)[0]:
        for j, v in robot.bus.sync_read("Present_Position").items():
            low[j], high[j] = min(low[j], v), max(high[j], v)
        print("\r" + " ".join(f"{j[:8]} {low[j]:6.1f}..{high[j]:6.1f}" for j in JOINTS), end="", flush=True)
    sys.stdin.readline()
    input("\nHold the arm in the pose every episode starts from, then press Enter. ")
    home = {j: float(v) for j, v in robot.bus.sync_read("Present_Position").items()}
    limits = {
        "min": {j: low[j] - margin_deg for j in JOINTS},
        "max": {j: high[j] + margin_deg for j in JOINTS},
        "home": home,
    }
    outside = [j for j in JOINTS if not limits["min"][j] <= home[j] <= limits["max"][j]]
    if outside:
        raise SystemExit(f"the home pose is outside the swept box on {outside}; run setup again")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(limits, indent=2) + "\n")
    return limits
