"""The reward of the plan's design choice 3, from the button module's latch.

+1 when green is the first button pressed, and the episode ends. Another button
first, or ``timeout_s`` without a press, is a failure. Every decision costs
``step_cost``, so pressing sooner is worth more. A safety stop ends the episode
with ``-safety_penalty``.
"""

from __future__ import annotations

from dataclasses import dataclass

from corl_rl_demo.buttons import GREEN


@dataclass
class RewardConfig:
    timeout_s: float = 15.0
    step_cost: float = 0.01
    safety_penalty: float = 1.0
    target: str = GREEN


@dataclass(frozen=True)
class Outcome:
    reward: float
    done: bool
    label: str  # "running", "success", "wrong_button", "timeout", "safety"


def score(presses: tuple[str, ...], elapsed_s: float, config: RewardConfig) -> Outcome:
    """The reward of one decision, given the presses latched since the episode
    began (first first) and the time since it began."""
    if presses:
        if presses[0] == config.target:
            return Outcome(1.0, True, "success")
        return Outcome(-config.step_cost, True, "wrong_button")
    if elapsed_s >= config.timeout_s:
        return Outcome(-config.step_cost, True, "timeout")
    return Outcome(-config.step_cost, False, "running")


def safety_outcome(config: RewardConfig) -> Outcome:
    return Outcome(-config.safety_penalty, True, "safety")
