# corl-rl-demo

A live demo for CoRL 2026: an SO-101 arm presses a button over and over and
gets better at it while you watch. A small frozen policy trained on a few demos
does the pressing; a residual RL learner improves it online. Everything runs on
one laptop, with no network.

The plan is in [`docs/plans/2026-11-10-corl-rl-demo.md`](docs/plans/2026-11-10-corl-rl-demo.md).

## Setup

Needs Python 3.12 and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest
```

## Button module

The four-button module is the reward signal. Find its serial port
(`ls /dev/cu.usbserial*` on macOS, `ls /dev/ttyUSB*` on Linux), then:

```bash
uv run button-monitor --port /dev/cu.usbserial-110
```

It prints each press and release with its arrival time, and on Ctrl-C the
number of presses and the shortest hold per pin. On our module D2 is green, D3
red, D4 yellow and D5 white; if yours differs, change `COLOURS` and `GREEN` in
`src/corl_rl_demo/buttons.py`.

## Rig check

```bash
uv run rig-check
```

It checks that the button module, the arm's six motors and the top camera all
answer, and saves a camera frame to `outputs/rig_check_top.png` so you can see
what the policy will see. It only reads: it never enables torque or moves the
arm. Ports and the camera index are set at the top of `src/corl_rl_demo/rig.py`
and can be overridden with flags (`uv run rig-check --help`).

## Calibration

Each arm is calibrated once per machine. LeRobot asks you to hold the arm in the
middle of its range, then sweep every joint through its full range:

```bash
uv run lerobot-calibrate --robot.type=so101_follower \
    --robot.port=/dev/cu.usbmodem5AB01584891 --robot.id=demo_follower
```
