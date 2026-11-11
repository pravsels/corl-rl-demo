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

It checks that the button module, the arm's five motors and the top camera all
answer, and saves a camera frame to `outputs/rig_check_top.png` so you can see
what the policy will see. It only reads: it never enables torque or moves the
arm. Ports and the camera index are set at the top of `src/corl_rl_demo/rig.py`
and can be overridden with flags (`uv run rig-check --help`).

## A five-motor arm

Our arm has no gripper motor: it now drives a joint whose own motor failed.
Pressing a button needs no gripper, so the jaw is fixed shut and the arm runs on
its five joints, motor IDs 1 to 5. The gripper is not recorded as a constant; it
is absent. States and actions are five numbers, in datasets and in policies.

LeRobot's commands expect six motors. `lerobot-5motor` runs `lerobot-calibrate`,
`-teleoperate`, `-record` and `-replay` with two more arm types,
`so101_5motor_follower` and `so101_5motor_leader`, and passes every other option
through. Both are defined in `src/corl_rl_demo/five_motor.py`. The leader type
reads five joints from an ordinary six-motor leader and ignores its gripper.

The commands below share these options:

```bash
FOLLOWER=(--robot.type=so101_5motor_follower --robot.port=/dev/cu.usbmodem5AB01584891 --robot.id=demo_follower)
LEADER=(--teleop.type=so101_5motor_leader --teleop.port=/dev/cu.usbmodemXXXX --teleop.id=demo_leader)
TOP='--robot.cameras={top: {type: opencv, index_or_path: 0, width: 320, height: 240, fps: 30}}'
```

`uv run lerobot-find-port` finds the leader's port.

### 1. Calibrate

Once per arm per machine. LeRobot asks you to hold the arm in the middle of its
range, then sweep every joint through its full range:

```bash
uv run lerobot-5motor calibrate "${FOLLOWER[@]}"
uv run lerobot-5motor calibrate "${LEADER[@]}"
```

### 2. Teleoperate

```bash
uv run lerobot-5motor teleoperate "${FOLLOWER[@]}" "${LEADER[@]}" "$TOP" --display_data=true
```

### 3. Record demos to the Hugging Face Hub

Log in once with `uv run hf auth login`, and set `HF_USER` to your Hub name.
The dataset is uploaded when recording ends.

```bash
uv run lerobot-5motor record "${FOLLOWER[@]}" "${LEADER[@]}" "$TOP" \
    --dataset.repo_id=${HF_USER}/corl-press-green \
    --dataset.single_task="Press the green button" \
    --dataset.num_episodes=50 --dataset.episode_time_s=15 --dataset.reset_time_s=5
```

Check it by replaying an episode on the arm:

```bash
uv run lerobot-5motor replay "${FOLLOWER[@]}" \
    --dataset.repo_id=${HF_USER}/corl-press-green --dataset.episode=0
```

### 4. Train ACT

On any machine with a GPU and `lerobot==0.5.1`. Training reads only the
dataset, so the plain LeRobot command works:

```bash
lerobot-train --dataset.repo_id=${HF_USER}/corl-press-green \
    --policy.type=act --policy.device=cuda \
    --policy.repo_id=${HF_USER}/act-corl-press-green \
    --output_dir=outputs/train/act-corl-press-green --job_name=act-corl-press-green \
    --wandb.enable=false
```

### 5. Run the policy on the arm

Leave out the leader and name the policy. `--policy.path` can also be a folder
on disk, which is what the venue needs. LeRobot wants the dataset name to start
with `eval_`.

```bash
uv run lerobot-5motor record "${FOLLOWER[@]}" "$TOP" \
    --policy.path=${HF_USER}/act-corl-press-green \
    --dataset.repo_id=${HF_USER}/eval_corl-press-green \
    --dataset.single_task="Press the green button" \
    --dataset.num_episodes=20 --dataset.episode_time_s=15 --dataset.reset_time_s=5
```

### 6. Online RL

Not built yet (step 6 of the plan). It will drive the same
`SO101FiveMotorFollower` class from Python.
