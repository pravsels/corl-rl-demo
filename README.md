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

It checks that the button module, the follower's five motors, the leader (if it
is plugged in) and both cameras all answer. It saves a frame from each camera,
`outputs/rig_check_top.png` and `outputs/rig_check_wrist.png`, so you can see
what the policy will see. Look at them: macOS renumbers cameras when one comes
or goes, and the two views are easy to swap. It only reads: it never enables
torque or moves an arm. Ports and camera indices are set at the top of
`src/corl_rl_demo/rig.py` and can be overridden with flags
(`uv run rig-check --help`). The button module's port is found by itself.

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
LEADER=(--teleop.type=so101_5motor_leader --teleop.port=/dev/cu.usbmodem5AE60838491 --teleop.id=demo_leader)
CAMERAS='--robot.cameras={top: {type: opencv, index_or_path: 1, width: 320, height: 240, fps: 30}, wrist: {type: opencv, index_or_path: 0, width: 320, height: 240, fps: 30}}'
```

The camera indices are the ones `rig-check` confirmed. `uv run lerobot-find-port`
finds an arm's port.

### 1. Calibrate

Once per arm. LeRobot asks you to hold the arm in the middle of its range, then
sweep the four joints it lists through their full range. Push each one to both
hard stops: a joint's angle is measured from the middle of what you swept, so a
short sweep leaves the follower tens of degrees off the leader.

```bash
uv run lerobot-5motor calibrate "${FOLLOWER[@]}"
uv run lerobot-5motor calibrate "${LEADER[@]}"
```

The results are saved in this repo, as `calibration/follower/demo_follower.json`
and `calibration/leader/demo_leader.json`. Commit them: another machine that
pulls the repo then drives the same two arms without calibrating again. The
files belong to these arms. With different arms, or after recalibrating, the
servos and the files disagree and LeRobot asks which to keep.

### 2. Teleoperate

```bash
uv run lerobot-5motor teleoperate "${FOLLOWER[@]}" "${LEADER[@]}" "$CAMERAS" --display_data=true
```

### 3. Record demos to the Hugging Face Hub

Log in once with `uv run hf auth login`, and set `HF_USER` to your Hub name.
The dataset is uploaded when recording ends.

```bash
uv run lerobot-5motor record "${FOLLOWER[@]}" "${LEADER[@]}" "$CAMERAS" \
    --dataset.repo_id=${HF_USER}/corl-press-green \
    --dataset.single_task="Press the green button" \
    --dataset.num_episodes=50 --dataset.episode_time_s=15 --dataset.reset_time_s=5
```

A run that stops before its first episode is saved leaves an empty dataset
folder behind, and the next run refuses to reuse it. Delete it first:
`rm -r ~/.cache/huggingface/lerobot/${HF_USER}/corl-press-green`.

Check the recording by replaying an episode on the arm:

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
uv run lerobot-5motor record "${FOLLOWER[@]}" "$CAMERAS" \
    --policy.path=${HF_USER}/act-corl-press-green \
    --dataset.repo_id=${HF_USER}/eval_corl-press-green \
    --dataset.single_task="Press the green button" \
    --dataset.num_episodes=20 --dataset.episode_time_s=15 --dataset.reset_time_s=5
```

### 6. Online RL

Not built yet (step 6 of the plan). It will drive the same
`SO101FiveMotorFollower` class from Python.
