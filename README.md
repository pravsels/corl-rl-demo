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

Once per arm. LeRobot asks you to hold the arm in the middle of its range, then
sweep every joint through its full range:

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

`rl-agent` learns on the rig while it runs, on the laptop. Three methods, one
loop (`src/corl_rl_demo/agent/`):

- `dsrl-diffusion`: LeRobot's Diffusion Policy, frozen, with SAC choosing the
  noise its denoiser starts from (DSRL).
- `dsrl-flow`: the same with a small flow policy (`rl-agent flow-train`).
- `residual-act`: ACT, frozen, plus a SAC correction of up to 5° per joint.

```bash
uv run rl-agent limits --dataset pravsels/corl-press-green   # joint box and start pose from the demos
uv run lerobot-train --dataset.repo_id=pravsels/corl-press-green --dataset.video_backend=pyav \
    --dataset.image_transforms.enable=true --policy.type=diffusion --policy.device=mps \
    --policy.resize_shape='[120,160]' --policy.crop_ratio=0.9 --policy.down_dims='[256,512,1024]' \
    --policy.noise_scheduler_type=DDIM --policy.num_inference_steps=10 --policy.push_to_hub=false \
    --output_dir=outputs/train/diffusion-corl-press-green --steps=10000 --save_freq=1000 --batch_size=16
scripts/try_diffusion.sh 4000 3 --speed-scale 0.5   # one checkpoint on the arm, no learning
uv run rl-agent run --method dsrl-diffusion --policy <pretrained_model dir> --exploration 3
```

The base policy alone runs with `--no-learn --base-noise random`
(`scripts/try_diffusion.sh`, `scripts/try_flow.sh`); `--dry-run` keeps torque
off and commands nothing. `--speed-scale 0.5` follows each plan at half speed.

Safety, from the outside in: goals are clipped to the demos' joint range plus 3°
and to 10° ahead of the arm; a commanded speed over 150°/s, acceleration over
2500°/s², or current over 280 counts for 3 ticks holds the arm and ends the
episode with a penalty; above 45 °C the run rests between episodes and at 50 °C
it stops; the servos are capped at 70 % torque, 180°/s and a gentler
acceleration (RAM registers, reset by a power cycle). Plans are smoothed over 5
ticks and cross-faded into each other.

Reward follows the plan: +1 when green is pressed first, a failure for another
button or the timeout (`--episode-timeout`, 15 s), a small cost per decision.
Every tenth episode runs SAC's mean action. Episodes, with a per-plan trace, are
logged to `<run dir>/episodes.jsonl`; `--resume` continues a run. Keys: space
stops the arm, `p` pauses, `q` quits.
