# CoRL workshop demo: an SO-101 that gets better at pressing a button

Date: 2026-11-10. Status: steps 1-2 done, step 3 started.

## Goal

An SO-101 arm presses the green button on a four-button module, over and over,
for the length of the workshop. It learns while it runs, and a screen shows its
success rate and time-to-press improving. Everything runs on one MacBook Air
(M3, 16 GB) with no network.

## Rig

- SO-101 follower arm, plus a leader arm for recording demos. The follower runs
  on five motors: its gripper motor now drives a joint whose own motor failed,
  and the jaw is fixed shut.
- Top camera on a pole: OpenCV index 0 on the Mac, recorded at 320x240, 30 fps.
  That mode is 4:3 and crops the sides of the camera's 16:9 view, so aim the
  camera with `rig-check`'s saved frame, not with a 16:9 preview. Wrist camera
  once it is wired.
- Four-button module on USB serial. It reports which buttons are down, once a
  second and at once on any change. D2 is green, D3 red, D4 yellow, D5 white.
- Arm, module and camera pole fixed to one board, so the geometry is the same
  in the hotel and at the venue.
- Arm calibration files are committed under `calibration/`, so any machine with
  the repo drives the same arms without calibrating again.

## Design choices

1. **Base policy: ACT, frozen.** Trained on 40-50 demos recorded on this rig, at
   low resolution (about 320x240). We want a base that succeeds 50-70% of the
   time, so there is something left to learn. Pick an earlier checkpoint if the
   last one is too good.
2. **Learner: residual SAC.** It sees ACT's image features, the joint positions
   and ACT's action, and adds a small bounded correction to that action. Small
   MLPs, trained on the CPU while the arm runs.
3. **Reward comes from the module.** +1 when green is the first button pressed,
   and the episode ends. Any other button first, or 15 s without a press, is a
   failure. A small cost per step rewards pressing sooner.
4. **No human reset.** The button springs back. The arm returns to a start pose
   drawn from a small range.
5. **Fallback base: a fixed reference trajectory.** Built from the demos, no
   camera. Same learner, selected by a flag, for when venue lighting breaks ACT.
6. **The camera set is fixed before recording.** ACT's inputs are whatever the
   demos contain. Demos recorded with the top camera only give a top-only
   policy; adding the wrist camera later means recording again.
7. **Offline at the venue.** Weights and dependencies are cached beforehand.
   Logs and plots stay local. Checked so far with downloads blocked (not yet in
   airplane mode, which is step 8): ACT builds from the cached ResNet-18
   weights and a 320x240 forward pass takes about 10 ms on the Mac's GPU;
   recording and reading back a video dataset works.
8. **Pins.** Python 3.12, `lerobot[feetech]==0.5.1`.
9. **Five joints, no gripper.** The gripper is left out, not held at a constant:
   states and actions are five numbers in the demos, in ACT and in the learner.
   `lerobot-5motor` runs LeRobot's commands with five-motor arm types. Checked
   with made-up data and no arm: a five-joint dataset records, ACT trains on it,
   and the checkpoint returns five-number actions.

## Steps

Each step ends with a check. Stop for review after each one.

1. **Repo and environment.** Check: `uv sync` and `uv run pytest` pass on the Mac.
2. **Reward reader.** `button-monitor` prints each press. Check: every button
   shows up, a quick tap is not missed, and we have written down which pin is
   which colour.
3. **Rig.** Mount everything on the board, calibrate the arms, define the start
   pose. Check: `rig-check` passes, teleoperation works and every camera we
   plan to use shows up.
4. **Demos.** Record 40-50, with varied start poses and a few with the module
   shifted 1-2 cm. Check: replaying one episode presses green.
5. **Base policy.** Train ACT on a cloud GPU. Run 20 rollouts per candidate
   checkpoint on the rig. Check: the chosen one lands at 50-70% and runs on the
   Mac, offline, at the control rate.
6. **RL loop.** Residual SAC with joint, step-size and torque limits. Check: in
   a 30 minute run success or time-to-press improves, with no limit violations.
7. **Display.** Live curves, an early-versus-now clip pair, and a control that
   restarts learning from the base policy.
8. **Dry run.** One hour in airplane mode. Check: no crashes, servo
   temperatures and disk use stay in bounds.

## Risks

- **Venue lighting breaks ACT.** Switch to the fallback base (choice 5).
- **Servos overheat over a full day.** Read temperatures and pause between
  episodes when they climb.
- **The module slides when pressed.** The board fixes this (step 3).
- **The reused gripper motor is weak as a joint.** LeRobot left it with half
  torque and a low overload limit. The follower class writes the other joints'
  limits to every joint each time it connects; read them back after the first
  calibration.
- **A servo's reply is late now and then while the camera streams.** Usually
  1 ms, sometimes 40-80 ms, and LeRobot gives up after about 50 ms. That stopped
  `lerobot-record` while connecting. The arm classes ask again up to three
  times. Count the late replies in the RL loop and in the dry run.
- **Quick taps missed.** Checked in step 2: the module reports on change, and
  finger taps of 0.12-0.16 s all registered (one per button). It debounces for
  50 ms, so the arm must hold the button down longer than that.

## Open questions

- Wrist camera, or top only? Must be settled before step 4.
- When does the workshop start? That sets how many of steps 6-8 we can afford.
- Where do we train ACT?
