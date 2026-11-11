# CoRL workshop demo: an SO-101 that gets better at pressing a button

Date: 2026-11-10. Status: draft, steps 1-2 started.

## Goal

An SO-101 arm presses the green button on a four-button module, over and over,
for the length of the workshop. It learns while it runs, and a screen shows its
success rate and time-to-press improving. Everything runs on one MacBook Air
(M3, 16 GB) with no network.

## Rig

- SO-101 follower arm, plus a leader arm for recording demos.
- Top camera on a pole. Wrist camera once it is wired.
- Four-button module on USB serial. It reports which buttons are down.
- Arm, module and camera pole fixed to one board, so the geometry is the same
  in the hotel and at the venue.

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
   Logs and plots stay local.
8. **Pins.** Python 3.12, `lerobot[feetech]==0.5.1`.

## Steps

Each step ends with a check. Stop for review after each one.

1. **Repo and environment.** Check: `uv sync` and `uv run pytest` pass on the Mac.
2. **Reward reader.** `button-monitor` prints each press. Check: every button
   shows up, a quick tap is not missed, and we have written down which pin is
   which colour.
3. **Rig.** Mount everything on the board, calibrate the arms, define the start
   pose. Check: teleoperation works and every camera we plan to use shows up.
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
- **Quick taps missed.** The module reports once a second for certain; whether
  it also reports on every change is checked in step 2.

## Open questions

- Wrist camera, or top only? Must be settled before step 4.
- When does the workshop start? That sets how many of steps 6-8 we can afford.
- Where do we train ACT?
