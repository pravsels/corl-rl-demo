#!/usr/bin/env bash
# Runs one flow checkpoint on the arm, base policy only (no learning), as `rl-agent
# run` with every safety limit on. The flow's noise is drawn once per episode
# (NOISE=random draws it fresh for every plan). Episodes time out after 45 s
# (TIMEOUT=<seconds> changes it).
#
#   scripts/try_flow.sh <checkpoint> [episodes, default 3] [extra rl-agent run flags]
#
# <checkpoint> is a folder, or a total step count of the augmented run:
#   2000 -> outputs/flow-aug,  3000..8000 -> outputs/flow-aug8k/step_<N-2000>
# Space stops the arm, q quits. Logs go to outputs/try/<checkpoint name>/; each episode's
# plans (time, planning ms, joints, where the plan ends) are in its "trace".
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
CHECKPOINT=${1:?usage: scripts/try_flow.sh <checkpoint folder or step> [episodes] [flags]}
EPISODES=${2:-3}
shift $(( $# >= 2 ? 2 : 1 ))
if [[ "$CHECKPOINT" =~ ^[0-9]+$ ]]; then
  if [ "$CHECKPOINT" -eq 2000 ]; then CHECKPOINT=outputs/flow-aug
  else CHECKPOINT=outputs/flow-aug8k/step_$((CHECKPOINT - 2000)); fi
fi
[ -f "$CHECKPOINT/norm_stats.json" ] || { echo "no checkpoint at $CHECKPOINT (not saved yet?)" >&2; exit 1; }
NAME=$(echo "$CHECKPOINT" | tr '/' '_')
echo "checkpoint $CHECKPOINT, $EPISODES episodes"
exec .venv/bin/rl-agent run --method dsrl-flow --policy "$CHECKPOINT" \
  --arm-port /dev/cu.wchusbserial5AB01584891 --button-port /dev/cu.wchusbserial1120 \
  --top-camera 1 --wrist-camera 0 --no-learn --base-noise "${NOISE:-episode}" \
  --steps-per-decision 30 --episodes "$EPISODES" --episode-timeout "${TIMEOUT:-45}" \
  --run-dir "outputs/try/$NAME" "$@"
