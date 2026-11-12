#!/usr/bin/env bash
# Runs one Diffusion Policy checkpoint on the arm, base policy only (no learning),
# as `rl-agent run` with every safety limit on. It draws fresh random noise for
# every plan, as the policy does natively (NOISE=episode keeps one per episode).
#
#   scripts/try_diffusion.sh <step> [episodes, default 3] [extra rl-agent run flags]
#
# <step> is a saved checkpoint of outputs/train/diffusion-corl-press-green
# (1000, 2000, ...), or a pretrained_model folder. Episodes time out after 45 s
# (TIMEOUT=<seconds>). Space stops the arm, q quits. Logs: outputs/try/diffusion-<step>/.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
STEP=${1:?usage: scripts/try_diffusion.sh <step or folder> [episodes] [flags]}
EPISODES=${2:-3}
shift $(( $# >= 2 ? 2 : 1 ))
if [[ "$STEP" =~ ^[0-9]+$ ]]; then
  POLICY=outputs/train/diffusion-corl-press-green/checkpoints/$(printf %06d "$STEP")/pretrained_model
else
  POLICY=$STEP
fi
[ -f "$POLICY/config.json" ] || { echo "no checkpoint at $POLICY (not saved yet?)" >&2; exit 1; }
echo "checkpoint $POLICY, $EPISODES episodes"
exec .venv/bin/rl-agent run --method dsrl-diffusion --policy "$POLICY" \
  --arm-port /dev/cu.wchusbserial5AB01584891 --button-port /dev/cu.wchusbserial1120 \
  --top-camera 1 --wrist-camera 0 --no-learn --base-noise "${NOISE:-random}" \
  --episodes "$EPISODES" --episode-timeout "${TIMEOUT:-45}" \
  --run-dir "outputs/try/diffusion-$(basename "$(dirname "$POLICY")")" "$@"
