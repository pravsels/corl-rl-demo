# DSRL Noise Normalization Plan

**Goal:** Keep SAC-selected diffusion noise near the standard Gaussian region
seen during base-policy training.

**Assumptions:** 16-step diffusion horizon and 6 joints, so the full noise is
96-dimensional. Options 1-2 use 4 SAC-selected rows (24 outputs); option 3 uses
one row (6 outputs).

## Problem

Repeating SAC's fourth row over the remaining horizon creates a `(16, 6)` input,
but its norm can be atypical for `N(0, I)`. The repeated rows also introduce
perfect temporal correlations absent from independent Gaussian training noise.

## Option 1: Renormalize the repeated noise

```python
# SAC selects 4 noise rows, each containing one value per joint.
rows = sac_output.reshape(4, 6) * 1.5

# Repeat the final row to fill the 16-step diffusion horizon.
noise = np.concatenate((rows, np.repeat(rows[-1:], 12, axis=0)))

# Project the 96D noise onto the typical Gaussian radius.
noise *= np.sqrt(96) / (np.linalg.norm(noise) + 1e-8)
```

This is the smallest change and corrects radial shift, but not repeated-row
correlations.

## Option 2: Steer full Gaussian noise

```python
# Create once: 24 fixed orthogonal directions in the 96D noise space.
projection, _ = np.linalg.qr(np.random.randn(96, 24))

# Start from ordinary independent Gaussian diffusion noise.
epsilon = np.random.randn(96)

# Map SAC's 24 outputs into a 96D steering offset.
steered = epsilon + projection @ sac_output

# Preserve the original Gaussian sample's norm.
steered *= np.linalg.norm(epsilon) / (np.linalg.norm(steered) + 1e-8)
noise = steered.reshape(16, 6)
```

This preserves more of the Gaussian structure, but adds stochastic base noise
and a fixed projection that must be seeded and saved with the learner.

## Option 3: Repeat and apply a fixed permutation

This is Alexander Soare's suggested approach, translated from OpenPI's
50-by-32 noise to a 16-step, 6-joint policy.

```python
# Create once and reuse for training, evaluation, and resumed runs.
permutation = np.random.default_rng(0).permutation(16 * 6)

# SAC selects one 6D noise row.
sac_output = sac.act(obs)

# Repeat its values to fill the required 96D noise input.
noise = np.repeat(sac_output[None, :], 16, axis=0)

# Apply the same shuffle every time, avoiding identical consecutive rows.
noise = noise.flatten()[permutation].reshape(16, 6)

# Optional: project onto the typical 96D Gaussian radius.
noise *= np.sqrt(96) / (np.linalg.norm(noise) + 1e-8)
```

The fixed permutation gives SAC a stationary mapping while spreading each of
its six outputs across joints and timesteps. It can remove instability caused
by locally identical rows and is cheap to learn.

This is a stability heuristic rather than a complete in-distribution fix. The
matrix still contains only six unique values, each repeated 16 times, so it
lies in a 6D subspace of the 96D noise space. Perfect correlations remain; the
permutation only relocates them. Optional norm projection corrects radial shift
but not those correlations. OpenPI used 32 SAC outputs, so the 6-output SO-101
version is also substantially more restrictive.

## Evaluation

Implement each option behind a configuration flag. Compare base success rate,
online learning rate, safety stops, noise norms, and row correlations against
the current repeated-last-row method before selecting a default. Save any fixed
permutation or projection in checkpoints so resumed and evaluation runs retain
the mapping learned by SAC.
