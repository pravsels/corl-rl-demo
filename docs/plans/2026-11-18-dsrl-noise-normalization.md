# DSRL Noise Normalization Plan

**Goal:** Keep SAC-selected diffusion noise near the standard Gaussian region
seen during base-policy training.

**Assumptions:** 16-step diffusion horizon, 6 joints, and 4 SAC-selected noise
rows, so the full noise is 96-dimensional and the SAC output is 24-dimensional.

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

## Evaluation

Implement each option behind a configuration flag. Compare base success rate,
online learning rate, safety stops, noise norms, and row correlations against
the current repeated-row method before selecting a default.
