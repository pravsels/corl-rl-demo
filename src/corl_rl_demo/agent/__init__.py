"""Online RL on the button rig (step 6 of the plan).

Two methods share one loop (``run.py``), one SAC learner (``sac.py``), the
safety gate (``safety.py``) and the button reward (``episode.py``):

- ``residual-act``: the frozen ACT policy acts; SAC adds a small bounded
  correction to its actions, from ACT's own encoder features, the joints and
  ACT's action (the plan's design choice 2).
- ``dsrl-flow``: a small flow-matching policy (``flow.py``) is the base, and
  SAC picks the initial noise its flow starts from (DSRL, Wagenmaker et al.
  2025). Every executed action is one the base policy produced.

Both start exactly at the base policy and run on the laptop's CPU while the arm
moves.
"""
