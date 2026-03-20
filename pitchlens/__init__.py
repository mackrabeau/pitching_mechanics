"""Pitching mechanics — OBP landmark tracking + MuJoCo simulation.

Modules:
  obp_fullsig            Load OBP full-signal landmark data.
  site_ik                Site-based damped least-squares IK solver.
  build_pitcher_fullbody Generate a full-body pitcher MJCF from OBP landmarks.
  replay_pitcher_fullbody Replay a pitch delivery in MuJoCo (kinematic or physics-tracked).
"""
