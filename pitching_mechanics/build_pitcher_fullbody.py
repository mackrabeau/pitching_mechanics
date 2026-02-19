"""Generate a full-body pitcher MJCF from OBP landmarks.

The model has:
  pelvis (freejoint) → torso (3 DOF)
    → throw arm  (3 shoulder + 1 elbow)
    → glove arm  (3 shoulder + 1 elbow)
  pelvis → rear leg (3 hip + 1 knee + 1 ankle)
  pelvis → lead leg (3 hip + 1 knee + 1 ankle)

All offsets / segment lengths are measured from the OBP landmark data
at a chosen event time.  OBP global frame ≡ MuJoCo world frame
(X = home plate, Y = pitcher's left, Z = up).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pitching_mechanics.obp_fullsig import load_pitch


def _interp3(t: float, t_ref: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return np.array([np.interp(t, t_ref, Y[:, i]) for i in range(3)])


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v * 0.0


def _pelvis_frame(rear_hip: np.ndarray, lead_hip: np.ndarray):
    """Return (midpoint, R_3x3) for the pelvis frame.

    Conventions (for a RHP):
      y = rear_hip → lead_hip (right → left)
      x = forward (≈ toward home plate), orthogonal to y & up
      z = up, orthogonalized
    """
    mid = 0.5 * (rear_hip + lead_hip)
    y = _normalize(lead_hip - rear_hip)
    up = np.array([0.0, 0.0, 1.0])
    x = _normalize(np.cross(y, up))
    z = np.cross(x, y)
    R = np.column_stack([x, y, z])      # world → pelvis columns
    return mid, R


def _fmt(v, n=6):
    """Format a 3-vector for XML."""
    return " ".join(f"{x:.{n}f}" for x in v)


def build_xml(
    *,
    throw_shoulder_off: np.ndarray,
    glove_shoulder_off: np.ndarray,
    rear_hip_off: np.ndarray,
    lead_hip_off: np.ndarray,
    throw_upper_len: float,
    throw_fore_len: float,
    glove_upper_len: float,
    glove_fore_len: float,
    rear_thigh_len: float,
    rear_shin_len: float,
    lead_thigh_len: float,
    lead_shin_len: float,
) -> str:
    # Arm default direction: +x in local frame.
    # Leg default direction: -z in local frame.
    return f"""\
<mujoco model="PitcherFullBody">
  <compiler angle="radian"/>
  <option timestep="0.001" gravity="0 0 -9.81" integrator="Euler"/>

  <default>
    <joint type="hinge" damping="5.0" limited="true"/>
    <position kp="80" kv="8"/>
    <geom type="capsule" density="500"/>
    <site size="0.018" rgba="0 0.8 0.8 1"/>
  </default>

  <worldbody>
    <light pos="1 0 3" dir="0 0 -1" directional="true"/>
    <geom name="floor" type="plane" size="5 5 0.1" rgba="0.25 0.3 0.35 1" condim="3"/>

    <!-- Mocap target for pelvis (kinematic root, invisible) -->
    <body name="pelvis_target" mocap="true" pos="0 0 1">
      <geom type="sphere" size="0.001" contype="0" conaffinity="0" rgba="0 0 0 0"/>
    </body>

    <body name="pelvis" pos="0 0 1">
      <freejoint name="root"/>
      <site name="pelvis_site" pos="0 0 0"/>
      <geom name="pelvis_geom" type="sphere" size="0.10" rgba="0.85 0.75 0.55 1"/>

      <!-- ═══════════ TORSO + ARMS ═══════════ -->
      <body name="torso" pos="0 0 0">
        <joint name="trunk_yaw"   axis="0 0 1" range="-3.14 3.14"/>
        <joint name="trunk_pitch" axis="0 1 0" range="-2.5 2.5"/>
        <joint name="trunk_roll"  axis="1 0 0" range="-2.5 2.5"/>
        <geom name="torso_geom" fromto="0 0 0  {_fmt(throw_shoulder_off)}" size="0.08" rgba="0.75 0.22 0.22 1"/>
        <geom name="torso_geom2" fromto="0 0 0  {_fmt(glove_shoulder_off)}" size="0.08" rgba="0.75 0.22 0.22 1"/>

        <!-- Throwing arm -->
        <body name="throw_upper_arm" pos="{_fmt(throw_shoulder_off)}">
          <site name="throw_shoulder_site" pos="0 0 0"/>
          <joint name="throw_shoulder_yaw"   axis="0 0 1" range="-3.14 3.14"/>
          <joint name="throw_shoulder_pitch" axis="0 1 0" range="-3.14 3.14"/>
          <joint name="throw_shoulder_roll"  axis="1 0 0" range="-3.14 3.14"/>
          <geom name="throw_upper_geom" fromto="0 0 0  {throw_upper_len:.6f} 0 0" size="0.045" rgba="0.85 0.30 0.30 1"/>

          <body name="throw_forearm" pos="{throw_upper_len:.6f} 0 0">
            <site name="throw_elbow_site" pos="0 0 0"/>
            <joint name="throw_elbow_flex" axis="0 1 0" range="-3.14 0.2"/>
            <geom name="throw_fore_geom" fromto="0 0 0  {throw_fore_len:.6f} 0 0" size="0.038" rgba="0.85 0.30 0.30 1"/>

            <body name="throw_hand" pos="{throw_fore_len:.6f} 0 0">
              <site name="throw_hand_site" pos="0 0 0"/>
              <geom name="throw_hand_geom" type="sphere" size="0.04" rgba="0.95 0.85 0.2 1"/>
              <!-- Baseball (visual, attached to hand) -->
              <geom name="ball_geom" type="sphere" size="0.0365" pos="0.05 0 0"
                    rgba="1.0 1.0 1.0 1" mass="0.145" contype="0" conaffinity="0"/>
              <site name="ball_site" pos="0.05 0 0"/>
            </body>
          </body>
        </body>

        <!-- Glove arm -->
        <body name="glove_upper_arm" pos="{_fmt(glove_shoulder_off)}">
          <site name="glove_shoulder_site" pos="0 0 0"/>
          <joint name="glove_shoulder_yaw"   axis="0 0 1" range="-3.14 3.14"/>
          <joint name="glove_shoulder_pitch" axis="0 1 0" range="-3.14 3.14"/>
          <joint name="glove_shoulder_roll"  axis="1 0 0" range="-3.14 3.14"/>
          <geom name="glove_upper_geom" fromto="0 0 0  {glove_upper_len:.6f} 0 0" size="0.045" rgba="0.30 0.40 0.85 1"/>

          <body name="glove_forearm" pos="{glove_upper_len:.6f} 0 0">
            <site name="glove_elbow_site" pos="0 0 0"/>
            <joint name="glove_elbow_flex" axis="0 1 0" range="-3.14 0.2"/>
            <geom name="glove_fore_geom" fromto="0 0 0  {glove_fore_len:.6f} 0 0" size="0.038" rgba="0.30 0.40 0.85 1"/>

            <body name="glove_hand" pos="{glove_fore_len:.6f} 0 0">
              <site name="glove_hand_site" pos="0 0 0"/>
              <geom name="glove_hand_geom" type="sphere" size="0.04" rgba="0.30 0.40 0.85 1"/>
            </body>
          </body>
        </body>
      </body>

      <!-- ═══════════ REAR LEG ═══════════ -->
      <body name="rear_thigh" pos="{_fmt(rear_hip_off)}">
        <site name="rear_hip_site" pos="0 0 0"/>
        <joint name="rear_hip_yaw"   axis="0 0 1" range="-3.14 3.14"/>
        <joint name="rear_hip_pitch" axis="0 1 0" range="-3.14 3.14"/>
        <joint name="rear_hip_roll"  axis="1 0 0" range="-3.14 3.14"/>
        <geom name="rear_thigh_geom" fromto="0 0 0  0 0 {-rear_thigh_len:.6f}" size="0.06" rgba="0.25 0.6 0.35 1"/>

        <body name="rear_shin" pos="0 0 {-rear_thigh_len:.6f}">
          <site name="rear_knee_site" pos="0 0 0"/>
          <joint name="rear_knee_flex" axis="0 1 0" range="-0.1 3.14"/>
          <geom name="rear_shin_geom" fromto="0 0 0  0 0 {-rear_shin_len:.6f}" size="0.05" rgba="0.25 0.6 0.35 1"/>

          <body name="rear_foot" pos="0 0 {-rear_shin_len:.6f}">
            <site name="rear_ankle_site" pos="0 0 0"/>
            <joint name="rear_ankle_flex" axis="0 1 0" range="-1.5 1.5"/>
            <geom name="rear_foot_geom" fromto="0 0 0  0.15 0 0" size="0.04" rgba="0.25 0.6 0.35 1"/>
          </body>
        </body>
      </body>

      <!-- ═══════════ LEAD LEG ═══════════ -->
      <body name="lead_thigh" pos="{_fmt(lead_hip_off)}">
        <site name="lead_hip_site" pos="0 0 0"/>
        <joint name="lead_hip_yaw"   axis="0 0 1" range="-3.14 3.14"/>
        <joint name="lead_hip_pitch" axis="0 1 0" range="-3.14 3.14"/>
        <joint name="lead_hip_roll"  axis="1 0 0" range="-3.14 3.14"/>
        <geom name="lead_thigh_geom" fromto="0 0 0  0 0 {-lead_thigh_len:.6f}" size="0.06" rgba="0.20 0.55 0.30 1"/>

        <body name="lead_shin" pos="0 0 {-lead_thigh_len:.6f}">
          <site name="lead_knee_site" pos="0 0 0"/>
          <joint name="lead_knee_flex" axis="0 1 0" range="-0.1 3.14"/>
          <geom name="lead_shin_geom" fromto="0 0 0  0 0 {-lead_shin_len:.6f}" size="0.05" rgba="0.20 0.55 0.30 1"/>

          <body name="lead_foot" pos="0 0 {-lead_shin_len:.6f}">
            <site name="lead_ankle_site" pos="0 0 0"/>
            <joint name="lead_ankle_flex" axis="0 1 0" range="-1.5 1.5"/>
            <geom name="lead_foot_geom" fromto="0 0 0  0.15 0 0" size="0.04" rgba="0.20 0.55 0.30 1"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>

  <actuator>
    <!-- trunk (heavy body, high gains) -->
    <position name="trunk_yaw"   joint="trunk_yaw"   kp="500" kv="50"/>
    <position name="trunk_pitch" joint="trunk_pitch" kp="500" kv="50"/>
    <position name="trunk_roll"  joint="trunk_roll"  kp="500" kv="50"/>
    <!-- throw arm (fast movements, needs stiff tracking) -->
    <position name="throw_shoulder_yaw"   joint="throw_shoulder_yaw"   kp="300" kv="30"/>
    <position name="throw_shoulder_pitch" joint="throw_shoulder_pitch" kp="300" kv="30"/>
    <position name="throw_shoulder_roll"  joint="throw_shoulder_roll"  kp="300" kv="30"/>
    <position name="throw_elbow_flex"     joint="throw_elbow_flex"     kp="300" kv="30"/>
    <!-- glove arm -->
    <position name="glove_shoulder_yaw"   joint="glove_shoulder_yaw"   kp="200" kv="20"/>
    <position name="glove_shoulder_pitch" joint="glove_shoulder_pitch" kp="200" kv="20"/>
    <position name="glove_shoulder_roll"  joint="glove_shoulder_roll"  kp="200" kv="20"/>
    <position name="glove_elbow_flex"     joint="glove_elbow_flex"     kp="200" kv="20"/>
    <!-- rear leg (push-off, high force) -->
    <position name="rear_hip_yaw"    joint="rear_hip_yaw"    kp="400" kv="40"/>
    <position name="rear_hip_pitch"  joint="rear_hip_pitch"  kp="400" kv="40"/>
    <position name="rear_hip_roll"   joint="rear_hip_roll"   kp="400" kv="40"/>
    <position name="rear_knee_flex"  joint="rear_knee_flex"  kp="400" kv="40"/>
    <position name="rear_ankle_flex" joint="rear_ankle_flex" kp="200" kv="20"/>
    <!-- lead leg (landing, high force) -->
    <position name="lead_hip_yaw"    joint="lead_hip_yaw"    kp="400" kv="40"/>
    <position name="lead_hip_pitch"  joint="lead_hip_pitch"  kp="400" kv="40"/>
    <position name="lead_hip_roll"   joint="lead_hip_roll"   kp="400" kv="40"/>
    <position name="lead_knee_flex"  joint="lead_knee_flex"  kp="400" kv="40"/>
    <position name="lead_ankle_flex" joint="lead_ankle_flex" kp="200" kv="20"/>
  </actuator>

  <equality>
    <!-- Weld pelvis to mocap target — very stiff spring for RL root tracking -->
    <weld body1="pelvis_target" body2="pelvis"
          solref="0.01 1" solimp="0.95 0.99 0.001"/>
  </equality>
</mujoco>
"""


# ── CLI ──────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path.cwd())
    p.add_argument("--session-pitch", required=True)
    p.add_argument("--event", default="fp_10_time")
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    root = args.root.resolve()

    pitch = load_pitch(root, args.session_pitch)
    t0 = float(pitch.events[args.event])

    # Interpolate all landmarks at the reference time
    def at(arr):
        return _interp3(t0, pitch.time, arr)

    rear_hip   = at(pitch.rear_hip_jc)
    lead_hip   = at(pitch.lead_hip_jc)
    shoulder   = at(pitch.shoulder_jc)
    elbow      = at(pitch.elbow_jc)
    hand       = at(pitch.hand_jc)
    g_shoulder = at(pitch.glove_shoulder_jc)
    g_elbow    = at(pitch.glove_elbow_jc)
    g_hand     = at(pitch.glove_hand_jc)
    rear_knee  = at(pitch.rear_knee_jc)
    rear_ankle = at(pitch.rear_ankle_jc)
    lead_knee  = at(pitch.lead_knee_jc)
    lead_ankle = at(pitch.lead_ankle_jc)

    pelvis_mid, R_pelvis = _pelvis_frame(rear_hip, lead_hip)
    R_inv = R_pelvis.T  # inverse rotation (world → pelvis local)

    # Offsets in pelvis-local frame
    throw_shoulder_off = R_inv @ (shoulder - pelvis_mid)
    glove_shoulder_off = R_inv @ (g_shoulder - pelvis_mid)
    rear_hip_off       = R_inv @ (rear_hip - pelvis_mid)
    lead_hip_off       = R_inv @ (lead_hip - pelvis_mid)

    # Segment lengths (Euclidean)
    throw_upper_len = float(np.linalg.norm(elbow - shoulder))
    throw_fore_len  = float(np.linalg.norm(hand - elbow))
    glove_upper_len = float(np.linalg.norm(g_elbow - g_shoulder))
    glove_fore_len  = float(np.linalg.norm(g_hand - g_elbow))
    rear_thigh_len  = float(np.linalg.norm(rear_knee - rear_hip))
    rear_shin_len   = float(np.linalg.norm(rear_ankle - rear_knee))
    lead_thigh_len  = float(np.linalg.norm(lead_knee - lead_hip))
    lead_shin_len   = float(np.linalg.norm(lead_ankle - lead_knee))

    xml = build_xml(
        throw_shoulder_off=throw_shoulder_off,
        glove_shoulder_off=glove_shoulder_off,
        rear_hip_off=rear_hip_off,
        lead_hip_off=lead_hip_off,
        throw_upper_len=throw_upper_len,
        throw_fore_len=throw_fore_len,
        glove_upper_len=glove_upper_len,
        glove_fore_len=glove_fore_len,
        rear_thigh_len=rear_thigh_len,
        rear_shin_len=rear_shin_len,
        lead_thigh_len=lead_thigh_len,
        lead_shin_len=lead_shin_len,
    )

    if args.out is None:
        out = root / "pitching_mechanics" / "models" / f"pitcher_fullbody_{args.session_pitch}.xml"
    else:
        out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(xml)

    print(f"wrote: {out}")
    print(f"measured at {args.event}={t0:.4f}s")
    print(f"  throw arm:  upper={throw_upper_len:.4f}  fore={throw_fore_len:.4f}")
    print(f"  glove arm:  upper={glove_upper_len:.4f}  fore={glove_fore_len:.4f}")
    print(f"  rear leg:   thigh={rear_thigh_len:.4f}  shin={rear_shin_len:.4f}")
    print(f"  lead leg:   thigh={lead_thigh_len:.4f}  shin={lead_shin_len:.4f}")
    print(f"  pelvis mid: {pelvis_mid}")
    print(f"  throw shoulder off (pelvis-local): {throw_shoulder_off}")
    print(f"  glove shoulder off (pelvis-local): {glove_shoulder_off}")
    print(f"  rear hip off (pelvis-local):       {rear_hip_off}")
    print(f"  lead hip off (pelvis-local):       {lead_hip_off}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
