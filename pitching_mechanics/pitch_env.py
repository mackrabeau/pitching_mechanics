"""Gymnasium environment for pitch optimization via residual RL.

The agent outputs small joint-angle offsets added to a pre-computed
reference IK trajectory.  Position actuators track the modified targets
while MuJoCo physics runs forward.  The pelvis root is pinned
kinematically from OBP data each step.

Reward = ball_release_speed + strike_bonus − energy_penalty − limit_penalty
"""
from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from pitching_mechanics.ball_flight import (
    BallRelease,
    StrikeCheck,
    check_strike,
    compute_reward as _ball_reward,
)
from pitching_mechanics.obp_fullsig import load_pitch
from pitching_mechanics.trajectory import precompute

# ──────────────────────────────────────────────────────────────────────────

WRIST_SPEED_RATIO = 1.5          # fingertip speed ≈ 1.5 × hand jc speed
FREEJOINT_NQ = 7
FREEJOINT_NV = 6


class PitchEnv(gym.Env):
    """Residual-policy pitch environment.

    Observation (49-d float32):
        hinge_angles (21) | hinge_vels (21) | hand_pos (3) | hand_vel (3) | time_frac (1)

    Action (21-d float32):
        Joint-angle offsets in [−1, 1], scaled by ``action_scale`` radians
        and added to the reference IK target for that timestep.
    """

    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(
        self,
        root: str | Path = ".",
        session_pitch: str = "1623_3",
        render_mode: str | None = None,
        frame_skip: int = 1,
        action_scale: float = 0.15,      # max offset per joint (rad)
        w_energy: float = 1e-4,           # per-step energy penalty weight
        w_limit: float = 0.5,             # per-step joint-limit penalty weight
        w_speed: float = 1.0,             # release speed reward weight
        w_strike: float = 10.0,           # release strike bonus weight
    ):
        super().__init__()

        self.root = Path(root).resolve()
        self.session_pitch = session_pitch
        self.render_mode = render_mode
        self.frame_skip = frame_skip
        self.action_scale = action_scale
        self.w_energy = w_energy
        self.w_limit = w_limit
        self.w_speed = w_speed
        self.w_strike = w_strike

        # ── Load OBP data ─────────────────────────────────────────────────
        self.pitch = load_pitch(self.root, session_pitch)
        self.t_br = float(self.pitch.events["BR_time"])

        # ── Load model ────────────────────────────────────────────────────
        xml_path = (
            self.root / "pitching_mechanics" / "models"
            / f"pitcher_fullbody_{session_pitch}.xml"
        )
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        # Use implicit integrator for better stability with stiff actuators
        self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICIT
        self.data = mujoco.MjData(self.model)
        self.dt = float(self.model.opt.timestep)

        self.n_hinge = self.model.nv - FREEJOINT_NV  # 21
        assert self.n_hinge == self.model.nu, (
            f"Expected {self.n_hinge} actuators, got {self.model.nu}"
        )

        # Site IDs
        self._hand_site_id = self.model.site("throw_hand_site").id

        # Joint limits for penalty computation  (nq,)
        # hinge joints start at index FREEJOINT_NQ in qpos
        self._jnt_lo = np.zeros(self.n_hinge, dtype=np.float64)
        self._jnt_hi = np.zeros(self.n_hinge, dtype=np.float64)
        for i, dof in enumerate(range(FREEJOINT_NV, self.model.nv)):
            jid = int(self.model.dof_jntid[dof])
            self._jnt_lo[i] = self.model.jnt_range[jid, 0]
            self._jnt_hi[i] = self.model.jnt_range[jid, 1]

        # ── Pre-compute reference trajectory ──────────────────────────────
        t_start = float(self.pitch.events["fp_10_time"])
        t_end = float(self.pitch.events["MIR_time"]) + 0.25
        self.ref = precompute(self.model, self.pitch, t_start, t_end, verbose=False)
        self.n_steps = self.ref.n_steps

        # Reference hinge angles for each frame (N, n_hinge)
        self._ref_ctrl = self.ref.qpos_traj[:, FREEJOINT_NQ:].copy()

        # BR frame index
        self._br_frame = int(np.argmin(np.abs(self.ref.times - self.t_br)))

        # ── Gym spaces ────────────────────────────────────────────────────
        obs_dim = self.n_hinge * 2 + 3 + 3 + 1  # 49
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            -1.0, 1.0, shape=(self.n_hinge,), dtype=np.float32
        )

        # ── Internal state ────────────────────────────────────────────────
        self._frame = 0
        self._prev_hand_pos: np.ndarray | None = None
        self._released = False
        self._episode_energy = 0.0
        self._viewer = None

    # ──────────────────────────────────────────────────────────────────────
    # Gym API
    # ──────────────────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._frame = 0
        self._released = False
        self._episode_energy = 0.0

        # Set state from reference frame 0
        self.data.qpos[:] = self.ref.qpos_traj[0]
        self.data.qvel[:] = self.ref.qvel_traj[0]
        self.data.ctrl[:] = self._ref_ctrl[0]
        # Position mocap target at reference pelvis pose
        self.data.mocap_pos[0] = self.ref.qpos_traj[0, 0:3]
        self.data.mocap_quat[0] = self.ref.qpos_traj[0, 3:7]
        mujoco.mj_forward(self.model, self.data)

        self._prev_hand_pos = self.data.site_xpos[self._hand_site_id].copy()
        return self._get_obs(), self._get_info()

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0).astype(np.float64)
        offset = action * self.action_scale  # radians

        reward = 0.0
        terminated = False
        truncated = False

        for _ in range(self.frame_skip):
            if self._frame >= self.n_steps:
                truncated = True
                break

            f = min(self._frame, self.n_steps - 1)

            # Reference ctrl + agent's offset
            ctrl = self._ref_ctrl[f] + offset
            self.data.ctrl[:] = ctrl

            # Drive the pelvis via mocap body (weld constraint pulls pelvis)
            self.data.mocap_pos[0] = self.ref.qpos_traj[f, 0:3]
            self.data.mocap_quat[0] = self.ref.qpos_traj[f, 3:7]

            # Physics step — weld constraint handles root, actuators handle hinges
            mujoco.mj_step(self.model, self.data)

            # ── Per-step penalties ────────────────────────────────────────
            reward += self._step_penalty(ctrl)

            # ── Ball release ──────────────────────────────────────────────
            if self._frame == self._br_frame and not self._released:
                self._released = True
                reward += self._release_reward()

            self._prev_hand_pos = self.data.site_xpos[self._hand_site_id].copy()
            self._frame += 1

            # Check for simulation blow-up (NaN or joints way past limits)
            q_hinge = self.data.qpos[FREEJOINT_NQ:]
            if np.any(np.isnan(self.data.qpos)) or np.any(np.abs(q_hinge) > 10.0):
                terminated = True
                reward -= 20.0  # penalty for instability
                break

        obs = self._get_obs()
        info = self._get_info()

        # Episode ends when we've played through the full window
        if self._frame >= self.n_steps:
            truncated = True

        return obs, float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            if self._viewer is None:
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._viewer.sync()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    # ──────────────────────────────────────────────────────────────────────
    # Internals
    # ──────────────────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        hinge_ang = self.data.qpos[FREEJOINT_NQ:].copy()
        hinge_vel = self.data.qvel[FREEJOINT_NV:].copy()
        hand_pos = self.data.site_xpos[self._hand_site_id].copy()

        # Hand velocity via site Jacobian
        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        mujoco.mj_jacSite(self.model, self.data, jacp, None, self._hand_site_id)
        hand_vel = jacp @ self.data.qvel

        time_frac = np.array([self._frame / max(self.n_steps - 1, 1)], dtype=np.float64)

        return np.concatenate([
            hinge_ang, hinge_vel, hand_pos, hand_vel, time_frac,
        ]).astype(np.float32)

    def _get_info(self) -> dict:
        return {
            "frame": self._frame,
            "time": float(self.ref.times[min(self._frame, self.n_steps - 1)]),
            "released": self._released,
        }

    def _step_penalty(self, ctrl: np.ndarray) -> float:
        """Per-step energy + joint-limit penalties (bounded)."""
        # Energy: sum of squared control * dt
        energy = float(np.sum(ctrl ** 2)) * self.dt
        self._episode_energy += energy

        # Joint-limit proximity penalty (clamped per-joint to avoid explosion)
        q_hinge = self.data.qpos[FREEJOINT_NQ:]
        margin = 0.1  # radians — start penalizing 0.1 rad from limit
        lo_viol = np.clip(self._jnt_lo + margin - q_hinge, 0.0, 1.0)
        hi_viol = np.clip(q_hinge - (self._jnt_hi - margin), 0.0, 1.0)
        limit_pen = float(np.sum(lo_viol ** 2 + hi_viol ** 2))

        return -self.w_energy * energy - self.w_limit * limit_pen

    def _release_reward(self) -> float:
        """Reward at ball release: speed + strike bonus."""
        hand_pos = self.data.site_xpos[self._hand_site_id].copy()

        # Hand velocity via Jacobian
        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        mujoco.mj_jacSite(self.model, self.data, jacp, None, self._hand_site_id)
        hand_vel = jacp @ self.data.qvel
        hand_speed = float(np.linalg.norm(hand_vel))

        # Estimated ball speed
        ball_speed = hand_speed * WRIST_SPEED_RATIO
        ball_speed_mph = ball_speed * 2.23694

        # Forward fraction
        fwd_frac = float(hand_vel[0] / hand_speed) if hand_speed > 1e-6 else 0.0

        # Build BallRelease + StrikeCheck for reward computation
        release = BallRelease(
            pos=hand_pos,
            hand_jc_vel=hand_vel,
            hand_jc_speed=hand_speed,
            est_ball_speed_ms=ball_speed,
            est_ball_speed_mph=ball_speed_mph,
            forward_frac=fwd_frac,
            time=float(self.ref.times[self._frame]),
        )
        strike = check_strike(release)
        return _ball_reward(release, strike, w_speed=self.w_speed, w_strike=self.w_strike)


# ── Quick smoke test ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path.cwd())
    p.add_argument("--session-pitch", default="1623_3")
    p.add_argument("--episodes", type=int, default=3)
    args = p.parse_args()

    env = PitchEnv(root=args.root, session_pitch=args.session_pitch)
    print(f"Observation space: {env.observation_space}")
    print(f"Action space:      {env.action_space}")
    print(f"Episode length:    {env.n_steps} steps ({env.n_steps * env.dt:.3f}s)")
    print(f"BR at frame:       {env._br_frame}")
    print()

    for ep in range(args.episodes):
        obs, info = env.reset()
        total_reward = 0.0
        steps = 0
        while True:
            action = env.action_space.sample()  # random actions
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            if terminated or truncated:
                break
        print(f"Episode {ep + 1}: steps={steps}  reward={total_reward:.2f}"
              f"  energy={env._episode_energy:.2f}  released={info['released']}")

    env.close()
    print("\nSmoke test passed ✓")
