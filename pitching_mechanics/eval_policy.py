"""Replay a trained PPO policy in the MuJoCo viewer.

Loads runs/ppo_pitcher/ppo_pitcher_final and steps PitchEnv with
render_mode=\"human\" so you can visually compare learned mechanics
against the reference replay.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from pitching_mechanics.pitch_env import PitchEnv


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path.cwd())
    p.add_argument("--session-pitch", default="1623_3")
    p.add_argument("--model-path", type=Path, default=None,
                   help="Path to .zip model (default: runs/ppo_pitcher/ppo_pitcher_final.zip)")
    p.add_argument("--sleep-mult", type=float, default=4.0,
                   help="Slowdown factor for playback (default 4 = quarter speed)")
    p.add_argument("--no-loop", action="store_true", help="Play once and hold final frame (default: loop)")
    args = p.parse_args()

    root = args.root.resolve()
    session_pitch = args.session_pitch

    model_path = args.model_path or (root / "runs" / "ppo_pitcher" / "ppo_pitcher_final.zip")
    if not model_path.exists():
        raise SystemExit(
            f"Model not found at {model_path}. "
            "Run train_ppo.py first or pass --model-path."
        )

    # Environment with rendering enabled
    env = PitchEnv(
        root=root,
        session_pitch=session_pitch,
        render_mode="human",
        frame_skip=1,
    )

    model = PPO.load(str(model_path), env=env)
    dt = env.dt * args.sleep_mult

    def run_episode():
        obs, info = env.reset()
        obs = np.asarray(obs, dtype=np.float32)
        done = False
        truncated = False
        total_reward = 0.0
        steps = 0
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            obs = np.asarray(obs, dtype=np.float32)
            total_reward += reward
            steps += 1
            env.render()
            time.sleep(dt)
        return steps, total_reward

    loop = not args.no_loop
    if loop:
        print("Looping delivery; press Ctrl-C to exit.")
        try:
            while True:
                steps, total_reward = run_episode()
                print(f"Episode: steps={steps}  reward={total_reward:.2f}")
        except KeyboardInterrupt:
            pass
    else:
        steps, total_reward = run_episode()
        print(f"Episode: steps={steps}  reward={total_reward:.2f}")
        print("Holding final frame; close the MuJoCo window or press Ctrl-C to exit.")
        try:
            while True:
                env.render()
                time.sleep(1.0 / 60.0)
        except KeyboardInterrupt:
            pass
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

