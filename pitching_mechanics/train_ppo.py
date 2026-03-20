"""Train a PPO agent on the PitchEnv.

This is Step 4 of the roadmap: a full RL training loop.

Usage (local CPU, quick smoke test):

    python -m pitching_mechanics.train_ppo \
      --root . --session-pitch 1623_3 \
      --timesteps 200000 \
      --logdir runs/ppo_pitcher

"""
from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure

from pitching_mechanics.pitch_env import PitchEnv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path.cwd())
    p.add_argument("--session-pitch", default="1623_3")
    p.add_argument(
        "--timesteps",
        type=int,
        default=1_000_000,
        help="Total PPO training timesteps",
    )
    p.add_argument(
        "--logdir",
        type=Path,
        default=Path("runs/ppo_pitcher"),
        help="Directory for TensorBoard & SB3 logs/checkpoints",
    )
    p.add_argument(
        "--device",
        default="auto",
        help="PyTorch device: 'cpu', 'cuda', or 'auto'",
    )
    p.add_argument(
        "--save-every",
        type=int,
        default=200_000,
        help="Checkpoint interval in timesteps",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    logdir = args.logdir.resolve()
    logdir.mkdir(parents=True, exist_ok=True)

    # Environment (single-env for now; can be vectorized later)
    env = PitchEnv(root=root, session_pitch=args.session_pitch)

    # SB3 logger (TensorBoard-compatible)
    new_logger = configure(str(logdir), ["stdout", "tensorboard"])

    # n_steps = episode length (440) so PPO rollouts cover whole delivery
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=440,
        batch_size=220,  # divisor of n_steps to avoid SB3 warning
        n_epochs=10,
        learning_rate=3e-4,
        device=args.device,
    )
    model.set_logger(new_logger)

    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_every // env.n_steps,
        save_path=str(logdir / "checkpoints"),
        name_prefix="ppo_pitcher",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    # Train
    model.learn(
        total_timesteps=args.timesteps,
        callback=[checkpoint_callback],
        progress_bar=True,
    )

    # Final save
    final_path = logdir / "ppo_pitcher_final"
    model.save(str(final_path))
    print(f"Final model saved to: {final_path}")

    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

