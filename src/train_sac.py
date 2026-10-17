"""Train a SAC agent on Pendulum-v1 (stable-baselines3) for several random seeds.

Usage:  python src/train_sac.py --seeds 0 1 2 --steps 20000
Saves models to results/models/ and per-episode training logs to results/logs/.
"""
import argparse
import os

import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    os.makedirs(f"{args.out}/models", exist_ok=True)
    os.makedirs(f"{args.out}/logs", exist_ok=True)

    for seed in args.seeds:
        env = Monitor(gym.make("Pendulum-v1"), filename=f"{args.out}/logs/sac_seed{seed}")
        model = SAC("MlpPolicy", env, seed=seed, verbose=0, device="cpu")
        model.learn(total_timesteps=args.steps)
        model.save(f"{args.out}/models/sac_seed{seed}")
        print(f"seed {seed}: done")


if __name__ == "__main__":
    main()
