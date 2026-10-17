"""Compare the classical controller (swing-up + PID) with trained SAC agents.

Scenarios (the same episodes/seeds are used for every controller):
  nominal       : default Pendulum-v1
  mass x0.7/1.5 : pendulum mass changed (the classical controller assumes the nominal model)
  noisy obs     : Gaussian noise (sigma = 0.1) added to the observation

Metrics: mean return, success rate (|theta| < 0.1 rad on average over the last 50 steps),
mean absolute torque (control effort).

Usage:  python src/evaluate.py --seeds 0 1 2 --episodes 100
"""
import argparse
import json
import os

import gymnasium as gym
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import SAC

from classical import SwingUpPID

SCENARIOS = {
    "nominal": dict(mass=1.0, obs_noise=0.0),
    "mass x0.7": dict(mass=0.7, obs_noise=0.0),
    "mass x1.5": dict(mass=1.5, obs_noise=0.0),
    "noisy obs": dict(mass=1.0, obs_noise=0.1),
}
EVAL_SEED_OFFSET = 1000  # evaluation episodes are different from any training seed


def run_episode(policy, scenario, seed, record=False):
    env = gym.make("Pendulum-v1")
    env.unwrapped.m = scenario["mass"]
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)
    if hasattr(policy, "reset"):
        policy.reset()
    total, thetas, torques = 0.0, [], []
    for _ in range(200):
        seen = obs + rng.normal(0, scenario["obs_noise"], size=obs.shape) if scenario["obs_noise"] else obs
        action = policy.act(seen) if hasattr(policy, "act") else policy.predict(seen, deterministic=True)[0]
        obs, reward, _, _, _ = env.step(action)
        total += reward
        thetas.append(float(np.arctan2(obs[1], obs[0])))
        torques.append(float(action[0]))
    env.close()
    success = np.mean(np.abs(thetas[-50:])) < 0.1
    out = dict(ret=total, success=bool(success), effort=float(np.mean(np.abs(torques))))
    if record:
        out["theta"], out["torque"] = thetas, torques
    return out


def evaluate(policy, scenario, episodes):
    rows = [run_episode(policy, scenario, EVAL_SEED_OFFSET + i) for i in range(episodes)]
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    controllers = {"Swing-up + PID": SwingUpPID()}
    for s in args.seeds:
        controllers[f"SAC (seed {s})"] = SAC.load(f"{args.out}/models/sac_seed{s}", device="cpu")

    records = []
    for sc_name, sc in SCENARIOS.items():
        for c_name, policy in controllers.items():
            df = evaluate(policy, sc, args.episodes)
            records.append(dict(scenario=sc_name, controller=c_name, ret_mean=df.ret.mean(),
                                ret_std=df.ret.std(), success=df.success.mean(), effort=df.effort.mean()))
    res = pd.DataFrame(records)
    res.to_csv(f"{args.out}/evaluation_per_controller.csv", index=False)

    # aggregate the SAC seeds (mean over seeds) so the table has one SAC row per scenario
    res["group"] = np.where(res.controller.str.startswith("SAC"), "SAC (mean of seeds)", res.controller)
    agg = res.groupby(["scenario", "group"], sort=False).agg(
        return_mean=("ret_mean", "mean"), return_std_across_seeds=("ret_mean", "std"),
        success_rate=("success", "mean"), mean_abs_torque=("effort", "mean")).reset_index()
    agg.to_csv(f"{args.out}/evaluation_summary.csv", index=False)
    print(agg.round(2).to_string(index=False))

    # ---- figure 1: learning curves ----
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for s in args.seeds:
        log = pd.read_csv(f"{args.out}/logs/sac_seed{s}.monitor.csv", skiprows=1)
        steps = log.l.cumsum()
        ax.plot(steps, log.r.rolling(5, min_periods=1).mean(), alpha=0.8, label=f"SAC seed {s}")
    pid_ret = agg[(agg.scenario == "nominal") & (agg.group == "Swing-up + PID")].return_mean.iloc[0]
    ax.axhline(pid_ret, color="k", ls="--", label="Swing-up + PID (nominal)")
    ax.set_xlabel("Training steps")
    ax.set_ylabel("Episode return (5-episode moving average)")
    ax.set_title("SAC learning curves vs classical controller")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{args.out}/learning_curves.png", dpi=150)
    plt.close(fig)

    # ---- figure 2: returns per scenario ----
    fig, ax = plt.subplots(figsize=(7, 4))
    names = list(SCENARIOS)
    width = 0.38
    pid_vals = [agg[(agg.scenario == n) & (agg.group == "Swing-up + PID")].return_mean.iloc[0] for n in names]
    sac_sub = [agg[(agg.scenario == n) & (agg.group == "SAC (mean of seeds)")] for n in names]
    sac_vals = [d.return_mean.iloc[0] for d in sac_sub]
    sac_err = [d.return_std_across_seeds.iloc[0] for d in sac_sub]
    x = np.arange(len(names))
    ax.bar(x - width / 2, pid_vals, width, label="Swing-up + PID")
    ax.bar(x + width / 2, sac_vals, width, yerr=sac_err, capsize=3, label="SAC (mean of seeds)")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Mean episode return (higher is better)")
    ax.set_title("Robustness to model changes and sensor noise")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(f"{args.out}/robustness.png", dpi=150)
    plt.close(fig)

    # ---- figure 3: one example trajectory (same initial state) ----
    sc = SCENARIOS["nominal"]
    # pick the first evaluation seed whose initial pendulum is hanging down (needs a swing-up)
    demo_seed = next(EVAL_SEED_OFFSET + i for i in range(100)
                     if abs(np.arctan2(*gym.make("Pendulum-v1").reset(seed=EVAL_SEED_OFFSET + i)[0][:2])) > 2.8)
    pid_ep = run_episode(controllers["Swing-up + PID"], sc, demo_seed, record=True)
    sac_ep = run_episode(controllers[f"SAC (seed {args.seeds[0]})"], sc, demo_seed, record=True)
    t = np.arange(200) * 0.05
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 5), sharex=True)
    axes[0].plot(t, pid_ep["theta"], label="Swing-up + PID")
    axes[0].plot(t, sac_ep["theta"], label=f"SAC (seed {args.seeds[0]})")
    axes[0].set_ylabel("Angle from upright (rad)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[1].plot(t, pid_ep["torque"])
    axes[1].plot(t, sac_ep["torque"])
    axes[1].set_ylabel("Torque (N.m)")
    axes[1].set_xlabel("Time (s)")
    axes[1].grid(alpha=0.3)
    fig.suptitle("Swing-up from the hanging position (same initial state, nominal model)")
    fig.tight_layout()
    fig.savefig(f"{args.out}/trajectory.png", dpi=150)
    plt.close(fig)

    json.dump({"episodes": args.episodes, "seeds": args.seeds}, open(f"{args.out}/eval_config.json", "w"))


if __name__ == "__main__":
    main()
