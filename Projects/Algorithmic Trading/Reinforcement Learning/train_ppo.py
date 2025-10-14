# train_ppo.py
import os
import numpy as np
from typing import Dict, Callable
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from trading_env import TradingEnv

# ---------------- Config ----------------
SEED = 42
CFG = dict(
    cost_bps=10,              # try {5,10,20}
    cooldown_k=2,
    lambda_turn=1e-4,
    lambda_risk=0.20,
    vol_win=20,
    trend_win=200,
    short_block_thresh=5e-4,
    risk_norm=True,
    beta=1.0,                 # excess-return beta: r_excess = r_asset - beta*r_mkt
    total_timesteps=300_000,
    eval_every=50_000,        # early-stop evaluation cadence
)
PPO_KW = dict(
    verbose=0,
    seed=SEED,
    n_steps=2048,
    batch_size=512,
    gamma=0.997,
    gae_lambda=0.98,
    ent_coef=0.002,
    learning_rate=2e-4,
    clip_range=0.12,
    vf_coef=0.7,
    max_grad_norm=0.5,
)
# ----------------------------------------


def try_load(path: str) -> np.ndarray | None:
    return np.load(path) if os.path.exists(path) else None


def load_arrays():
    Xtr = np.load("env_data/X_train.npy"); Rtr = np.load("env_data/rets_train.npy")
    Xva = np.load("env_data/X_val.npy");   Rva = np.load("env_data/rets_val.npy")
    Xte = np.load("env_data/X_test.npy");  Rte = np.load("env_data/rets_test.npy")
    # Optional market returns for excess-return reward
    Rtr_m = try_load("env_data/rets_train_mkt.npy")
    Rva_m = try_load("env_data/rets_val_mkt.npy")
    Rte_m = try_load("env_data/rets_test_mkt.npy")
    return (Xtr, Rtr, Rtr_m), (Xva, Rva, Rva_m), (Xte, Rte, Rte_m)


def make_env(X, R, Rm) -> TradingEnv:
    return TradingEnv(
        X, R, Rm,
        beta=CFG["beta"],
        cost_bps=CFG["cost_bps"],
        cooldown_k=CFG["cooldown_k"],
        lambda_turn=CFG["lambda_turn"],
        lambda_risk=CFG["lambda_risk"],
        vol_win=CFG["vol_win"],
        trend_win=CFG["trend_win"],
        short_block_thresh=CFG["short_block_thresh"],
        risk_norm=CFG["risk_norm"],
        seed=SEED,
    )


def metrics_from_equity(eq: np.ndarray, pos: np.ndarray, freq: int = 252) -> Dict[str, float]:
    eq = np.asarray(eq, np.float32)
    pos = np.asarray(pos, np.float32)
    ret = np.diff(eq) / np.clip(eq[:-1], 1e-12, None)
    std = ret.std()
    sharpe = 0.0 if std == 0 else (ret.mean() / (std + 1e-12)) * np.sqrt(freq)
    mdd = (1.0 - eq / np.maximum.accumulate(eq)).max() if len(eq) > 1 else 0.0
    turn = np.mean(np.abs(np.diff(pos))) if len(pos) > 1 else 0.0
    return {"final_equity": float(eq[-1]), "sharpe": float(sharpe), "max_dd": float(mdd), "turnover": float(turn)}


def rollout(env: TradingEnv, policy: Callable[[np.ndarray], int]) -> Dict[str, float]:
    obs, _ = env.reset(seed=SEED)
    eq = [1.0]; pos = [0]
    done = False
    while not done:
        a = policy(obs)
        obs, r, done, _, info = env.step(a)
        eq.append(info["equity"])
        pos.append(info["pos"])
    return metrics_from_equity(eq, pos[:-1])


def random_policy(_obs: np.ndarray) -> int:
    return np.random.randint(0, 3)


def buyhold_metrics(rets: np.ndarray) -> Dict[str, float]:
    eq = np.cumprod(1.0 + rets).astype(np.float32)
    eq = np.insert(eq, 0, 1.0).astype(np.float32)
    pos = np.zeros(len(eq) - 1, dtype=np.float32)
    return metrics_from_equity(eq, pos)


def main():
    set_random_seed(SEED); np.random.seed(SEED)
    (Xtr, Rtr, Rtr_m), (Xva, Rva, Rva_m), (Xte, Rte, Rte_m) = load_arrays()

    env_tr = make_env(Xtr, Rtr, Rtr_m)
    env_va = make_env(Xva, Rva, Rva_m)
    env_te = make_env(Xte, Rte, Rte_m)

    model = PPO("MlpPolicy", env_tr, **PPO_KW)

    # ----- Early stopping loop on VAL Sharpe -----
    best_val = -np.inf
    best_path = "best_ppo.zip"
    steps = 0
    while steps < CFG["total_timesteps"]:
        chunk = min(CFG["eval_every"], CFG["total_timesteps"] - steps)
        model.learn(total_timesteps=chunk, reset_num_timesteps=False)
        steps += chunk

        # Evaluate on VAL
        val_metrics = rollout(env_va, lambda obs: model.predict(obs, deterministic=True)[0])
        if val_metrics["sharpe"] > best_val:
            best_val = val_metrics["sharpe"]
            model.save(best_path)

    # Load best checkpoint (by VAL Sharpe)
    if os.path.exists(best_path):
        model = PPO.load(best_path, env=env_tr, device="auto", print_system_info=False)

    # ----- Baselines & PPO -----
    rand_va = rollout(env_va, random_policy)
    rand_te = rollout(env_te, random_policy)
    bh_va = buyhold_metrics(Rva)
    bh_te = buyhold_metrics(Rte)
    ppo_va = rollout(env_va, lambda obs: model.predict(obs, deterministic=True)[0])
    ppo_te = rollout(env_te, lambda obs: model.predict(obs, deterministic=True)[0])

    print(
        f"\n=== COST_BPS={CFG['cost_bps']}  COOLDOWN_K={CFG['cooldown_k']}  "
        f"LAMBDA_TURN={CFG['lambda_turn']}  LAMBDA_RISK={CFG['lambda_risk']}  "
        f"RISK_NORM={CFG['risk_norm']}  BETA={CFG['beta']} ===\n"
    )
    print("RANDOM  VAL :", rand_va)
    print("RANDOM  TEST:", rand_te)
    print("B&H     VAL :", bh_va)
    print("B&H     TEST:", bh_te)
    print("PPO     VAL :", ppo_va)
    print("PPO     TEST:", ppo_te)


if __name__ == "__main__":
    main()
