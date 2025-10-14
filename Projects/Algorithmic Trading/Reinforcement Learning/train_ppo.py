# train_ppo.py
import os
import numpy as np
from typing import Dict, Callable
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from trading_env import TradingEnv

# --------------- Config ---------------
SEED = 42
CFG = dict(
    cost_bps=10,
    cooldown_k=2,
    lambda_turn=1e-4,
    lambda_risk=0.20,
    vol_win=20,
    trend_win=200,
    short_block_thresh=5e-4,
    risk_norm=True,
    beta=1.0,                # default; will be replaced by TRAIN-estimated beta if market series exist
    total_timesteps=300_000,
    eval_every=50_000,       # early-stop cadence
    seeds=[42, 43, 44],      # multi-seed average
    do_grid=False,           # set True for tiny grid sweep
    grid_vals=[(0.1, 3e-4), (0.2, 5e-4), (0.3, 8e-4)],
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
# --------------------------------------


def try_load(path: str) -> np.ndarray | None:
    return np.load(path) if os.path.exists(path) else None


def load_arrays():
    Xtr = np.load("env_data/X_train.npy"); Rtr = np.load("env_data/rets_train.npy")
    Xva = np.load("env_data/X_val.npy");   Rva = np.load("env_data/rets_val.npy")
    Xte = np.load("env_data/X_test.npy");  Rte = np.load("env_data/rets_test.npy")
    Rtr_m = try_load("env_data/rets_train_mkt.npy")
    Rva_m = try_load("env_data/rets_val_mkt.npy")
    Rte_m = try_load("env_data/rets_test_mkt.npy")
    return (Xtr, Rtr, Rtr_m), (Xva, Rva, Rva_m), (Xte, Rte, Rte_m)


def estimate_beta(r: np.ndarray, r_m: np.ndarray) -> float:
    """OLS beta on TRAIN: cov(r, r_m) / var(r_m)"""
    r = r.astype(np.float64); r_m = r_m.astype(np.float64)
    v = np.var(r_m, ddof=1)
    if v <= 1e-12:
        return 1.0
    cov = np.cov(r, r_m, ddof=1)[0, 1]
    return float(cov / v)


def make_env(X, R, Rm, beta) -> TradingEnv:
    return TradingEnv(
        X, R, Rm,
        beta=beta,
        cost_bps=CFG["cost_bps"],
        cooldown_k=CFG["cooldown_k"],
        lambda_turn=CFG["lambda_turn"],
        lambda_risk=CFG["lambda_risk"],
        vol_win=CFG["vol_win"],
        trend_win=CFG["trend_win"],
        short_block_thresh=CFG["short_block_thresh"],
        risk_norm=CFG["risk_norm"],
        seed=None,  # we seed per-run
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


def rollout(env: TradingEnv, policy: Callable[[np.ndarray], int], seed: int) -> Dict[str, float]:
    obs, _ = env.reset(seed=seed)
    eq = [1.0]; pos = [0]
    done = False
    while not done:
        a = policy(obs)
        obs, r, done, _, info = env.step(a)
        eq.append(info["equity"]); pos.append(info["pos"])
    return metrics_from_equity(eq, pos[:-1])


def random_policy(_obs: np.ndarray) -> int:
    return np.random.randint(0, 3)


def buyhold_metrics(rets: np.ndarray) -> Dict[str, float]:
    eq = np.cumprod(1.0 + rets).astype(np.float32)
    eq = np.insert(eq, 0, 1.0).astype(np.float32)
    pos = np.zeros(len(eq) - 1, dtype=np.float32)
    return metrics_from_equity(eq, pos)


def train_once(seed: int, beta: float, arrays):
    (Xtr, Rtr, Rtr_m), (Xva, Rva, Rva_m), (Xte, Rte, Rte_m) = arrays
    set_random_seed(seed); np.random.seed(seed)

    env_tr = make_env(Xtr, Rtr, Rtr_m, beta)
    env_va = make_env(Xva, Rva, Rva_m, beta)
    env_te = make_env(Xte, Rte, Rte_m, beta)

    model = PPO("MlpPolicy", env_tr, **{**PPO_KW, "seed": seed})

    # Early stopping loop on VAL Sharpe
    best_val = -np.inf
    steps = 0
    best_path = f"best_ppo_seed{seed}.zip"
    while steps < CFG["total_timesteps"]:
        chunk = min(CFG["eval_every"], CFG["total_timesteps"] - steps)
        model.learn(total_timesteps=chunk, reset_num_timesteps=False)
        steps += chunk
        val_m = rollout(env_va, lambda o: model.predict(o, deterministic=True)[0], seed)
        if val_m["sharpe"] > best_val:
            best_val = val_m["sharpe"]
            model.save(best_path)

    # Load best snapshot
    model = PPO.load(best_path, env=env_tr, device="auto", print_system_info=False)

    # Baselines & PPO
    rand_va = rollout(env_va, random_policy, seed)
    rand_te = rollout(env_te, random_policy, seed)
    bh_va = buyhold_metrics(Rva)
    bh_te = buyhold_metrics(Rte)
    ppo_va = rollout(env_va, lambda o: model.predict(o, deterministic=True)[0], seed)
    ppo_te = rollout(env_te, lambda o: model.predict(o, deterministic=True)[0], seed)

    return {"rand_va": rand_va, "rand_te": rand_te, "bh_va": bh_va, "bh_te": bh_te, "ppo_va": ppo_va, "ppo_te": ppo_te}


def average_metrics(runs, key):
    cols = ["final_equity", "sharpe", "max_dd", "turnover"]
    return {c: float(np.mean([r[key][c] for r in runs])) for c in cols}


def main():
    arrays = load_arrays()
    (Xtr, Rtr, Rtr_m), (Xva, Rva, Rva_m), (Xte, Rte, Rte_m) = arrays

    # Data-driven beta (TRAIN)
    beta = CFG["beta"]
    if Rtr_m is not None:
        beta = estimate_beta(Rtr, Rtr_m)
        print(f"Estimated TRAIN beta = {beta:.3f}")

    # Optional tiny grid sweep over lambda_risk & short_block_thresh (single seed)
    if CFG["do_grid"]:
        best = None
        for lam_risk, gate in CFG["grid_vals"]:
            CFG["lambda_risk"] = lam_risk
            CFG["short_block_thresh"] = gate
            out = train_once(SEED, beta, arrays)
            score = out["ppo_va"]["sharpe"]
            print(f"λrisk={lam_risk}, gate={gate}  -> VAL Sharpe {score:.3f}")
            if best is None or score > best[0]:
                best = (score, lam_risk, gate)
        # lock the best config
        if best:
            CFG["lambda_risk"] = best[1]; CFG["short_block_thresh"] = best[2]
            print("Selected by VAL:", best)

    # Multi-seed training/eval
    runs = [train_once(s, beta, arrays) for s in CFG["seeds"]]

    print(
        f"\n=== COST_BPS={CFG['cost_bps']}  COOLDOWN_K={CFG['cooldown_k']}  "
        f"LAMBDA_TURN={CFG['lambda_turn']}  LAMBDA_RISK={CFG['lambda_risk']}  "
        f"RISK_NORM={CFG['risk_norm']}  BETA={beta:.3f} ===\n"
    )
    print("RANDOM  VAL :", average_metrics(runs, "rand_va"))
    print("RANDOM  TEST:", average_metrics(runs, "rand_te"))
    print("B&H     VAL :", average_metrics(runs, "bh_va"))
    print("B&H     TEST:", average_metrics(runs, "bh_te"))
    print("PPO     VAL :", average_metrics(runs, "ppo_va"))
    print("PPO     TEST:", average_metrics(runs, "ppo_te"))


if __name__ == "__main__":
    main()
