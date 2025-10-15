# train_rl.py
import os
import numpy as np
from typing import Dict, Callable, Tuple, List, Optional
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from stable_baselines3.common.utils import set_random_seed
from trading_env import TradingEnv

# =======================
# High-level configuration
# =======================
SEEDS: List[int] = [42, 43, 44]

# Use SAC with continuous sizing by default
ALGO: str = "SAC"          # Only SAC used here
MODE: str = "continuous"   # "continuous" actions (position sizing)
MAX_LEVERAGE: float = 1.5  # position range [-L, +L]

TOTAL_TIMESTEPS: int = 300_000
EVAL_EVERY: int = 50_000   # early-stop check cadence

# Environment knobs (costs/penalties/regime handling)
CFG = dict(
    cost_bps_per_trade=10,   # (unused in continuous mode)
    cost_bps_per_unit=10,    # per |Δpos| cost (continuous)
    cooldown_k=2,
    lambda_turn=1e-4,
    lambda_risk=0.20,        # risk aversion (|pos| * vol)
    vol_win=20,
    trend_win=200,
    short_block_thresh=5e-4, # stronger short gate in uptrends
    risk_norm=True,
)

# SAC hyperparams (continuous)
SAC_KW = dict(
    verbose=0,
    learning_rate=3e-4,
    buffer_size=200_000,
    batch_size=512,
    tau=0.02,
    gamma=0.995,
    train_freq=64,
    gradient_steps=64,
    ent_coef="auto",
)

# =======================
# Utility loaders/helpers
# =======================
def try_load(path: str) -> Optional[np.ndarray]:
    return np.load(path) if os.path.exists(path) else None

def load_splits():
    Xtr = np.load("env_data/X_train.npy"); Rtr = np.load("env_data/rets_train.npy"); Dtr = np.load("env_data/dates_train.npy")
    Xva = np.load("env_data/X_val.npy");   Rva = np.load("env_data/rets_val.npy");   Dva = np.load("env_data/dates_val.npy")
    Xte = np.load("env_data/X_test.npy");  Rte = np.load("env_data/rets_test.npy");  Dte = np.load("env_data/dates_test.npy")
    Rtr_m = try_load("env_data/rets_train_mkt.npy")
    Rva_m = try_load("env_data/rets_val_mkt.npy")
    Rte_m = try_load("env_data/rets_test_mkt.npy")
    return (Xtr, Rtr, Rtr_m, Dtr), (Xva, Rva, Rva_m, Dva), (Xte, Rte, Rte_m, Dte)

def estimate_beta(r: np.ndarray, r_m: np.ndarray) -> float:
    """OLS beta on TRAIN: cov(r, r_m) / var(r_m)."""
    r = r.astype(np.float64); r_m = r_m.astype(np.float64)
    v = np.var(r_m, ddof=1)
    if v <= 1e-12: return 1.0
    cov = np.cov(r, r_m, ddof=1)[0, 1]
    return float(cov / v)

def make_env(X, R, Rm, beta, seed) -> TradingEnv:
    return TradingEnv(
        X, R, Rm,
        mode=MODE, max_leverage=MAX_LEVERAGE,
        beta=beta,
        cost_bps_per_trade=CFG["cost_bps_per_trade"],
        cost_bps_per_unit=CFG["cost_bps_per_unit"],
        cooldown_k=CFG["cooldown_k"],
        lambda_turn=CFG["lambda_turn"],
        lambda_risk=CFG["lambda_risk"],
        vol_win=CFG["vol_win"],
        trend_win=CFG["trend_win"],
        short_block_thresh=CFG["short_block_thresh"],
        risk_norm=CFG["risk_norm"],
        seed=seed,
    )

def metrics_from_equity(eq: np.ndarray, pos: np.ndarray, freq: int = 252) -> Dict[str, float]:
    """Metrics from a true-equity series (eq includes initial 1.0)."""
    eq = np.asarray(eq, np.float32); pos = np.asarray(pos, np.float32)
    ret = np.diff(eq) / np.clip(eq[:-1], 1e-12, None)
    std = ret.std()
    sharpe = 0.0 if std == 0 else (ret.mean() / (std + 1e-12)) * np.sqrt(freq)
    mdd = (1.0 - eq / np.maximum.accumulate(eq)).max() if len(eq) > 1 else 0.0
    turn = float(np.mean(np.abs(np.diff(pos)))) if len(pos) > 1 else 0.0
    return {"final_equity": float(eq[-1]), "sharpe": float(sharpe), "max_dd": float(mdd), "turnover": turn}

def rollout(env: TradingEnv, policy: Callable[[np.ndarray], np.ndarray], seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (equity_series_with_initial_1, positions_series)."""
    obs, _ = env.reset(seed=seed)
    eq = [1.0]; pos = [0.0]
    done = False
    while not done:
        a = policy(obs)
        obs, r, done, _, info = env.step(a)
        eq.append(info["equity"]); pos.append(info["pos"])
    return np.asarray(eq, np.float32), np.asarray(pos[:-1], np.float32)

def random_policy(_obs: np.ndarray) -> np.ndarray:
    # random continuous action in [-1, 1] then env rescales to [-L, +L]
    return np.array([np.random.uniform(-1.0, 1.0)], dtype=np.float32)

def buyhold_metrics(rets: np.ndarray) -> Tuple[Dict[str, float], np.ndarray]:
    """Return metrics and equity (with initial 1.0) for buy & hold of the asset."""
    eq = np.cumprod(1.0 + rets).astype(np.float32)
    eq = np.insert(eq, 0, 1.0).astype(np.float32)
    pos = np.zeros(len(eq) - 1, dtype=np.float32)
    return metrics_from_equity(eq, pos), eq

def plot_equity(dates: np.ndarray, curves: List[Tuple[str, np.ndarray]], out_path: str):
    """
    Plot equity curves. Each curve is an equity series with initial 1.0.
    Align to dates by dropping the first point of equity (off-by-one fix).
    """
    plt.figure(figsize=(9, 4.8))
    dates = np.asarray(dates)
    for label, eq in curves:
        y = np.asarray(eq, np.float32)
        m = min(len(dates), len(y) - 1)
        if m <= 0: continue
        plt.plot(dates[:m], y[1:1 + m], label=label)
    plt.title("Equity Curve")
    plt.legend(); plt.tight_layout()
    plt.savefig(out_path); plt.close()

# =======================
# Training / evaluation
# =======================
def train_once(seed: int, arrays, beta: float) -> Dict[str, Dict[str, float]]:
    set_random_seed(seed); np.random.seed(seed)
    (Xtr, Rtr, Rtr_m, Dtr), (Xva, Rva, Rva_m, Dva), (Xte, Rte, Rte_m, Dte) = arrays

    env_tr = make_env(Xtr, Rtr, Rtr_m, beta, seed)
    env_va = make_env(Xva, Rva, Rva_m, beta, seed)
    env_te = make_env(Xte, Rte, Rte_m, beta, seed)

    # SAC model (continuous)
    model = SAC("MlpPolicy", env_tr, seed=seed, **SAC_KW)
    predict = lambda obs: model.predict(obs, deterministic=True)[0]
    load_model = lambda p: SAC.load(p, env=env_tr, device="auto", print_system_info=False)

    # Early stopping on VAL Sharpe
    best_val = -np.inf
    steps = 0
    best_path = f"best_{ALGO}_seed{seed}.zip"
    while steps < TOTAL_TIMESTEPS:
        chunk = min(EVAL_EVERY, TOTAL_TIMESTEPS - steps)
        model.learn(total_timesteps=chunk, reset_num_timesteps=False)
        steps += chunk

        eq_v, pos_v = rollout(env_va, predict, seed)
        val_m = metrics_from_equity(eq_v, pos_v)
        if val_m["sharpe"] > best_val:
            best_val = val_m["sharpe"]
            model.save(best_path)

    # Load best snapshot
    model = load_model(best_path)
    predict = lambda obs: model.predict(obs, deterministic=True)[0]

    # Evaluate
    eq_v, pos_v = rollout(env_va, predict, seed)
    eq_t, pos_t = rollout(env_te, predict, seed)
    val_m = metrics_from_equity(eq_v, pos_v)
    test_m = metrics_from_equity(eq_t, pos_t)

    bh_val_m, bh_val_eq = buyhold_metrics(Rva)
    bh_test_m, bh_test_eq = buyhold_metrics(Rte)

    # Save TEST equity plot (B&H vs model)
    plot_equity(Dte, [("B&H", bh_test_eq), (ALGO, eq_t)], f"env_data/eq_test_seed{seed}.png")

    return {"val": val_m, "test": test_m, "bh_val": bh_val_m, "bh_test": bh_test_m}

def average_dict(ds: List[Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
    keys = ds[0].keys()
    out: Dict[str, Dict[str, float]] = {}
    for k in keys:
        cols = ds[0][k].keys()
        out[k] = {c: float(np.mean([d[k][c] for d in ds])) for c in cols}
    return out

def walk_forward(arrays, beta: float, k_folds: int = 3) -> Dict[str, float]:
    """
    Quick walk-forward: train on TRAIN + earlier VAL/TEST folds, evaluate on next fold.
    """
    (Xtr, Rtr, Rtr_m, _), (Xva, Rva, Rva_m, _), (Xte, Rte, Rte_m, Dte) = arrays
    X_all = np.concatenate([Xva, Xte], axis=0)
    R_all = np.concatenate([Rva, Rte], axis=0)
    M_all = np.concatenate([Rva_m, Rte_m], axis=0) if (Rva_m is not None and Rte_m is not None) else None

    N = len(R_all)
    fold = max(1, N // k_folds)
    results = []
    for i in range(k_folds):
        s = i * fold
        e = (i + 1) * fold if i < k_folds - 1 else N

        X_train = np.concatenate([Xtr, X_all[:s]], axis=0)
        R_train = np.concatenate([Rtr, R_all[:s]], axis=0)
        M_train = np.concatenate([Rtr_m, M_all[:s]], axis=0) if (M_all is not None and Rtr_m is not None) else None

        X_test = X_all[s:e]; R_test = R_all[s:e]; M_test = M_all[s:e] if M_all is not None else None

        env_tr = make_env(X_train, R_train, M_train, beta, seed=123 + i)
        env_te = make_env(X_test,  R_test,  M_test,  beta, seed=123 + i)

        model = SAC("MlpPolicy", env_tr, seed=123 + i, **SAC_KW)
        predict = lambda obs: model.predict(obs, deterministic=True)[0]

        # Shorter train per fold to keep it quick
        model.learn(total_timesteps=max(10_000, TOTAL_TIMESTEPS // 2), reset_num_timesteps=False)
        eq_t, pos_t = rollout(env_te, predict, seed=123 + i)
        results.append(metrics_from_equity(eq_t, pos_t))

    # Average walk-forward metrics
    out = {k: float(np.mean([r[k] for r in results])) for k in results[0].keys()}
    return out

# =======================
# Main
# =======================
def main():
    arrays = load_splits()
    (Xtr, Rtr, Rtr_m, _), (Xva, Rva, Rva_m, _), (Xte, Rte, Rte_m, _) = arrays

    # Data-driven beta (TRAIN) if market series exist
    beta = 1.0
    if Rtr_m is not None:
        beta = estimate_beta(Rtr, Rtr_m)
        print(f"Estimated TRAIN beta = {beta:.3f}")

    # Multi-seed training/eval
    runs = [train_once(s, arrays, beta) for s in SEEDS]
    avg = average_dict(runs)

    print(
        f"\n=== ALGO={ALGO} MODE={MODE} MAX_LEV={MAX_LEVERAGE}  "
        f"COST_unit={CFG['cost_bps_per_unit']}bps  CD={CFG['cooldown_k']}  "
        f"λ_turn={CFG['lambda_turn']}  λ_risk={CFG['lambda_risk']}  "
        f"risk_norm={CFG['risk_norm']}  beta={beta:.3f} ===\n"
    )
    print("AVG  B&H   VAL :", avg["bh_val"])
    print("AVG  B&H   TEST:", avg["bh_test"])
    print("AVG  MODEL VAL :", avg["val"])
    print("AVG  MODEL TEST:", avg["test"])

    # Walk-forward check
    wf = walk_forward(arrays, beta, k_folds=3)
    print("\nWalk-Forward (3 folds) TEST avg metrics:", wf)

if __name__ == "__main__":
    main()
