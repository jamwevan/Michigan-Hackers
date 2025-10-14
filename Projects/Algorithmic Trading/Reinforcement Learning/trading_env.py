# trading_env.py
import gymnasium as gym
import numpy as np
from typing import Optional

class TradingEnv(gym.Env):
    """
    Discrete actions: 0=short, 1=flat, 2=long
    Observation: features[t] + current position (-1/0/+1)

    Two tracks:
      • r_learn: reward to the agent (excess-return, risk-normalized, penalties)
      • r_pnl  : true P&L return (pos*asset_ret - cost) for equity/metrics

    Optional market returns (rets_mkt) enable excess-return reward:
      ret_excess = asset_ret - beta * mkt_ret
    """
    metadata = {}

    def __init__(
        self,
        X: np.ndarray,
        rets: np.ndarray,
        rets_mkt: Optional[np.ndarray] = None,
        *,
        beta: float = 1.0,
        cost_bps: int = 10,
        cooldown_k: int = 0,
        lambda_turn: float = 0.0,
        lambda_risk: float = 0.0,
        vol_win: int = 20,
        trend_win: int = 200,
        short_block_thresh: float = 5e-4,
        risk_norm: bool = True,
        seed: Optional[int] = 42,
    ):
        super().__init__()
        assert len(X) == len(rets), "X and rets must align"
        self.X = X.astype(np.float32)
        self.rets = rets.astype(np.float32)

        self.rets_mkt = None
        if rets_mkt is not None:
            assert len(rets_mkt) == len(rets), "rets_mkt must align with rets"
            self.rets_mkt = rets_mkt.astype(np.float32)

        # Config
        self.beta = float(beta)
        self.cost = float(cost_bps) / 1e4
        self.cooldown_K = int(cooldown_k)
        self.lambda_turn = float(lambda_turn)
        self.lambda_risk = float(lambda_risk)
        self.risk_norm = bool(risk_norm)
        self.vol_win = max(2, int(vol_win))
        self.trend_win = max(2, int(trend_win))
        self.short_block_thresh = float(short_block_thresh)

        # Base series for learning (asset or excess)
        if self.rets_mkt is None:
            self.ret_base = self.rets
        else:
            self.ret_base = self.rets - self.beta * self.rets_mkt

        # EWMA vol & trend on learning base series
        self.roll_vol = self._ewma_vol(self.ret_base, self.vol_win) + 1e-8
        self.trend = self._ewma(self.ret_base, self.trend_win)

        # Spaces
        self.T, self.F = self.X.shape
        self.action_space = gym.spaces.Discrete(3)  # 0 short, 1 flat, 2 long
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.F + 1,), dtype=np.float32
        )

        self._rng = np.random.default_rng(seed)
        self.reset()

    # ---------- helpers ----------
    @staticmethod
    def _ewma(x: np.ndarray, win: int) -> np.ndarray:
        alpha = 2.0 / (win + 1.0)
        out = np.empty_like(x, dtype=np.float32)
        m = 0.0
        for i, v in enumerate(x):
            m = (1 - alpha) * m + alpha * float(v)
            out[i] = m
        return out

    def _ewma_vol(self, x: np.ndarray, win: int) -> np.ndarray:
        alpha = 2.0 / (win + 1.0)
        out = np.empty_like(x, dtype=np.float32)
        v = 0.0
        for i, r in enumerate(x):
            v = (1 - alpha) * v + alpha * float(r) * float(r)
            out[i] = np.sqrt(max(v, 0.0))
        return out

    def _obs(self) -> np.ndarray:
        return np.concatenate([self.X[self.t], np.array([self.pos], dtype=np.float32)])

    # ---------- Gym API ----------
    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.t = 0
        self.pos = 0
        self.prev_pos = 0
        self.cooldown = 0
        self.eq_true = 1.0  # equity from true P&L returns
        return self._obs(), {}

    def step(self, action):
        desired_pos = int(action) - 1  # {-1,0,+1}

        # Trend gate: block fresh shorts in strong positive trend
        if self.trend[self.t] > self.short_block_thresh and desired_pos == -1 and self.pos >= 0:
            desired_pos = 0

        # Cooldown
        traded = False
        if self.cooldown > 0 and desired_pos != self.pos:
            new_pos = self.pos
            pen_turn = 0.0
            self.cooldown -= 1
        else:
            traded = (desired_pos != self.pos)
            pen_turn = self.lambda_turn * abs(desired_pos - self.pos)
            new_pos = desired_pos
            self.cooldown = self.cooldown_K if (traded and self.cooldown_K > 0) else max(0, self.cooldown - 1)

        self.prev_pos, self.pos = self.pos, new_pos

        # Returns
        asset_ret = float(self.rets[self.t])
        base_ret = float(self.ret_base[self.t])
        vol = float(self.roll_vol[self.t])

        # True P&L return (for equity/metrics)
        trade_cost = self.cost if traded else 0.0
        r_pnl = self.pos * asset_ret - trade_cost
        self.eq_true *= (1.0 + r_pnl)

        # Learning reward (risk-normalized, penalties)
        base = self.pos * base_ret
        if self.risk_norm:
            base = base / vol
        r_learn = base - trade_cost - pen_turn - (self.lambda_risk * abs(self.pos) * vol)

        self.t += 1
        terminated = (self.t >= self.T - 1)
        info = {"equity": self.eq_true, "pos": self.pos, "r_pnl": r_pnl, "r_learn": r_learn}
        return self._obs(), r_learn, terminated, False, info


if __name__ == "__main__":
    # Quick smoke test
    X = np.load("env_data/X_train.npy")
    R = np.load("env_data/rets_train.npy")
    try:
        Rm = np.load("env_data/rets_train_mkt.npy")
    except Exception:
        Rm = None
    env = TradingEnv(X, R, Rm, beta=1.0, cost_bps=10, cooldown_k=2, lambda_turn=1e-4,
                     lambda_risk=0.2, risk_norm=True, seed=0)
    obs, _ = env.reset()
    done = False
    while not done:
        a = np.random.randint(0, 3)
        obs, r, done, _, info = env.step(a)
    print("Random-agent final equity:", info["equity"])
