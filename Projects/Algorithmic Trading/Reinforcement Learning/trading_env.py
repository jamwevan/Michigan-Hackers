# trading_env.py
import gymnasium as gym
import numpy as np
from typing import Optional, Literal

ActionMode = Literal["discrete", "continuous"]

class TradingEnv(gym.Env):
    """
    Two tracks:
      - r_learn: agent reward (asset or excess-return, risk-normalized, penalties)
      - r_pnl  : true P&L return (pos * asset_ret - cost) for equity/metrics

    Modes:
      - "discrete": actions {0,1,2} → positions {-1,0,+1}; cost charged per trade
      - "continuous": actions in [-L,+L] (position size); cost proportional to |Δpos|

    Costs:
      - discrete: cost_bps_per_trade (bps) charged when pos changes
      - continuous: cost_bps_per_unit (bps) × |Δpos|
    """
    metadata = {}

    def __init__(
        self,
        X: np.ndarray,
        rets: np.ndarray,
        rets_mkt: Optional[np.ndarray] = None,
        *,
        mode: ActionMode = "discrete",
        max_leverage: float = 1.5,               # for continuous
        beta: float = 1.0,                       # for excess returns
        cost_bps_per_trade: int = 10,            # discrete mode
        cost_bps_per_unit: int = 10,             # continuous mode
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
        self.rets_mkt = rets_mkt.astype(np.float32) if rets_mkt is not None else None

        self.mode = mode
        self.L = float(max_leverage)
        self.beta = float(beta)

        self.cost_trade = float(cost_bps_per_trade)/1e4
        self.cost_unit  = float(cost_bps_per_unit)/1e4

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

        # EWMA vol & trend on learning base
        self.roll_vol = self._ewma_vol(self.ret_base, self.vol_win) + 1e-8
        self.trend = self._ewma(self.ret_base, self.trend_win)

        # Spaces
        self.T, self.F = self.X.shape
        if self.mode == "discrete":
            self.action_space = gym.spaces.Discrete(3)
        else:
            self.action_space = gym.spaces.Box(low=-self.L, high=self.L, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.F+1,), dtype=np.float32)

        self._rng = np.random.default_rng(seed)
        self.reset()

    # --- helpers ---
    @staticmethod
    def _ewma(x: np.ndarray, win: int) -> np.ndarray:
        alpha = 2.0 / (win + 1.0)
        out = np.empty_like(x, dtype=np.float32); m = 0.0
        for i, v in enumerate(x): m = (1 - alpha)*m + alpha*float(v); out[i] = m
        return out

    def _ewma_vol(self, x: np.ndarray, win: int) -> np.ndarray:
        alpha = 2.0 / (win + 1.0)
        out = np.empty_like(x, dtype=np.float32); v = 0.0
        for i, r in enumerate(x): v = (1 - alpha)*v + alpha*float(r)*float(r); out[i] = np.sqrt(max(v,0.0))
        return out

    def _obs(self): return np.concatenate([self.X[self.t], np.array([self.pos], np.float32)])

    # --- gym api ---
    def reset(self, seed=None, options=None):
        if seed is not None: self._rng = np.random.default_rng(seed)
        self.t = 0
        self.pos = 0.0  # float position (works for both modes)
        self.prev_pos = 0.0
        self.cooldown = 0
        self.eq_true = 1.0
        return self._obs(), {}

    def step(self, action):
        # Desired position
        if self.mode == "discrete":
            desired_pos = float(int(action) - 1)  # {-1,0,+1}
        else:
            desired_pos = float(np.clip(action, -self.L, self.L).reshape(-1)[0])

        # Trend gate (discourage fresh shorts in strong positive trend)
        if self.trend[self.t] > self.short_block_thresh and desired_pos < 0 and self.pos >= 0:
            desired_pos = 0.0

        # Cooldown: optionally freeze position changes
        if self.cooldown > 0 and abs(desired_pos - self.pos) > 1e-8:
            new_pos = self.pos
            delta = 0.0
            self.cooldown -= 1
        else:
            new_pos = desired_pos
            delta = new_pos - self.pos
            self.cooldown = self.cooldown_K if (abs(delta) > 1e-8 and self.cooldown_K > 0) else max(0, self.cooldown - 1)

        self.prev_pos, self.pos = self.pos, new_pos

        # Returns
        asset_ret = float(self.rets[self.t])
        base_ret  = float(self.ret_base[self.t])
        vol       = float(self.roll_vol[self.t])

        # Transaction costs
        if self.mode == "discrete":
            trade_cost = self.cost_trade if abs(delta) > 1e-8 else 0.0
        else:
            trade_cost = self.cost_unit * abs(delta)  # proportional to size change

        # True P&L (for equity)
        r_pnl = self.pos * asset_ret - trade_cost
        self.eq_true *= (1.0 + r_pnl)

        # Learning reward
        base = self.pos * base_ret
        if self.risk_norm: base = base / vol
        turn_pen = self.lambda_turn * abs(delta)
        risk_pen = self.lambda_risk * abs(self.pos) * vol
        r_learn = base - trade_cost - turn_pen - risk_pen

        self.t += 1
        terminated = (self.t >= self.T - 1)
        info = {"equity": self.eq_true, "pos": float(self.pos), "r_pnl": r_pnl, "r_learn": r_learn}
        return self._obs(), r_learn, terminated, False, info
