"""
env.py — 2D Point Mass Environment
Supports 1st-order (state=[x,y], action=[vx,vy])
and 2nd-order+damping (state=[x,y,vx,vy], action=[ax,ay]).
"""
import numpy as np

DT    = 0.05
DRAG  = 0.15
BOUND = 10.0

DEFAULT_OBSTACLES = [
    {"pos": [3.0, 3.0], "r": 1.2},
    {"pos": [5.0, 1.5], "r": 1.0},
    {"pos": [1.5, 6.0], "r": 0.9},
    {"pos": [5.5, 5.5], "r": 1.1},
    {"pos": [7.5, 3.0], "r": 0.8},
]


class PointMass2DEnv:
    def __init__(self):
        self.DT       = DT
        self.DRAG     = DRAG
        self.BOUND    = BOUND
        self.goal     = np.array([7.0, 7.0])
        self.start    = np.array([-7.0, -7.0])
        self.max_speed = 4.0
        self.max_acc   = 2.0
        self.dyn_model = "2nd"   # "1st" | "2nd"
        self.obstacles = [
            {"pos": np.array(o["pos"], dtype=float), "r": float(o["r"])}
            for o in DEFAULT_OBSTACLES
        ]

    # ── helpers ──────────────────────────────────────────
    def state_dim(self) -> int:
        return 2 if self.dyn_model == "1st" else 4

    def action_lim(self) -> float:
        return self.max_speed if self.dyn_model == "1st" else self.max_acc

    def init_state(self) -> np.ndarray:
        s = np.zeros(self.state_dim())
        s[:2] = self.start
        return s

    # ── single step ──────────────────────────────────────
    def step(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        lim = self.action_lim()
        a = np.clip(action, -lim, lim)
        if self.dyn_model == "1st":
            x = np.clip(state[0] + a[0] * self.DT, -self.BOUND, self.BOUND)
            y = np.clip(state[1] + a[1] * self.DT, -self.BOUND, self.BOUND)
            return np.array([x, y])
        else:
            x, y, vx, vy = state
            vx += a[0] * self.DT - self.DRAG * vx * self.DT
            vy += a[1] * self.DT - self.DRAG * vy * self.DT
            spd = np.hypot(vx, vy)
            if spd > self.max_speed:
                vx *= self.max_speed / spd
                vy *= self.max_speed / spd
            x += vx * self.DT;  y += vy * self.DT
            if abs(x) > self.BOUND: x = np.clip(x, -self.BOUND, self.BOUND); vx *= -0.5
            if abs(y) > self.BOUND: y = np.clip(y, -self.BOUND, self.BOUND); vy *= -0.5
            return np.array([x, y, vx, vy])

    # ── batch step [K, sd] ────────────────────────────────
    def batch_step(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        lim = self.action_lim()
        a = np.clip(actions, -lim, lim)
        if self.dyn_model == "1st":
            nx = np.clip(states[:, 0] + a[:, 0] * self.DT, -self.BOUND, self.BOUND)
            ny = np.clip(states[:, 1] + a[:, 1] * self.DT, -self.BOUND, self.BOUND)
            return np.stack([nx, ny], axis=1)
        else:
            x, y   = states[:, 0].copy(), states[:, 1].copy()
            vx, vy = states[:, 2].copy(), states[:, 3].copy()
            vx += a[:, 0] * self.DT - self.DRAG * vx * self.DT
            vy += a[:, 1] * self.DT - self.DRAG * vy * self.DT
            spd  = np.hypot(vx, vy)
            mask = spd > self.max_speed
            vx[mask] *= self.max_speed / spd[mask]
            vy[mask] *= self.max_speed / spd[mask]
            # ── Bug fix: bounce velocity at boundaries (mirrors step()) ──
            x_new, y_new = x + vx * self.DT, y + vy * self.DT
            vx = np.where(np.abs(x_new) > self.BOUND, vx * -0.5, vx)
            vy = np.where(np.abs(y_new) > self.BOUND, vy * -0.5, vy)
            x  = np.clip(x_new, -self.BOUND, self.BOUND)
            y  = np.clip(y_new, -self.BOUND, self.BOUND)
            return np.stack([x, y, vx, vy], axis=1)

    # ── cost functions ────────────────────────────────────
    def running_cost(self, states: np.ndarray) -> np.ndarray:
        gx, gy = self.goal
        x, y   = states[:, 0], states[:, 1]
        cost   = np.hypot(x - gx, y - gy) * 2.0
        for obs in self.obstacles:
            ox, oy = obs["pos"]
            d = np.hypot(x - ox, y - oy) - obs["r"]
            cost += np.where(d < 0.5, (0.5 - np.maximum(d, 0.0)) * 50.0, 0.0)
        # 2nd-order: soft speed penalty
        if self.dyn_model == "2nd":
            spd    = np.hypot(states[:, 2], states[:, 3])
            excess = spd - self.max_speed * 0.85
            cost  += np.where(excess > 0, excess ** 2 * 8.0, 0.0)
        return cost

    def terminal_cost(self, states: np.ndarray) -> np.ndarray:
        gx, gy = self.goal
        return np.hypot(states[:, 0] - gx, states[:, 1] - gy) * 10.0

    # ── serialisation helpers ─────────────────────────────
    def obstacles_to_list(self):
        return [{"pos": o["pos"].tolist(), "r": o["r"]} for o in self.obstacles]

    def set_obstacles_from_list(self, data: list):
        self.obstacles = [
            {"pos": np.array(o["pos"], dtype=float), "r": float(o["r"])}
            for o in data
        ]
