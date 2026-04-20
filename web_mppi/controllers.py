"""
controllers.py — MPPI, DIAL-MPC, CEM, iCEM controllers.

All controllers expose:
  • compute(state) -> action [2]
  • reset(seed=None)
  • get_rollout_trajs(state, n_disp) -> (trajs, costs, indices, best_pos, n_explore)
  • cost_history : list[float]
  • last_samples : np.ndarray [K, H, 2]
  • last_costs   : np.ndarray [K]
  • n_explore    : int   (0 for CEM/iCEM)
"""
import numpy as np
from scipy.signal import savgol_filter
from env import PointMass2DEnv


# ═══════════════════════════════════════════════════════
#  Shared helpers
# ═══════════════════════════════════════════════════════

def _rollout_costs(env, state, samples):
    """Vectorised rollout. samples: [K, H, 2]. Returns costs [K]."""
    K, H, _ = samples.shape
    states = np.tile(state, (K, 1))
    costs  = np.zeros(K)
    for t in range(H):
        states = env.batch_step(states, samples[:, t, :])
        costs += env.running_cost(states)
    costs += env.terminal_cost(states)
    return costs


def _traj_from_samples(env, state, samples, indices):
    """Forward-integrate a subset of action sequences -> xy trajectories."""
    is_1st = env.dyn_model == "1st"
    H = samples.shape[1]
    trajs = []
    for ki in indices:
        traj = [state[:2].tolist()]
        sx, sy = float(state[0]), float(state[1])
        svx    = float(state[2]) if not is_1st and len(state) > 2 else 0.0
        svy    = float(state[3]) if not is_1st and len(state) > 3 else 0.0
        for t in range(H):
            ax, ay = float(samples[ki, t, 0]), float(samples[ki, t, 1])
            if is_1st:
                sx = float(np.clip(sx + ax * env.DT, -env.BOUND, env.BOUND))
                sy = float(np.clip(sy + ay * env.DT, -env.BOUND, env.BOUND))
            else:
                svx += ax * env.DT - env.DRAG * svx * env.DT
                svy += ay * env.DT - env.DRAG * svy * env.DT
                spd = (svx ** 2 + svy ** 2) ** 0.5
                if spd > env.max_speed:
                    svx *= env.max_speed / spd
                    svy *= env.max_speed / spd
                sx = float(np.clip(sx + svx * env.DT, -env.BOUND, env.BOUND))
                sy = float(np.clip(sy + svy * env.DT, -env.BOUND, env.BOUND))
            traj.append([sx, sy])
        trajs.append(traj)
    return trajs


def _pick_display_indices(K, n_explore, n_disp=20, best_idx=0):
    n_half  = n_disp // 2
    exp_idx = (np.linspace(0, max(n_explore - 1, 0), n_half, dtype=int).tolist()
               if n_explore > 0 else [])
    ws_cnt  = K - n_explore
    ws_idx  = (np.linspace(n_explore, K - 1, n_half, dtype=int).tolist()
               if ws_cnt > 0 else [])
    return sorted(set(exp_idx + ws_idx + [best_idx]))


def _apply_sg(U, lim, sg_enabled, sg_win, sg_order):
    if not sg_enabled:
        return U
    win   = sg_win if sg_win % 2 == 1 else sg_win + 1
    order = min(sg_order, win - 1)
    if win < 3 or order < 1 or U.shape[0] < win:
        return U
    try:
        return np.clip(
            savgol_filter(U, window_length=win, polyorder=order, axis=0),
            -lim, lim
        )
    except Exception:
        return U


# ═══════════════════════════════════════════════════════
#  MPPI
# ═══════════════════════════════════════════════════════
class MPPIController:
    label = "MPPI"

    def __init__(self, env, *, H=30, K=400, lam=0.05, sigma=0.9, alpha=1.0,
                 sg_enabled=False, sg_win=7, sg_order=3, seed=42):
        self.env = env
        self.H, self.K   = H, K
        self.lam, self.sigma, self.alpha = lam, sigma, alpha
        self.sg_enabled  = sg_enabled
        self.sg_win, self.sg_order = sg_win, sg_order
        self.rng = np.random.default_rng(seed)
        self.U   = np.zeros((H, 2))
        self.last_samples = None
        self.last_costs   = None
        self.n_explore    = int((1.0 - alpha) * K)
        self.cost_history = []

    def reset(self, seed=None):
        self.U[:] = 0.0
        self.last_samples = self.last_costs = None
        self.cost_history.clear()
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def compute(self, state):
        H, K       = self.H, self.K
        lam, sigma = self.lam, self.sigma
        alpha      = self.alpha
        gamma      = lam * (1.0 - alpha)
        inv_sig2   = 1.0 / (sigma ** 2)
        n_explore  = int((1.0 - alpha) * K)
        lim        = self.env.action_lim()

        eps         = self.rng.standard_normal((K, H, 2)) * sigma
        all_actions = np.empty((K, H, 2))
        states      = np.tile(state, (K, 1))
        costs       = np.zeros(K)

        for t in range(H):
            u0, u1 = self.U[t, 0], self.U[t, 1]
            all_actions[:n_explore, t, 0] = np.clip(eps[:n_explore, t, 0], -lim, lim)
            all_actions[:n_explore, t, 1] = np.clip(eps[:n_explore, t, 1], -lim, lim)
            all_actions[n_explore:, t, 0] = np.clip(u0 + eps[n_explore:, t, 0], -lim, lim)
            all_actions[n_explore:, t, 1] = np.clip(u1 + eps[n_explore:, t, 1], -lim, lim)
            states = self.env.batch_step(states, all_actions[:, t, :])
            rc     = self.env.running_cost(states)
            if gamma > 0:
                costs += rc + gamma * inv_sig2 * (u0 * eps[:, t, 0] + u1 * eps[:, t, 1])
            else:
                costs += rc

        costs += self.env.terminal_cost(states)
        self.last_costs = costs.copy()

        beta    = costs.min()
        weights = np.exp(-(costs - beta) / lam)
        weights /= weights.sum() + 1e-8
        self.U  = np.clip((weights[:, None, None] * all_actions).sum(axis=0), -lim, lim)
        self.U  = _apply_sg(self.U, lim, self.sg_enabled, self.sg_win, self.sg_order)

        self.last_samples = all_actions
        self.n_explore    = n_explore
        self.cost_history.append(float(costs.mean()))
        action = self.U[0].copy()
        self.U = np.roll(self.U, -1, axis=0); self.U[-1] = 0.0
        return action

    def get_rollout_trajs(self, state, n_disp=20):
        if self.last_samples is None: return [], [], [], 0, 0
        best_idx = int(np.argmin(self.last_costs))
        indices  = _pick_display_indices(self.K, self.n_explore, n_disp, best_idx)
        trajs    = _traj_from_samples(self.env, state, self.last_samples, indices)
        costs    = [float(self.last_costs[ki]) for ki in indices]
        return trajs, costs, indices, indices.index(best_idx) if best_idx in indices else 0, self.n_explore

    def sigma_schedule_list(self, *_): return []


# ═══════════════════════════════════════════════════════
#  DIAL-MPC
# ═══════════════════════════════════════════════════════
class DIALController:
    label = "DIAL-MPC"

    def __init__(self, env, *, H=30, K=400, N=5, lam=0.05,
                 sigma_base=1.2, beta1=0.6, beta2=0.5, alpha=1.0,
                 sg_enabled=False, sg_win=7, sg_order=3, seed=42):
        self.env = env
        self.H, self.K, self.N = H, K, N
        self.lam, self.alpha   = lam, alpha
        self.sigma_base        = sigma_base
        self.beta1, self.beta2 = beta1, beta2
        self.sg_enabled        = sg_enabled
        self.sg_win, self.sg_order = sg_win, sg_order
        self.rng = np.random.default_rng(seed)
        self.U   = np.zeros((H, 2))
        self.last_samples = None
        self.last_costs   = None
        self.last_sigmas  = None
        self.n_explore    = int((1.0 - alpha) * K)
        self.cost_history = []

    def reset(self, seed=None):
        self.U[:] = 0.0
        self.last_samples = self.last_costs = self.last_sigmas = None
        self.cost_history.clear()
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def _sigma_schedule(self, i):
        H, N, b1, b2, sb = self.H, self.N, self.beta1, self.beta2, self.sigma_base
        h = np.arange(H, dtype=float)
        return sb * np.exp(-((N - i) / (b1 * N) + (H - 1 - h) / (b2 * H)) / 2.0)

    def compute(self, state):
        H, K, N   = self.H, self.K, self.N
        lam, alpha = self.lam, self.alpha
        gamma      = lam * (1.0 - alpha)
        n_explore  = int((1.0 - alpha) * K)
        total_cost = 0.0
        last_actions = None

        for i in range(N, 0, -1):
            sigmas      = self._sigma_schedule(i)
            eps         = self.rng.standard_normal((K, H, 2)) * sigmas[None, :, None]
            all_actions = np.empty((K, H, 2))
            states      = np.tile(state, (K, 1))
            costs       = np.zeros(K)

            for t in range(H):
                u0, u1  = self.U[t, 0], self.U[t, 1]
                inv_s2  = 1.0 / (float(sigmas[t]) ** 2 + 1e-12)
                lim     = self.env.action_lim()
                all_actions[:n_explore, t, 0] = np.clip(eps[:n_explore, t, 0], -lim, lim)
                all_actions[:n_explore, t, 1] = np.clip(eps[:n_explore, t, 1], -lim, lim)
                all_actions[n_explore:, t, 0] = np.clip(u0 + eps[n_explore:, t, 0], -lim, lim)
                all_actions[n_explore:, t, 1] = np.clip(u1 + eps[n_explore:, t, 1], -lim, lim)
                states = self.env.batch_step(states, all_actions[:, t, :])
                rc     = self.env.running_cost(states)
                if gamma > 0:
                    costs += rc + gamma * inv_s2 * (u0 * eps[:, t, 0] + u1 * eps[:, t, 1])
                else:
                    costs += rc

            costs += self.env.terminal_cost(states)
            if i == 1:
                self.last_costs  = costs.copy()
                self.last_sigmas = sigmas
                last_actions     = all_actions.copy()

            beta    = costs.min()
            weights = np.exp(-(costs - beta) / lam)
            weights /= weights.sum() + 1e-8
            lim = self.env.action_lim()
            self.U = np.clip((weights[:, None, None] * all_actions).sum(axis=0), -lim, lim)
            self.U = _apply_sg(self.U, lim, self.sg_enabled, self.sg_win, self.sg_order)
            total_cost += float(costs.mean())

        self.last_samples = last_actions
        self.n_explore    = n_explore
        self.cost_history.append(total_cost / N)
        action = self.U[0].copy()
        self.U = np.roll(self.U, -1, axis=0); self.U[-1] = 0.0
        return action

    def get_rollout_trajs(self, state, n_disp=20):
        if self.last_samples is None: return [], [], [], 0, 0
        best_idx = int(np.argmin(self.last_costs))
        indices  = _pick_display_indices(self.K, self.n_explore, n_disp, best_idx)
        trajs    = _traj_from_samples(self.env, state, self.last_samples, indices)
        costs    = [float(self.last_costs[ki]) for ki in indices]
        return trajs, costs, indices, indices.index(best_idx) if best_idx in indices else 0, self.n_explore

    def sigma_schedule_list(self, i=1):
        return self._sigma_schedule(i).tolist()


# ═══════════════════════════════════════════════════════
#  CEM
# ═══════════════════════════════════════════════════════
class CEMController:
    """
    Cross-Entropy Method MPC.
    Iteratively updates Gaussian(mu, sigma) using top-K_elite trajectories.
    """
    label     = "CEM"
    n_explore = 0

    def __init__(self, env, *, H=30, K=400, lam=0.05, alpha=1.0,
                 N_iter=5, elite_frac=0.1, sigma=1.5,
                 sg_enabled=False, sg_win=7, sg_order=3, seed=42):
        self.env = env
        self.H, self.K  = H, K
        self.lam        = lam
        self.N_iter     = N_iter
        self.elite_frac = elite_frac
        self.sigma_init = sigma
        self.sg_enabled = sg_enabled
        self.sg_win, self.sg_order = sg_win, sg_order
        self.rng = np.random.default_rng(seed)
        self.U   = np.zeros((H, 2))
        self.last_samples = None
        self.last_costs   = None
        self.cost_history = []

    def reset(self, seed=None):
        self.U[:] = 0.0
        self.last_samples = self.last_costs = None
        self.cost_history.clear()
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def compute(self, state):
        H, K    = self.H, self.K
        lim     = self.env.action_lim()
        N_elite = max(2, int(K * self.elite_frac))
        mu      = self.U.copy()
        sigma   = np.full((H, 2), self.sigma_init)

        for _ in range(self.N_iter):
            noise   = self.rng.standard_normal((K, H, 2)) * sigma[None]
            samples = np.clip(mu[None] + noise, -lim, lim)
            costs   = _rollout_costs(self.env, state, samples)

            elite_idx = np.argsort(costs)[:N_elite]
            elites    = samples[elite_idx]
            mu        = elites.mean(axis=0)
            sigma     = np.maximum(elites.std(axis=0), 0.05)

        self.last_samples = samples
        self.last_costs   = costs
        self.U = _apply_sg(np.clip(mu, -lim, lim), lim,
                           self.sg_enabled, self.sg_win, self.sg_order)
        self.cost_history.append(float(costs.mean()))
        action = self.U[0].copy()
        self.U = np.roll(self.U, -1, axis=0); self.U[-1] = 0.0
        return action

    def get_rollout_trajs(self, state, n_disp=20):
        if self.last_samples is None: return [], [], [], 0, 0
        K        = self.K
        best_idx = int(np.argmin(self.last_costs))
        step     = max(1, K // n_disp)
        indices  = sorted(set(list(range(0, K, step))[:n_disp] + [best_idx]))
        trajs    = _traj_from_samples(self.env, state, self.last_samples, indices)
        costs    = [float(self.last_costs[ki]) for ki in indices]
        return trajs, costs, indices, indices.index(best_idx) if best_idx in indices else 0, 0


# ═══════════════════════════════════════════════════════
#  iCEM  (Pinneri et al., CoRL 2020)
# ═══════════════════════════════════════════════════════
class iCEMController:
    """
    Improved CEM with:
      1. Colored (power-law) noise for temporally-smooth samples
      2. Elite reuse: shift previous elites into next step's pool
      3. Best-action execution: execute best-elite's first action, not mu
    """
    label     = "iCEM"
    n_explore = 0

    def __init__(self, env, *, H=30, K=400, lam=0.05, alpha=1.0,
                 N_iter=5, elite_frac=0.1, sigma=1.5,
                 beta_color=1.0, elite_reuse=0.3,
                 sg_enabled=False, sg_win=7, sg_order=3, seed=42):
        self.env = env
        self.H, self.K    = H, K
        self.lam          = lam
        self.N_iter       = N_iter
        self.elite_frac   = elite_frac
        self.sigma_init   = sigma
        self.beta_color   = beta_color
        self.elite_reuse  = elite_reuse
        self.sg_enabled   = sg_enabled
        self.sg_win, self.sg_order = sg_win, sg_order
        self.rng          = np.random.default_rng(seed)
        self.U            = np.zeros((H, 2))
        self.prev_elites  = None
        self.last_samples = None
        self.last_costs   = None
        self.cost_history = []

    def reset(self, seed=None):
        self.U[:] = 0.0
        self.prev_elites  = None
        self.last_samples = self.last_costs = None
        self.cost_history.clear()
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def _colored_noise(self, K, H, sigma):
        """Power-law colored noise PSD ∝ 1/f^beta. sigma: [H,2]."""
        beta = self.beta_color
        if beta == 0.0 or H < 4:
            return self.rng.standard_normal((K, H, 2)) * sigma[None]
        f = np.fft.rfftfreq(H)
        f[0] = 1.0
        power = f ** (-beta / 2.0); power[0] = 0.0
        white = self.rng.standard_normal((K, H, 2))
        spec  = np.fft.rfft(white, axis=1) * power[None, :, None]
        colored = np.fft.irfft(spec, n=H, axis=1)
        s = colored.std(axis=(0, 1), keepdims=True)
        return colored / np.where(s > 1e-8, s, 1.0) * sigma[None]

    def compute(self, state):
        H, K    = self.H, self.K
        lim     = self.env.action_lim()
        N_elite = max(2, int(K * self.elite_frac))
        mu      = self.U.copy()
        sigma   = np.full((H, 2), self.sigma_init)
        best_elite = None

        for _ in range(self.N_iter):
            noise   = self._colored_noise(K, H, sigma)
            samples = np.clip(mu[None] + noise, -lim, lim)

            # Elite reuse: replace last n_reuse samples with shifted prev elites
            if self.prev_elites is not None:
                n_reuse = max(1, int(len(self.prev_elites) * self.elite_reuse))
                shifted = np.roll(self.prev_elites[:n_reuse], -1, axis=1)
                shifted[:, -1, :] = 0.0
                samples[-n_reuse:] = np.clip(shifted, -lim, lim)

            costs     = _rollout_costs(self.env, state, samples)
            elite_idx = np.argsort(costs)[:N_elite]
            elites    = samples[elite_idx]
            best_elite = elites[0]
            mu         = elites.mean(axis=0)
            sigma      = np.maximum(elites.std(axis=0), 0.05)

        self.prev_elites  = elites.copy()
        self.last_samples = samples
        self.last_costs   = costs
        self.U = _apply_sg(np.clip(mu, -lim, lim), lim,
                           self.sg_enabled, self.sg_win, self.sg_order)
        self.cost_history.append(float(costs.mean()))

        # Best-action execution (iCEM §3.4)
        action = np.clip(best_elite[0], -lim, lim)
        self.U = np.roll(self.U, -1, axis=0); self.U[-1] = 0.0
        return action

    def get_rollout_trajs(self, state, n_disp=20):
        if self.last_samples is None: return [], [], [], 0, 0
        K        = self.K
        best_idx = int(np.argmin(self.last_costs))
        step     = max(1, K // n_disp)
        indices  = sorted(set(list(range(0, K, step))[:n_disp] + [best_idx]))
        trajs    = _traj_from_samples(self.env, state, self.last_samples, indices)
        costs    = [float(self.last_costs[ki]) for ki in indices]
        return trajs, costs, indices, indices.index(best_idx) if best_idx in indices else 0, 0
