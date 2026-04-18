"""
controllers.py — MPPI and DIAL-MPC controllers.

Both controllers:
  • run the sampling/optimisation loop
  • store rollout data for the frontend (last eps, weights, costs)
  • expose get_rollout_trajs() to send pre-computed trajectories to the client
"""
import numpy as np
from scipy.signal import savgol_filter
from env import PointMass2DEnv


# ═══════════════════════════════════════════════════════
#  MPPI
# ═══════════════════════════════════════════════════════
class MPPIController:
    def __init__(self, env: PointMass2DEnv, *,
                 H=30, K=400, lam=0.05, sigma=0.9, alpha=1.0,
                 sg_enabled=False, sg_win=7, sg_order=3, seed=42):
        self.env = env
        self.H, self.K   = H, K
        self.lam, self.sigma, self.alpha = lam, sigma, alpha
        self.sg_enabled  = sg_enabled
        self.sg_win, self.sg_order = sg_win, sg_order
        self.rng         = np.random.default_rng(seed)
        self.U           = np.zeros((H, 2))          # nominal control sequence
        # visualisation data
        self.last_weights = None
        self.last_eps     = None
        self.last_costs   = None
        self.cost_history: list[float] = []

    def reset(self, seed: int | None = None):
        self.U[:] = 0.0
        self.last_weights = self.last_eps = self.last_costs = None
        self.cost_history.clear()
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    # ── main compute ──────────────────────────────────────
    def compute(self, state: np.ndarray) -> np.ndarray:
        H, K          = self.H, self.K
        lam, sigma    = self.lam, self.sigma
        alpha         = self.alpha
        gamma         = lam * (1.0 - alpha)
        inv_sig2      = 1.0 / (sigma ** 2)
        n_explore     = int((1.0 - alpha) * K)
        lim           = self.env.action_lim()
        sd            = self.env.state_dim()

        # noise [K, H, 2]
        eps    = self.rng.standard_normal((K, H, 2)) * sigma
        states = np.tile(state, (K, 1))
        costs  = np.zeros(K)
        all_actions = np.empty((K, H, 2))   # Bug fix ②: store actual clamped actions

        for t in range(H):
            u0, u1 = self.U[t, 0], self.U[t, 1]
            actions = np.empty((K, 2))
            # exploration: pure noise
            actions[:n_explore, 0] = np.clip(eps[:n_explore, t, 0], -lim, lim)
            actions[:n_explore, 1] = np.clip(eps[:n_explore, t, 1], -lim, lim)
            # warm-start: U + noise
            actions[n_explore:, 0] = np.clip(u0 + eps[n_explore:, t, 0], -lim, lim)
            actions[n_explore:, 1] = np.clip(u1 + eps[n_explore:, t, 1], -lim, lim)
            all_actions[:, t, :] = actions  # save for update

            states = self.env.batch_step(states, actions)
            rc     = self.env.running_cost(states)
            if gamma > 0:
                costs += rc + gamma * inv_sig2 * (u0 * eps[:, t, 0] + u1 * eps[:, t, 1])
            else:
                costs += rc

        costs += self.env.terminal_cost(states)
        self.last_costs = costs.copy()

        # weights
        beta    = costs.min()
        weights = np.exp(-(costs - beta) / lam)
        weights /= weights.sum() + 1e-8

        # Bug fix ②: U = weighted average of actual actions (correct under clamping)
        # Standard MPPI: U += Σ w_k ε_k  assumes no clamping (ε_k = v_k − U).
        # With clamping the actual perturbation ≠ ε_k, so use the actual actions directly.
        w_actions = (weights[:, None, None] * all_actions).sum(axis=0)  # [H, 2]
        self.U = np.clip(w_actions, -lim, lim)
        self._apply_sg(lim)

        self.last_weights = weights
        self.last_eps     = eps
        self.last_U_vis   = self.U.copy()  # Bug fix ③: save pre-shift U for visualisation
        self.cost_history.append(float(costs.mean()))

        action  = self.U[0].copy()
        self.U  = np.roll(self.U, -1, axis=0)
        self.U[-1] = 0.0
        return action

    def _apply_sg(self, lim: float):
        if not self.sg_enabled:
            return
        win = self.sg_win if self.sg_win % 2 == 1 else self.sg_win + 1
        order = min(self.sg_order, win - 1)
        if win < 3 or order < 1 or self.H < win:
            return
        try:
            self.U = np.clip(
                savgol_filter(self.U, window_length=win, polyorder=order, axis=0),
                -lim, lim
            )
        except Exception:
            pass

    # ── rollout trajectories for visualisation ────────────
    def get_rollout_trajs(self, state: np.ndarray, n_disp: int = 20):
        """Return (trajs, costs, indices, best_pos_in_indices)."""
        if self.last_weights is None:
            return [], [], [], 0
        K, H    = self.K, self.H
        weights = self.last_weights
        eps     = self.last_eps
        costs   = self.last_costs if self.last_costs is not None else np.zeros(K)
        n_exp   = int((1.0 - self.alpha) * K)
        lim     = self.env.action_lim()
        is_1st  = self.env.dyn_model == "1st"
        # Bug fix ③: use pre-shift U so rollout matches the plan that was just executed
        U_vis   = getattr(self, "last_U_vis", self.U)

        best_idx = int(np.argmax(weights))
        n_half   = n_disp // 2
        exp_idx  = (np.linspace(0, max(n_exp - 1, 0), n_half, dtype=int)
                    if n_exp > 0 else np.array([], dtype=int))
        ws_cnt   = K - n_exp
        ws_idx   = (np.linspace(n_exp, K - 1, n_half, dtype=int)
                    if ws_cnt > 0 else np.array([], dtype=int))
        indices  = np.unique(np.concatenate([exp_idx, ws_idx, [best_idx]])).tolist()
        indices  = [i for i in indices if 0 <= i < K]

        trajs, traj_costs = [], []
        for ki in indices:
            is_explore = ki < n_exp
            traj = [state[:2].tolist()]
            if is_1st:
                sx, sy = float(state[0]), float(state[1])
            else:
                sx, sy = float(state[0]), float(state[1])
                svx, svy = float(state[2]), float(state[3])

            for t in range(H):
                e0, e1 = float(eps[ki, t, 0]), float(eps[ki, t, 1])
                if is_explore:
                    ax = float(np.clip(e0, -lim, lim))
                    ay = float(np.clip(e1, -lim, lim))
                else:
                    ax = float(np.clip(U_vis[t, 0] + e0, -lim, lim))
                    ay = float(np.clip(U_vis[t, 1] + e1, -lim, lim))

                if is_1st:
                    sx = float(np.clip(sx + ax * self.env.DT, -self.env.BOUND, self.env.BOUND))
                    sy = float(np.clip(sy + ay * self.env.DT, -self.env.BOUND, self.env.BOUND))
                else:
                    svx += ax * self.env.DT - self.env.DRAG * svx * self.env.DT
                    svy += ay * self.env.DT - self.env.DRAG * svy * self.env.DT
                    spd = (svx ** 2 + svy ** 2) ** 0.5
                    if spd > self.env.max_speed:
                        svx *= self.env.max_speed / spd
                        svy *= self.env.max_speed / spd
                    sx = float(np.clip(sx + svx * self.env.DT, -self.env.BOUND, self.env.BOUND))
                    sy = float(np.clip(sy + svy * self.env.DT, -self.env.BOUND, self.env.BOUND))
                traj.append([sx, sy])
            trajs.append(traj)
            traj_costs.append(float(costs[ki]))

        best_pos = indices.index(best_idx) if best_idx in indices else 0
        return trajs, traj_costs, indices, best_pos


# ═══════════════════════════════════════════════════════
#  DIAL-MPC
# ═══════════════════════════════════════════════════════
class DIALController:
    """Diffusion-Inspired Annealing for Legged MPC (Xue et al., 2409.15610)."""
    def __init__(self, env: PointMass2DEnv, *,
                 H=30, K=400, N=5, lam=0.05,
                 sigma_base=1.2, beta1=0.6, beta2=0.5, alpha=1.0,
                 sg_enabled=False, sg_win=7, sg_order=3, seed=42):
        self.env = env
        self.H, self.K, self.N = H, K, N
        self.lam, self.alpha   = lam, alpha
        self.sigma_base        = sigma_base
        self.beta1, self.beta2 = beta1, beta2
        self.sg_enabled        = sg_enabled
        self.sg_win, self.sg_order = sg_win, sg_order
        self.rng               = np.random.default_rng(seed)
        self.U                 = np.zeros((H, 2))
        # visualisation data
        self.last_weights = None
        self.last_eps     = None
        self.last_costs   = None
        self.last_sigmas  = None
        self.cost_history: list[float] = []

    def reset(self, seed: int | None = None):
        self.U[:] = 0.0
        self.last_weights = self.last_eps = self.last_costs = self.last_sigmas = None
        self.cost_history.clear()
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def _sigma_schedule(self, i: int) -> np.ndarray:
        """eq.7: Σ^i_{t+h} = exp[-(N-i)/(β1·N) - (H-1-h)/(β2·H)] · I"""
        H, N, b1, b2, sb = self.H, self.N, self.beta1, self.beta2, self.sigma_base
        h = np.arange(H, dtype=float)
        traj_term   = (N - i) / (b1 * N)
        action_term = (H - 1 - h) / (b2 * H)
        return sb * np.exp(-(traj_term + action_term) / 2.0)

    # ── main compute ──────────────────────────────────────
    def compute(self, state: np.ndarray) -> np.ndarray:
        H, K, N   = self.H, self.K, self.N
        lam       = self.lam
        alpha     = self.alpha
        gamma     = lam * (1.0 - alpha)
        n_explore = int((1.0 - alpha) * K)
        sd        = self.env.state_dim()
        total_cost = 0.0
        lW = lE = None

        for i in range(N, 0, -1):
            sigmas = self._sigma_schedule(i)      # [H]
            eps    = self.rng.standard_normal((K, H, 2)) * sigmas[None, :, None]

            states      = np.tile(state, (K, 1))
            costs       = np.zeros(K)
            all_actions = np.empty((K, H, 2))  # Bug fix ②

            for t in range(H):
                u0, u1  = self.U[t, 0], self.U[t, 1]
                sv      = float(sigmas[t])
                inv_s2  = 1.0 / (sv ** 2 + 1e-12)
                lim     = self.env.action_lim()

                actions = np.empty((K, 2))
                actions[:n_explore, 0] = np.clip(eps[:n_explore, t, 0], -lim, lim)
                actions[:n_explore, 1] = np.clip(eps[:n_explore, t, 1], -lim, lim)
                actions[n_explore:, 0] = np.clip(u0 + eps[n_explore:, t, 0], -lim, lim)
                actions[n_explore:, 1] = np.clip(u1 + eps[n_explore:, t, 1], -lim, lim)
                all_actions[:, t, :] = actions  # save

                states = self.env.batch_step(states, actions)
                rc     = self.env.running_cost(states)
                if gamma > 0:
                    costs += rc + gamma * inv_s2 * (u0 * eps[:, t, 0] + u1 * eps[:, t, 1])
                else:
                    costs += rc

            costs += self.env.terminal_cost(states)
            if i == 1:
                self.last_costs  = costs.copy()
                self.last_sigmas = sigmas

            beta    = costs.min()
            weights = np.exp(-(costs - beta) / lam)
            weights /= weights.sum() + 1e-8

            lim = self.env.action_lim()
            # Bug fix ②: weighted average of actual (clamped) actions
            w_actions = (weights[:, None, None] * all_actions).sum(axis=0)  # [H, 2]
            self.U = np.clip(w_actions, -lim, lim)
            self._apply_sg(lim)

            total_cost += float(costs.mean())
            lW, lE = weights, eps

        self.last_weights = lW
        self.last_eps     = lE
        self.last_U_vis   = self.U.copy()  # Bug fix ③: save pre-shift U
        self.cost_history.append(total_cost / N)

        action  = self.U[0].copy()
        self.U  = np.roll(self.U, -1, axis=0)
        self.U[-1] = 0.0
        return action

    def _apply_sg(self, lim: float):
        if not self.sg_enabled:
            return
        win   = self.sg_win if self.sg_win % 2 == 1 else self.sg_win + 1
        order = min(self.sg_order, win - 1)
        if win < 3 or order < 1 or self.H < win:
            return
        try:
            self.U = np.clip(
                savgol_filter(self.U, window_length=win, polyorder=order, axis=0),
                -lim, lim
            )
        except Exception:
            pass

    def get_rollout_trajs(self, state: np.ndarray, n_disp: int = 20):
        """Delegate to same logic as MPPI (uses last_eps / last_weights)."""
        if self.last_weights is None:
            return [], [], [], 0
        K, H    = self.K, self.H
        weights = self.last_weights
        eps     = self.last_eps
        costs   = self.last_costs if self.last_costs is not None else np.zeros(K)
        n_exp   = int((1.0 - self.alpha) * K)
        lim     = self.env.action_lim()
        is_1st  = self.env.dyn_model == "1st"
        # Bug fix ③: use pre-shift U for accurate visualisation
        U_vis   = getattr(self, "last_U_vis", self.U)

        best_idx = int(np.argmax(weights))
        n_half   = n_disp // 2
        exp_idx  = (np.linspace(0, max(n_exp - 1, 0), n_half, dtype=int)
                    if n_exp > 0 else np.array([], dtype=int))
        ws_cnt   = K - n_exp
        ws_idx   = (np.linspace(n_exp, K - 1, n_half, dtype=int)
                    if ws_cnt > 0 else np.array([], dtype=int))
        indices  = np.unique(np.concatenate([exp_idx, ws_idx, [best_idx]])).tolist()
        indices  = [i for i in indices if 0 <= i < K]

        trajs, traj_costs = [], []
        for ki in indices:
            is_explore = ki < n_exp
            traj = [state[:2].tolist()]
            if is_1st:
                sx, sy = float(state[0]), float(state[1])
            else:
                sx, sy = float(state[0]), float(state[1])
                svx, svy = float(state[2]), float(state[3])

            for t in range(H):
                e0, e1 = float(eps[ki, t, 0]), float(eps[ki, t, 1])
                if is_explore:
                    ax = float(np.clip(e0, -lim, lim))
                    ay = float(np.clip(e1, -lim, lim))
                else:
                    ax = float(np.clip(U_vis[t, 0] + e0, -lim, lim))
                    ay = float(np.clip(U_vis[t, 1] + e1, -lim, lim))

                if is_1st:
                    sx = float(np.clip(sx + ax * self.env.DT, -self.env.BOUND, self.env.BOUND))
                    sy = float(np.clip(sy + ay * self.env.DT, -self.env.BOUND, self.env.BOUND))
                else:
                    svx += ax * self.env.DT - self.env.DRAG * svx * self.env.DT
                    svy += ay * self.env.DT - self.env.DRAG * svy * self.env.DT
                    spd  = (svx ** 2 + svy ** 2) ** 0.5
                    if spd > self.env.max_speed:
                        svx *= self.env.max_speed / spd
                        svy *= self.env.max_speed / spd
                    sx = float(np.clip(sx + svx * self.env.DT, -self.env.BOUND, self.env.BOUND))
                    sy = float(np.clip(sy + svy * self.env.DT, -self.env.BOUND, self.env.BOUND))
                traj.append([sx, sy])
            trajs.append(traj)
            traj_costs.append(float(costs[ki]))

        best_pos = indices.index(best_idx) if best_idx in indices else 0
        return trajs, traj_costs, indices, best_pos

    def sigma_schedule_list(self, i: int = 1) -> list:
        return self._sigma_schedule(i).tolist()
