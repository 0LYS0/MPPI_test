"""
=============================================================
  DIAL-MPC  ·  Diffusion-Inspired Annealing for Legged MPC
  2D Point Mass Simulator  ·  Gymnasium + Matplotlib

  Reference:
    Xue et al., "Full-Order Sampling-Based MPC for
    Torque-Level Locomotion Control via Diffusion-Style
    Annealing", arXiv:2409.15610, 2024.

  설치:
      pip install gymnasium numpy matplotlib

  실행:
      python dial_mpc_point_sim.py [--compare]

  옵션:
      --compare   MPPI vs DIAL-MPC 나란히 비교 모드
=============================================================
"""

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import warnings
warnings.filterwarnings("ignore")

import gymnasium as gym
from gymnasium import spaces


# ══════════════════════════════════════════════════════════
#  1.  Gymnasium 환경 – 2D Point Mass
# ══════════════════════════════════════════════════════════
class PointMass2DEnv(gym.Env):
    """
    State  : [x, y, vx, vy]
    Action : [ax, ay]   (가속도, 클립: ±2)
    Goal   : 장애물을 피해 목표점 도달
    """
    metadata = {"render_modes": [], "render_fps": 30}

    DT          = 0.05   # 타임스텝 [s]
    DRAG        = 0.15   # 감쇠 계수
    MAX_SPEED   = 4.0    # 최대 속도 제한
    BOUND       = 10.0   # 월드 경계

    def __init__(self, obstacles=None, goal=None, start=None):
        super().__init__()
        self.action_space = spaces.Box(
            low=-5.0, high=5.0, shape=(2,), dtype=np.float32
        )
        lim = np.array([self.BOUND, self.BOUND, self.MAX_SPEED, self.MAX_SPEED], dtype=np.float32)
        self.observation_space = spaces.Box(low=-lim, high=lim, dtype=np.float32)

        self.goal  = np.array(goal  if goal  is not None else [ 7.0,  7.0])
        self.start = np.array(start if start is not None else [-7.0, -7.0])
        self.obstacles = obstacles if obstacles is not None else [
            {"pos": np.array([2.5, 3.0]), "r": 1.0},
            {"pos": np.array([5.0, 1.0]), "r": 1.0},
            {"pos": np.array([1.5, 6.0]), "r": 0.9},
            {"pos": np.array([-2.5, -2.5]), "r": 2.1},
            {"pos": np.array([7.5, 3.0]), "r": 0.8},
        ]
        self.state      = np.zeros(4, dtype=np.float64)
        self._step_cnt  = 0

    # ── 리셋 ──────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state     = np.array([self.start[0], self.start[1], 0., 0.])
        self._step_cnt = 0
        return self.state.astype(np.float32), {}

    # ── 스텝 ──────────────────────────────────────────────
    def step(self, action):
        action = np.clip(action, -2.0, 2.0)
        x, y, vx, vy = self.state
        vx += action[0] * self.DT - self.DRAG * vx * self.DT
        vy += action[1] * self.DT - self.DRAG * vy * self.DT
        spd = np.hypot(vx, vy)
        if spd > self.MAX_SPEED:
            vx *= self.MAX_SPEED / spd
            vy *= self.MAX_SPEED / spd
        x += vx * self.DT;  y += vy * self.DT
        if abs(x) > self.BOUND: x = np.clip(x, -self.BOUND, self.BOUND); vx *= -0.5
        if abs(y) > self.BOUND: y = np.clip(y, -self.BOUND, self.BOUND); vy *= -0.5
        self.state = np.array([x, y, vx, vy])
        self._step_cnt += 1
        dist   = np.linalg.norm(self.state[:2] - self.goal)
        reward = -dist * 0.1
        for o in self.obstacles:
            d = np.linalg.norm(self.state[:2] - o["pos"]) - o["r"]
            if d < 0.3: reward -= (0.3 - max(d, 0)) * 5.0
        terminated = dist < 0.4
        truncated  = self._step_cnt >= 1200
        return self.state.astype(np.float32), reward, terminated, truncated, {}

    # ── 배치 동역학 (컨트롤러 내부 시뮬용) ───────────────
    def dynamics(self, state, action):
        """state [B,4], action [B,2] → next_state [B,4]"""
        action = np.clip(action, -2.0, 2.0)
        x, y, vx, vy = state[:,0], state[:,1], state[:,2], state[:,3]
        vx = vx + action[:,0]*self.DT - self.DRAG*vx*self.DT
        vy = vy + action[:,1]*self.DT - self.DRAG*vy*self.DT
        spd  = np.sqrt(vx**2 + vy**2)
        mask = spd > self.MAX_SPEED
        vx[mask] *= self.MAX_SPEED / spd[mask]
        vy[mask] *= self.MAX_SPEED / spd[mask]
        x = np.clip(x + vx*self.DT, -self.BOUND, self.BOUND)
        y = np.clip(y + vy*self.DT, -self.BOUND, self.BOUND)
        return np.stack([x, y, vx, vy], axis=1)

    # ── 비용 함수 ─────────────────────────────────────────
    def running_cost(self, state):
        """state [B,4] → cost [B]"""
        dist = np.linalg.norm(state[:,:2] - self.goal, axis=1)
        cost = dist * 2.0
        for o in self.obstacles:
            d = np.linalg.norm(state[:,:2] - o["pos"], axis=1) - o["r"]
            cost += np.where(d < 0.5, (0.5 - np.maximum(d, 0)) * 50.0, 0.0)
        return cost

    def terminal_cost(self, state):
        return np.linalg.norm(state[:,:2] - self.goal, axis=1) * 10.0


# ══════════════════════════════════════════════════════════
#  2.  MPPI 컨트롤러 (비교 기준선)
# ══════════════════════════════════════════════════════════
class MPPIController:
    """표준 MPPI (단일 노이즈 스케일, 단일 반복)"""

    def __init__(self, env, horizon=30, n_samples=512,
                 temperature=0.05, noise_sigma=0.8):
        self.env   = env
        self.H     = horizon
        self.K     = n_samples
        self.lam   = temperature
        self.sigma = noise_sigma
        self.U     = np.zeros((horizon, env.action_space.shape[0]))

        self.label         = "MPPI"
        self.cost_history  = []
        self.sigma_history = []   # 매 스텝 사용 sigma (단일값)

    def reset(self):
        self.U[:] = 0.0

    def compute_action(self, state):
        K, H    = self.K, self.H
        act_dim = self.env.action_space.shape[0]
        lo, hi  = self.env.action_space.low, self.env.action_space.high

        epsilon = np.random.randn(K, H, act_dim) * self.sigma
        states  = np.tile(state, (K, 1))
        costs   = np.zeros(K)
        for t in range(H):
            u = np.clip(self.U[t] + epsilon[:, t], lo, hi)
            states = self.env.dynamics(states, u)
            costs += self.env.running_cost(states)
        costs += self.env.terminal_cost(states)

        beta    = costs.min()
        weights = np.exp(-(costs - beta) / self.lam)
        weights /= weights.sum() + 1e-8

        for t in range(H):
            self.U[t] += np.sum(weights[:, None] * epsilon[:, t], axis=0)
        self.U = np.clip(self.U, lo, hi)

        self.cost_history.append(costs.mean())
        # 각 horizon step 에 같은 sigma 사용
        self.sigma_history.append(
            np.full(H, self.sigma)
        )

        action = self.U[0].copy()
        self.U[:-1] = self.U[1:]
        self.U[-1]  = 0.0
        return action, weights, epsilon


# ══════════════════════════════════════════════════════════
#  3.  DIAL-MPC 컨트롤러
# ══════════════════════════════════════════════════════════
class DIALMPCController:
    """
    DIAL-MPC (Diffusion-Inspired Annealing for Legged MPC)
    Xue et al., arXiv:2409.15610

    핵심 아이디어
    ─────────────
    MPPI 업데이트를 1회 수행하는 대신, N번의 *어닐링 이터레이션*을
    수행하며 노이즈 스케일을 점차 줄인다.

    이중 루프 공분산 설계 (eq. 7):
        Σ^i_{t+h} = exp[ -(N-i)/(β1·N)  ←trajectory  
                         -(H-h)/(β2·H) ] ←action  · I

    ‣ trajectory-level (외부 루프, i: N→1):
        초기에는 큰 노이즈로 넓은 탐색(coverage),
        후반에는 작은 노이즈로 정밀 수렴(convergence)

    ‣ action-level (내부, h: 0→H):
        가까운 미래(h=0)에는 정밀 노이즈,
        먼 미래(h=H)에는 큰 노이즈 → 업데이트 횟수 보상
    """

    def __init__(self, env, horizon=30, n_samples=512,
                 n_iterations=5, temperature=0.05,
                 sigma_base=1.2, beta1=0.5, beta2=0.5):
        self.env       = env
        self.H         = horizon
        self.K         = n_samples
        self.N         = n_iterations      # 어닐링 이터레이션 수
        self.lam       = temperature
        self.sigma_base = sigma_base
        self.beta1     = beta1             # trajectory-level 온도
        self.beta2     = beta2             # action-level 온도

        self.U = np.zeros((horizon, env.action_space.shape[0]))

        self.label         = "DIAL-MPC"
        self.cost_history  = []
        self.sigma_history = []            # [step] → [H] 마지막 이터레이션 sigmas

    def reset(self):
        self.U[:] = 0.0

    # ── 어닐링 스케줄 (eq. 7) ─────────────────────────────
    def _sigma_schedule(self, i):
        """
        이터레이션 i (1-indexed, N=가장 넓음) 에서
        각 horizon h 에 대한 노이즈 표준편차를 반환 [H,]
        """
        h_idx = np.arange(self.H, dtype=float)          # 0 … H-1
        traj_term   = (self.N - i) / (self.beta1 * self.N)
        action_term = (self.H - 1 - h_idx) / (self.beta2 * self.H)  # h=0 → (H-1)/β2H 크다, h=H-1 → 0
        log_sigma_sq = -(traj_term + action_term)
        return self.sigma_base * np.exp(log_sigma_sq / 2.0)          # std = sqrt(exp(...))

    # ── 메인 ─────────────────────────────────────────────
    def compute_action(self, state):
        K, H    = self.K, self.H
        act_dim = self.env.action_space.shape[0]
        lo, hi  = self.env.action_space.low, self.env.action_space.high

        last_weights = None
        last_epsilon = None
        all_costs    = []

        # ── N번 어닐링 이터레이션 (i = N → 1) ────────────
        for i in range(self.N, 0, -1):
            sigmas = self._sigma_schedule(i)          # [H]

            # 노이즈 샘플링 [K, H, act_dim]
            epsilon = np.random.randn(K, H, act_dim)
            epsilon *= sigmas[None, :, None]           # horizon별 스케일

            # 궤적 롤아웃 & 비용 계산
            states = np.tile(state, (K, 1))
            costs  = np.zeros(K)
            for t in range(H):
                u = np.clip(self.U[t] + epsilon[:, t], lo, hi)
                states = self.env.dynamics(states, u)
                costs += self.env.running_cost(states)
            costs += self.env.terminal_cost(states)

            # MPPI 스타일 가중치 및 업데이트
            beta    = costs.min()
            weights = np.exp(-(costs - beta) / self.lam)
            weights /= weights.sum() + 1e-8

            for t in range(H):
                self.U[t] += np.sum(weights[:, None] * epsilon[:, t], axis=0)
            self.U = np.clip(self.U, lo, hi)

            last_weights = weights
            last_epsilon = epsilon
            all_costs.append(costs.mean())

        # 통계 저장
        self.cost_history.append(np.mean(all_costs))
        self.sigma_history.append(self._sigma_schedule(1))  # 마지막 sigma 저장

        # Receding-horizon 시프트
        action = self.U[0].copy()
        self.U[:-1] = self.U[1:]
        self.U[-1]  = 0.0

        return action, last_weights, last_epsilon


# ══════════════════════════════════════════════════════════
#  4.  단독 실행 시각화
# ══════════════════════════════════════════════════════════
DARK_BG   = "#0d0d0d"
PANEL_BG  = "#111111"
GOAL_COL  = "#2ecc71"
AGENT_COL = "#ffffff"
OBS_COL   = "#e74c3c"
TRAJ_COL  = "#f39c12"
VEL_COL   = "#00d4ff"
SAMP_COL  = "#9b59b6"
MPPI_COL  = "#e67e22"
DIAL_COL  = "#1abc9c"


def _make_world_ax(ax, env, title="", title_color="white"):
    ax.set_facecolor(PANEL_BG)
    ax.set_xlim(-env.BOUND - 0.5, env.BOUND + 0.5)
    ax.set_ylim(-env.BOUND - 0.5, env.BOUND + 0.5)
    ax.set_aspect("equal")
    ax.set_title(title, color=title_color, fontsize=10)
    ax.tick_params(colors="gray", labelsize=7)
    for sp in ax.spines.values(): sp.set_color("#444"); sp.set_linewidth(0.5)
    ax.grid(True, color="#1e1e1e", linewidth=0.4, zorder=0)

    # 경계
    border = mpatches.Rectangle(
        (-env.BOUND, -env.BOUND), 2*env.BOUND, 2*env.BOUND,
        linewidth=1.2, edgecolor="#555", facecolor="none", zorder=1)
    ax.add_patch(border)

    # 장애물
    for o in env.obstacles:
        ax.add_patch(plt.Circle(o["pos"], o["r"], color=OBS_COL, alpha=0.75, zorder=2))
        ax.add_patch(plt.Circle(o["pos"], o["r"]+0.5, color=OBS_COL, alpha=0.12,
                                fill=False, linewidth=0.7, linestyle="--", zorder=2))

    # 목표
    ax.add_patch(plt.Circle(env.goal, 0.4,  color=GOAL_COL, alpha=0.9, zorder=3))
    ax.add_patch(plt.Circle(env.goal, 0.9,  color=GOAL_COL, alpha=0.18, zorder=2))
    ax.text(env.goal[0], env.goal[1]+1.0, "GOAL",
            color=GOAL_COL, ha="center", fontsize=7, fontweight="bold", zorder=5)

    # 시작
    ax.plot(env.start[0], env.start[1], "o", color="#3498db", ms=7, zorder=4)
    ax.text(env.start[0], env.start[1]-0.9, "START",
            color="#3498db", ha="center", fontsize=7)


class SingleViz:
    """DIAL-MPC 단독 시각화 (4-패널 레이아웃)"""

    def __init__(self, env, ctrl):
        self.env   = env
        self.ctrl  = ctrl
        self.fig   = plt.figure(figsize=(15, 9), facecolor=DARK_BG)
        self.fig.suptitle("DIAL-MPC  ·  Diffusion-Inspired Annealing MPC  (2D Point Mass)",
                          color="white", fontsize=13, fontweight="bold", y=0.97)
        gs = gridspec.GridSpec(3, 3, figure=self.fig,
                               left=0.05, right=0.97, top=0.93, bottom=0.07,
                               wspace=0.38, hspace=0.5)

        self.ax_world  = self.fig.add_subplot(gs[:, :2])
        self.ax_cost   = self.fig.add_subplot(gs[0, 2])
        self.ax_sigma  = self.fig.add_subplot(gs[1, 2])
        self.ax_anneal = self.fig.add_subplot(gs[2, 2])

        _make_world_ax(self.ax_world, env, "World (DIAL-MPC)")
        self._setup_panels()

        self.traj_x, self.traj_y = [], []
        self.step_cnt    = 0
        self.total_reward = 0.0

    def _setup_panels(self):
        for ax, title in [
            (self.ax_cost,   "Avg Sample Cost"),
            (self.ax_sigma,  "Noise σ per Horizon Step"),
            (self.ax_anneal, "Annealing Schedule (σ vs Iteration)"),
        ]:
            ax.set_facecolor(PANEL_BG)
            ax.set_title(title, color="white", fontsize=8.5)
            ax.tick_params(colors="gray", labelsize=7)
            for sp in ax.spines.values(): sp.set_color("#444"); sp.set_linewidth(0.5)
            ax.grid(True, color="#1e1e1e", linewidth=0.4)

        self.line_cost, = self.ax_cost.plot([], [], "-", color=DIAL_COL, lw=1.2)

        # 어닐링 시각화: sigma vs horizon for each iteration
        cmap = plt.get_cmap("plasma")
        H, N = self.ctrl.H, self.ctrl.N
        self.anneal_lines = []
        for i in range(N, 0, -1):
            color = cmap((N - i) / max(N - 1, 1))
            sigmas = self.ctrl._sigma_schedule(i)
            ln, = self.ax_anneal.plot(range(H), sigmas, "-",
                                      color=color, lw=1.0, alpha=0.8)
            self.anneal_lines.append(ln)
        self.ax_anneal.set_xlabel("Horizon step h", color="gray", fontsize=7)
        self.ax_anneal.set_ylabel("σ", color="gray", fontsize=7)
        sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=1, vmax=N))
        sm.set_array([])
        cb = self.fig.colorbar(sm, ax=self.ax_anneal, pad=0.02)
        cb.set_label("Iteration i", color="gray", fontsize=7)
        cb.ax.tick_params(colors="gray", labelsize=6)

        # sigma 현재값 라인
        self.sigma_line, = self.ax_sigma.plot(
            range(self.ctrl.H),
            self.ctrl._sigma_schedule(1),
            "-o", color=DIAL_COL, lw=1.2, ms=3)
        self.ax_sigma.set_xlabel("Horizon step h", color="gray", fontsize=7)
        self.ax_sigma.set_ylabel("σ (last iter)", color="gray", fontsize=7)

        # World 요소
        self.line_traj,  = self.ax_world.plot([], [], "-", color=TRAJ_COL, lw=1.2, alpha=0.7, zorder=5)
        self.agent_dot,  = self.ax_world.plot([], [], "o", color=AGENT_COL, ms=11, zorder=8,
                                              mec=TRAJ_COL, mew=2)
        self.vel_ann     = self.ax_world.annotate("", xy=(0,0), xytext=(0,0),
                                                  arrowprops=dict(arrowstyle="->", color=VEL_COL, lw=1.8), zorder=7)
        self.samp_lines  = [self.ax_world.plot([], [], "-", color=SAMP_COL,
                                               alpha=0.10, lw=0.7, zorder=3)[0] for _ in range(50)]
        self.info_text   = self.ax_world.text(
            -self.env.BOUND+0.3, self.env.BOUND-0.4, "",
            color="white", fontsize=8.5, va="top",
            bbox=dict(boxstyle="round,pad=0.4", fc="#1a1a2e", alpha=0.85), zorder=10)

    def update(self, state, action, weights, epsilon, reward, done):
        x, y, vx, vy = state
        self.traj_x.append(x); self.traj_y.append(y)
        self.line_traj.set_data(self.traj_x, self.traj_y)
        self.agent_dot.set_data([x], [y])

        # 속도 화살표
        self.vel_ann.remove()
        self.vel_ann = self.ax_world.annotate(
            "", xy=(x + vx*1.2, y + vy*1.2), xytext=(x, y),
            arrowprops=dict(arrowstyle="->", color=VEL_COL, lw=2.0), zorder=7)

        # 상위 샘플 궤적
        # top_idx = np.argsort(weights)[-25:][::-1]
        top_idx = np.arange(0,len(weights),len(weights)/50)
        for li, idx in enumerate(top_idx):
            idx = int(idx)
            sx, sy = [x], [y]
            s = np.tile(state, (1, 1))
            for t in range(min(self.ctrl.H, 18)):
                u = np.clip(self.ctrl.U[t] + epsilon[idx, t], -2, 2).reshape(1,-1)
                s = self.env.dynamics(s, u)
                sx.append(s[0,0]); sy.append(s[0,1])
            self.samp_lines[li].set_data(sx, sy)

        # 비용 그래프
        n = len(self.ctrl.cost_history)
        self.line_cost.set_data(range(n), self.ctrl.cost_history)
        self.ax_cost.relim(); self.ax_cost.autoscale_view()

        # sigma 그래프 (마지막 이터레이션)
        if self.ctrl.sigma_history:
            self.sigma_line.set_ydata(self.ctrl.sigma_history[-1])
            self.ax_sigma.relim(); self.ax_sigma.autoscale_view()

        # 정보 텍스트
        self.total_reward += reward; self.step_cnt += 1
        dist = np.linalg.norm(state[:2] - self.env.goal)
        self.info_text.set_text(
            f" N_iter  : {self.ctrl.N}  β1={self.ctrl.beta1}  β2={self.ctrl.beta2}\n"
            f" Step    : {self.step_cnt:4d}\n"
            f" Pos     : ({x:5.2f}, {y:5.2f})\n"
            f" Dist    : {dist:.3f}\n"
            f" Reward  : {reward:+.3f}  Σ{self.total_reward:+.1f}")
        if done:
            self.ax_world.set_title("World (DIAL-MPC)  ✓ GOAL!", color=GOAL_COL, fontsize=10)


# ══════════════════════════════════════════════════════════
#  5.  비교 시각화 (MPPI vs DIAL-MPC 나란히)
# ══════════════════════════════════════════════════════════
class CompareViz:
    """두 컨트롤러를 나란히 비교하는 시각화"""

    def __init__(self, env_mppi, env_dial, ctrl_mppi, ctrl_dial):
        self.envs  = [env_mppi,  env_dial]
        self.ctrls = [ctrl_mppi, ctrl_dial]
        self.colors = [MPPI_COL, DIAL_COL]
        self.labels = ["MPPI",  "DIAL-MPC"]

        self.fig = plt.figure(figsize=(16, 9), facecolor=DARK_BG)
        self.fig.suptitle("MPPI  vs  DIAL-MPC  ·  2D Point Mass",
                          color="white", fontsize=13, fontweight="bold", y=0.97)
        gs = gridspec.GridSpec(3, 4, figure=self.fig,
                               left=0.04, right=0.97, top=0.93, bottom=0.07,
                               wspace=0.35, hspace=0.50)

        self.ax_worlds = [self.fig.add_subplot(gs[:, 0:2]),
                          self.fig.add_subplot(gs[:, 2:4])]
        # 우측 패널: 없음 (비교를 위해 두 월드를 크게)

        for ax, env, lbl, col in zip(self.ax_worlds, self.envs,
                                     self.labels, self.colors):
            _make_world_ax(ax, env, lbl, col)

        # 비교 패널 (두 월드 위에 오버레이하는 대신, fig 전체 하단에 작게)
        gs2 = gridspec.GridSpec(1, 2, figure=self.fig,
                                left=0.04, right=0.97, top=0.20, bottom=0.07,
                                wspace=0.30)

        self._traj_x  = [[], []]
        self._traj_y  = [[], []]
        self._rewards = [0.0, 0.0]
        self._steps   = [0, 0]
        self._done    = [False, False]

        # World 드로잉 요소
        self.line_traj  = [None, None]
        self.agent_dot  = [None, None]
        self.vel_ann    = [None, None]
        self.samp_lines = [None, None]
        self.info_text  = [None, None]

        for k in range(2):
            ax = self.ax_worlds[k]
            col = self.colors[k]
            self.line_traj[k], = ax.plot([], [], "-", color=col, lw=1.2, alpha=0.7, zorder=5)
            self.agent_dot[k], = ax.plot([], [], "o", color=AGENT_COL, ms=11, zorder=8,
                                         mec=col, mew=2)
            self.vel_ann[k] = ax.annotate("", xy=(0,0), xytext=(0,0),
                                          arrowprops=dict(arrowstyle="->", color=VEL_COL, lw=1.8), zorder=7)
            self.samp_lines[k] = [ax.plot([], [], "-", color=SAMP_COL,
                                          alpha=0.09, lw=0.7, zorder=3)[0] for _ in range(50)]
            self.info_text[k] = ax.text(
                -self.envs[k].BOUND+0.3, self.envs[k].BOUND-0.4, "",
                color="white", fontsize=8, va="top",
                bbox=dict(boxstyle="round,pad=0.4", fc="#1a1a2e", alpha=0.85), zorder=10)

    def update(self, k, state, weights, epsilon, reward, done):
        env, ctrl = self.envs[k], self.ctrls[k]
        ax = self.ax_worlds[k]
        x, y, vx, vy = state
        self._traj_x[k].append(x); self._traj_y[k].append(y)
        self.line_traj[k].set_data(self._traj_x[k], self._traj_y[k])
        self.agent_dot[k].set_data([x], [y])

        self.vel_ann[k].remove()
        self.vel_ann[k] = ax.annotate(
            "", xy=(x + vx*1.2, y + vy*1.2), xytext=(x, y),
            arrowprops=dict(arrowstyle="->", color=VEL_COL, lw=2.0), zorder=7)

        # top_idx = np.argsort(weights)[-20:][::-1]
        top_idx = np.arange(0,len(weights),len(weights)/50)
        for li, idx in enumerate(top_idx):
            idx = int(idx)
            sx, sy = [x], [y]
            s = np.tile(state, (1,1))
            for t in range(min(ctrl.H, 16)):
                u = np.clip(ctrl.U[t] + epsilon[idx, t], -2, 2).reshape(1,-1)
                s = env.dynamics(s, u)
                sx.append(s[0,0]); sy.append(s[0,1])
            self.samp_lines[k][li].set_data(sx, sy)

        self._rewards[k] += reward; self._steps[k] += 1
        dist = np.linalg.norm(state[:2] - env.goal)
        label = self.labels[k]
        extra = ""
        if isinstance(ctrl, DIALMPCController):
            extra = f"\n N_iter={ctrl.N}  β1={ctrl.beta1}  β2={ctrl.beta2}"
        self.info_text[k].set_text(
            f" {label}{extra}\n"
            f" Step    : {self._steps[k]:4d}\n"
            f" Dist    : {dist:.3f}\n"
            f" Reward  : {reward:+.3f}  Σ{self._rewards[k]:+.1f}")
        if done and not self._done[k]:
            ax.set_title(f"{label}  ✓ GOAL!  ({self._steps[k]} steps)",
                         color=GOAL_COL, fontsize=10)
            self._done[k] = True


# ══════════════════════════════════════════════════════════
#  6.  메인 실행
# ══════════════════════════════════════════════════════════
def run_single():
    env  = PointMass2DEnv()
    ctrl = DIALMPCController(
        env,
        horizon      = 30,
        n_samples    = 512,
        n_iterations = 6,          # 어닐링 이터레이션 수 N
        temperature  = 0.05,
        sigma_base   = 2.5,
        beta1        = 0.6,        # trajectory-level 온도
        beta2        = 0.5,        # action-level 온도
    )
    state, _ = env.reset(); ctrl.reset()
    viz = SingleViz(env, ctrl)

    print("=" * 55)
    print("  DIAL-MPC  ·  2D Point Mass  (단독 실행)")
    print("=" * 55)
    print(f"  N_iterations = {ctrl.N}, H = {ctrl.H}, K = {ctrl.K}")
    print(f"  σ_base={ctrl.sigma_base}  β1={ctrl.beta1}  β2={ctrl.beta2}")
    print("=" * 55)

    done_flag = [False]
    def animate(_):
        if done_flag[0]: return
        action, weights, epsilon = ctrl.compute_action(state)
        ns, reward, term, trunc, _ = env.step(action)
        done = term or trunc
        viz.update(state, action, weights, epsilon, reward, done)
        state[:] = ns
        if done:
            tag = "✓ 목표 도달" if term else "✗ 타임아웃"
            print(f"\n  {tag}  |  Steps: {viz.step_cnt}  |  ΣReward: {viz.total_reward:.1f}")
            done_flag[0] = True

    ani=FuncAnimation(viz.fig, animate, interval=60, cache_frame_data=False, blit=False)
    plt.show()


def run_compare():
    # 같은 장애물/목표 환경을 두 개 생성
    kwargs = {}
    env_m = PointMass2DEnv(**kwargs)
    env_d = PointMass2DEnv(**kwargs)

    ctrl_m = MPPIController(
        env_m, horizon=35, n_samples=1024, temperature=0.05, noise_sigma=2.5
    )
    ctrl_d = DIALMPCController(
        env_d, horizon=35, n_samples=1024, n_iterations=3,
        temperature=0.05, sigma_base=2.5, beta1=0.6, beta2=0.5
    )

    state_m, _ = env_m.reset(); ctrl_m.reset()
    state_d, _ = env_d.reset(); ctrl_d.reset()
    viz = CompareViz(env_m, env_d, ctrl_m, ctrl_d)

    print("=" * 55)
    print("  MPPI  vs  DIAL-MPC  비교 모드")
    print("=" * 55)

    done_m, done_d = [False], [False]

    def animate(_):
        # MPPI
        if not done_m[0]:
            a, w, e = ctrl_m.compute_action(state_m)
            ns, r, t, tr, _ = env_m.step(a)
            d = t or tr
            viz.update(0, state_m, w, e, r, d)
            state_m[:] = ns
            if d:
                tag = "✓ 목표 도달" if t else "✗ 타임아웃"
                print(f"  [MPPI]     {tag}  | steps={viz._steps[0]}")
                done_m[0] = True

        # DIAL-MPC
        if not done_d[0]:
            a, w, e = ctrl_d.compute_action(state_d)
            ns, r, t, tr, _ = env_d.step(a)
            d = t or tr
            viz.update(1, state_d, w, e, r, d)
            state_d[:] = ns
            if d:
                tag = "✓ 목표 도달" if t else "✗ 타임아웃"
                print(f"  [DIAL-MPC] {tag}  | steps={viz._steps[1]}")
                done_d[0] = True

    ani = FuncAnimation(viz.fig, animate, interval=60, cache_frame_data=False, blit=False)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare", action="store_true",
                        help="MPPI vs DIAL-MPC 나란히 비교 모드")
    args = parser.parse_args()

    if args.compare:
        run_compare()
    else:
        run_single()
