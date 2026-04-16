"""
=============================================================
  2D Point Mass Simulator  +  MPPI Controller
  - Gymnasium custom environment
  - MPPI (Model Predictive Path Integral) control
  - Matplotlib real-time visualization
=============================================================

설치:
    pip install gymnasium numpy matplotlib

실행:
    python mppi_point_sim.py
=============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
import gymnasium as gym
from gymnasium import spaces
import warnings
warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────
# 1.  Gymnasium 환경: 2D Point Mass
# ──────────────────────────────────────────────────────────
class PointMass2DEnv(gym.Env):
    """
    2D 점 질량 환경.

    State  : [x, y, vx, vy]
    Action : [ax, ay]  (가속도)
    Goal   : 목표 지점(goal)에 도달하고 장애물 회피

    관측 공간:
        x, y  ∈ [-10, 10]
        vx, vy ∈ [-5, 5]

    행동 공간:
        ax, ay ∈ [-2, 2]
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    # 물리 파라미터
    DT          = 0.05   # 타임스텝 [s]
    DRAG        = 0.15   # 감쇠 계수
    MAX_SPEED   = 4.0    # 최대 속도 제한
    BOUND       = 10.0   # 월드 경계

    def __init__(self, render_mode=None, obstacles=None, goal=None):
        super().__init__()

        # ── 행동 / 관측 공간 ──
        self.action_space = spaces.Box(
            low=-5.0, high=5.0, shape=(2,), dtype=np.float32
        )
        obs_low  = np.array([-self.BOUND, -self.BOUND, -self.MAX_SPEED, -self.MAX_SPEED], dtype=np.float32)
        obs_high = np.array([ self.BOUND,  self.BOUND,  self.MAX_SPEED,  self.MAX_SPEED], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # ── 목표 & 장애물 ──
        self.goal = np.array(goal if goal is not None else [7.0, 7.0], dtype=np.float64)
        self.obstacles = obstacles if obstacles is not None else [
            {"pos": np.array([3.0, 3.0]), "r": 1.2},
            {"pos": np.array([5.0, 1.5]), "r": 1.0},
            {"pos": np.array([1.5, 6.0]), "r": 0.9},
            {"pos": np.array([-2.5, -2.5]), "r": 2.1},
            {"pos": np.array([7.5, 3.0]), "r": 0.8},
        ]

        # 초기 상태
        self.state = np.zeros(4, dtype=np.float64)
        self.render_mode = render_mode
        self._step_count = 0

    # ── 리셋 ──
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.array([-7.0, -7.0, 0.0, 0.0], dtype=np.float64)
        self._step_count = 0
        return self.state.astype(np.float32), {}

    # ── 스텝 ──
    def step(self, action):
        action = np.clip(action, -2.0, 2.0)
        x, y, vx, vy = self.state

        # 오일러 적분
        vx = vx + action[0] * self.DT - self.DRAG * vx * self.DT
        vy = vy + action[1] * self.DT - self.DRAG * vy * self.DT

        # 속도 제한
        speed = np.sqrt(vx**2 + vy**2)
        if speed > self.MAX_SPEED:
            vx *= self.MAX_SPEED / speed
            vy *= self.MAX_SPEED / speed

        x = x + vx * self.DT
        y = y + vy * self.DT

        # 경계 처리 (반발)
        if abs(x) > self.BOUND:
            x = np.clip(x, -self.BOUND, self.BOUND)
            vx *= -0.5
        if abs(y) > self.BOUND:
            y = np.clip(y, -self.BOUND, self.BOUND)
            vy *= -0.5

        self.state = np.array([x, y, vx, vy])
        self._step_count += 1

        # 보상 계산
        dist_goal = np.linalg.norm(self.state[:2] - self.goal)
        reward     = -dist_goal * 0.1

        # 장애물 패널티
        for obs in self.obstacles:
            d = np.linalg.norm(self.state[:2] - obs["pos"]) - obs["r"]
            if d < 0.3:
                reward -= (0.3 - max(d, 0)) * 5.0

        # 종료 조건
        terminated = dist_goal < 0.4
        truncated  = self._step_count >= 1000

        return self.state.astype(np.float32), reward, terminated, truncated, {}

    # ── 동역학 함수 (MPPI 내부 시뮬레이션용) ──
    def dynamics(self, state, action):
        """배치 동역학: state [B, 4], action [B, 2] → next_state [B, 4]"""
        action = np.clip(action, -2.0, 2.0)
        x,  y  = state[:, 0], state[:, 1]
        vx, vy = state[:, 2], state[:, 3]

        vx = vx + action[:, 0] * self.DT - self.DRAG * vx * self.DT
        vy = vy + action[:, 1] * self.DT - self.DRAG * vy * self.DT

        speed = np.sqrt(vx**2 + vy**2)
        mask  = speed > self.MAX_SPEED
        vx[mask] *= self.MAX_SPEED / speed[mask]
        vy[mask] *= self.MAX_SPEED / speed[mask]

        x = np.clip(x + vx * self.DT, -self.BOUND, self.BOUND)
        y = np.clip(y + vy * self.DT, -self.BOUND, self.BOUND)

        return np.stack([x, y, vx, vy], axis=1)

    # ── 비용 함수 (MPPI용) ──
    def running_cost(self, state):
        """state [B, 4] → cost [B]"""
        dist_goal = np.linalg.norm(state[:, :2] - self.goal, axis=1)
        cost = dist_goal * 2.0

        for obs in self.obstacles:
            d = np.linalg.norm(state[:, :2] - obs["pos"], axis=1) - obs["r"]
            cost += np.where(d < 0.5, (0.5 - np.maximum(d, 0)) * 50.0, 0.0)

        return cost

    def terminal_cost(self, state):
        """state [B, 4] → cost [B]"""
        return np.linalg.norm(state[:, :2] - self.goal, axis=1) * 10.0


# ──────────────────────────────────────────────────────────
# 2.  MPPI 컨트롤러
# ──────────────────────────────────────────────────────────
class MPPIController:
    """
    Model Predictive Path Integral (MPPI) 컨트롤러.

    핵심 아이디어:
      1. K개의 랜덤 노이즈 시퀀스를 샘플링
      2. 각 샘플 궤적의 비용을 계산
      3. 비용 기반 소프트맥스 가중치로 제어 입력 업데이트
    """

    def __init__(
        self,
        env,
        horizon    = 30,     # 예측 지평선 (타임스텝)
        n_samples  = 512,    # 샘플 수
        temperature= 0.05,   # 소프트맥스 온도 λ  (낮을수록 좋은 샘플에 집중)
        noise_sigma= 0.8,    # 제어 노이즈 표준편차
    ):
        self.env        = env
        self.H          = horizon
        self.K          = n_samples
        self.lam        = temperature
        self.sigma      = noise_sigma

        act_dim = env.action_space.shape[0]
        self.U  = np.zeros((horizon, act_dim))   # 현재 제어 시퀀스

        # 통계 기록
        self.cost_history    = []
        self.weight_entropy  = []

    def reset(self):
        self.U[:] = 0.0

    def compute_action(self, state):
        """현재 상태에서 MPPI 최적 행동을 계산."""
        K, H       = self.K, self.H
        act_dim    = self.env.action_space.shape[0]
        act_low    = self.env.action_space.low
        act_high   = self.env.action_space.high

        # ── 노이즈 샘플링 [K, H, act_dim] ──
        epsilon = np.random.randn(K, H, act_dim) * self.sigma

        # ── 궤적 롤아웃 ──
        states  = np.tile(state, (K, 1))          # [K, 4]
        costs   = np.zeros(K)

        for t in range(H):
            u_noisy = np.clip(self.U[t] + epsilon[:, t, :], act_low, act_high)
            states  = self.env.dynamics(states, u_noisy)
            costs  += self.env.running_cost(states)

        costs += self.env.terminal_cost(states)

        # ── MPPI 가중치 계산 ──
        beta    = costs.min()
        weights = np.exp(-(costs - beta) / self.lam)
        weights = weights / (weights.sum() + 1e-8)

        # ── 제어 시퀀스 업데이트 ──
        for t in range(H):
            self.U[t] += np.sum(weights[:, None] * epsilon[:, t, :], axis=0)
        self.U = np.clip(self.U, act_low, act_high)

        # 통계 저장
        self.cost_history.append(costs.mean())
        entropy = -np.sum(weights * np.log(weights + 1e-10))
        self.weight_entropy.append(entropy)

        # ── 첫 번째 행동 반환 & 시퀀스 시프트 ──
        action = self.U[0].copy()
        self.U[:-1] = self.U[1:]
        self.U[-1]  = 0.0

        return action, weights, epsilon


# ──────────────────────────────────────────────────────────
# 3.  실시간 Matplotlib 시각화
# ──────────────────────────────────────────────────────────
class Visualizer:
    def __init__(self, env, controller):
        self.env  = env
        self.ctrl = controller

        # ── Figure 레이아웃 ──
        self.fig = plt.figure(figsize=(14, 8), facecolor="#0d0d0d")
        self.fig.suptitle("MPPI  ·  2D Point Mass Simulator", color="white",
                          fontsize=14, fontweight="bold", y=0.97)
        gs = GridSpec(2, 3, figure=self.fig,
                      left=0.05, right=0.97, top=0.92, bottom=0.08,
                      wspace=0.35, hspace=0.45)

        # 메인 2D 월드
        self.ax_world = self.fig.add_subplot(gs[:, :2])
        # 비용 이력
        self.ax_cost  = self.fig.add_subplot(gs[0, 2])
        # 가중치 엔트로피
        self.ax_ent   = self.fig.add_subplot(gs[1, 2])

        self._setup_world()
        self._setup_plots()

        # 추적 기록
        self.traj_x, self.traj_y = [], []
        self.step_count  = 0
        self.total_reward = 0.0

    # ── 월드 초기화 ──
    def _setup_world(self):
        ax = self.ax_world
        ax.set_facecolor("#111111")
        ax.set_xlim(-self.env.BOUND - 0.5, self.env.BOUND + 0.5)
        ax.set_ylim(-self.env.BOUND - 0.5, self.env.BOUND + 0.5)
        ax.set_aspect("equal")
        ax.set_title("2D World", color="white", fontsize=11)
        ax.tick_params(colors="gray")
        ax.spines[:].set_color("gray")
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)

        # 격자
        ax.grid(True, color="#222222", linewidth=0.5, zorder=0)

        # 경계
        border = patches.Rectangle(
            (-self.env.BOUND, -self.env.BOUND),
            2 * self.env.BOUND, 2 * self.env.BOUND,
            linewidth=1.5, edgecolor="#555555", facecolor="none", zorder=1
        )
        ax.add_patch(border)

        # 장애물
        self.obs_patches = []
        for obs in self.env.obstacles:
            c = plt.Circle(obs["pos"], obs["r"], color="#e74c3c", alpha=0.75, zorder=2)
            ax.add_patch(c)
            # 위험 반경
            c2 = plt.Circle(obs["pos"], obs["r"] + 0.5,
                            color="#e74c3c", alpha=0.15, linestyle="--",
                            fill=False, linewidth=0.8, zorder=2)
            ax.add_patch(c2)

        # 목표 지점
        self.goal_patch = plt.Circle(self.env.goal, 0.4,
                                     color="#2ecc71", alpha=0.9, zorder=3)
        ax.add_patch(self.goal_patch)
        # 목표 글로우
        self.goal_glow  = plt.Circle(self.env.goal, 0.8,
                                     color="#2ecc71", alpha=0.2, zorder=2)
        ax.add_patch(self.goal_glow)
        ax.text(self.env.goal[0], self.env.goal[1] + 0.9, "GOAL",
                color="#2ecc71", ha="center", fontsize=8, fontweight="bold", zorder=5)

        # 시작 지점
        ax.plot(-7, -7, "o", color="#3498db", markersize=8, zorder=4)
        ax.text(-7, -7.9, "START", color="#3498db", ha="center", fontsize=8)

        # 에이전트 & 궤적
        self.line_traj, = ax.plot([], [], "-", color="#f39c12",
                                  linewidth=1.2, alpha=0.6, zorder=5)
        self.agent_dot, = ax.plot([], [], "o", color="#ffffff",
                                  markersize=10, zorder=7,
                                  markeredgecolor="#f39c12", markeredgewidth=2)
        self.vel_arrow  = ax.annotate("", xy=(0, 0), xytext=(0, 0),
                                      arrowprops=dict(arrowstyle="->",
                                                      color="#00d4ff", lw=1.5),
                                      zorder=6)

        # MPPI 샘플 궤적 (상위 20개)
        self.sample_lines = [
            ax.plot([], [], "-", color="#7dc96e", alpha=0.3, linewidth=0.6, zorder=3)[0]
            for _ in range(50)
        ]

        # 정보 텍스트
        self.info_text = ax.text(
            -self.env.BOUND + 0.3, self.env.BOUND - 0.5, "",
            color="white", fontsize=8.5, va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#1a1a2e", alpha=0.8),
            zorder=10
        )

    # ── 보조 플롯 초기화 ──
    def _setup_plots(self):
        for ax, title, color in [
            (self.ax_cost, "Avg Sample Cost",    "#e67e22"),
            (self.ax_ent,  "Weight Entropy",     "#9b59b6"),
        ]:
            ax.set_facecolor("#111111")
            ax.set_title(title, color="white", fontsize=9)
            ax.tick_params(colors="gray", labelsize=7)
            ax.spines[:].set_color("gray")
            ax.grid(True, color="#222222", linewidth=0.5)

        self.line_cost, = self.ax_cost.plot([], [], "-", color="#e67e22", linewidth=1.2)
        self.line_ent,  = self.ax_ent.plot([], [],  "-", color="#9b59b6", linewidth=1.2)

    # ── 업데이트 ──
    def update(self, state, action, weights, epsilon, reward, done):
        x, y, vx, vy = state

        # 에이전트 위치
        self.traj_x.append(x)
        self.traj_y.append(y)
        self.line_traj.set_data(self.traj_x, self.traj_y)
        self.agent_dot.set_data([x], [y])

        # 속도 화살표
        self.vel_arrow.remove()
        scale = 1.2
        self.vel_arrow = self.ax_world.annotate(
            "", xy=(x + vx * scale, y + vy * scale), xytext=(x, y),
            arrowprops=dict(arrowstyle="->", color="#00d4ff", lw=1.8),
            zorder=6
        )

        # 상위 샘플 궤적 그리기
        # top_idx = np.argsort(weights)[-20:][::-1]
        
        top_idx = np.arange(0,len(weights),len(weights)/50)
        for i, idx in enumerate(top_idx):
            idx = int(idx)
            sx, sy = [x], [y]
            s_state = np.tile(state, (1, 1))
            for t in range(self.ctrl.H):
                u = np.clip(self.ctrl.U[t] + epsilon[idx, t], -2, 2).reshape(1, -1)
                s_state = self.env.dynamics(s_state, u)
                sx.append(s_state[0, 0])
                sy.append(s_state[0, 1])
            self.sample_lines[i].set_data(sx, sy)

        # 통계 그래프
        n = len(self.ctrl.cost_history)
        xs = list(range(n))
        self.line_cost.set_data(xs, self.ctrl.cost_history)
        self.ax_cost.relim(); self.ax_cost.autoscale_view()

        self.line_ent.set_data(xs, self.ctrl.weight_entropy)
        self.ax_ent.relim(); self.ax_ent.autoscale_view()

        # 정보 텍스트
        self.total_reward += reward
        self.step_count   += 1
        dist = np.linalg.norm(state[:2] - self.env.goal)
        self.info_text.set_text(
            f"Step : {self.step_count:4d}\n"
            f"Pos  : ({x:5.2f}, {y:5.2f})\n"
            f"Vel  : ({vx:5.2f}, {vy:5.2f})\n"
            f"Dist : {dist:.3f}\n"
            f"Rew  : {reward:+.3f}\n"
            f"Total: {self.total_reward:+.1f}"
        )

        if done:
            self.ax_world.set_title("2D World  ✓ GOAL REACHED!", color="#2ecc71", fontsize=11)


# ──────────────────────────────────────────────────────────
# 4.  메인 루프
# ──────────────────────────────────────────────────────────
def run():
    # 환경 & 컨트롤러 생성
    env  = PointMass2DEnv()
    ctrl = MPPIController(
        env,
        horizon     = 35,
        n_samples   = 1024,
        temperature = 0.1,
        noise_sigma = 2.5,
    )

    state, _ = env.reset()
    ctrl.reset()
    viz = Visualizer(env, ctrl)

    print("=" * 50)
    print("  MPPI + 2D Point Mass Simulator")
    print("=" * 50)
    print("  목표: (-7,-7) → (7,7)")
    print("  장애물 5개 회피")
    print("  창을 닫으면 종료됩니다.")
    print("=" * 50)

    # ── FuncAnimation 콜백 ──
    episode_done = [False]

    def animate(frame):
        if episode_done[0]:
            return

        action, weights, epsilon = ctrl.compute_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)

        done = terminated or truncated
        viz.update(state, action, weights, epsilon, reward, done)
        state[:] = next_state

        if terminated:
            print(f"\n✓ 목표 도달!  스텝: {viz.step_count}  총 보상: {viz.total_reward:.1f}")
            episode_done[0] = True
        elif truncated:
            print(f"\n✗ 타임아웃.  스텝: {viz.step_count}  총 보상: {viz.total_reward:.1f}")
            episode_done[0] = True

        viz.fig.canvas.draw_idle()

    ani = FuncAnimation(
        viz.fig,
        animate,
        interval=50,     # ms
        cache_frame_data=False,
        blit=False,
    )

    plt.show()


if __name__ == "__main__":
    run()
