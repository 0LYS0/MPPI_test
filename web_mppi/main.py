"""
main.py — FastAPI + WebSocket server.

Run:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

Then open http://localhost:8000 in your browser.

WebSocket protocol
──────────────────
Client → Server:
    {"type": "start"}
    {"type": "pause"}
    {"type": "reset"}
    {"type": "params", "data": {...}}
    {"type": "set_algo", "algo": "mppi"|"dial"|"both"}
    {"type": "set_obstacles", "obstacles": [...]}
    {"type": "set_goal", "goal": [x, y]}
    {"type": "set_seed", "seed": 42}
    {"type": "set_dyn_model", "model": "1st"|"2nd"}

Server → Client (every simulation step + on demand):
    {"type": "state", "step": n, "running": bool, "goal_reached": bool,
     "total_reward": float, "algo": str,
     "obstacles": [...], "goal": [x,y],
     "agents": {"mppi": {...}, "dial": {...}}}
"""
import asyncio
import json
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from env import PointMass2DEnv
from controllers import MPPIController, DIALController

app = FastAPI(title="MPPI · DIAL-MPC Simulator")


# ═══════════════════════════════════════════════════════
#  Simulation session (one per WebSocket connection)
# ═══════════════════════════════════════════════════════
class SimSession:
    MAX_STEPS = 1200

    def __init__(self):
        self.env   = PointMass2DEnv()
        self.algo  = "mppi"          # "mppi" | "dial" | "both"
        self.seed  = 42
        self.params = dict(
            H=30, K=400, lam=0.05, sigma=0.9, alpha=1.0,
            N=5, sigma_base=1.2, beta1=0.6, beta2=0.5,
            sg_enabled=False, sg_win=7, sg_order=3,
        )
        self.ctrls:  dict[str, MPPIController | DIALController] = {}
        self.agents: dict[str, dict] = {}
        self.step_count   = 0
        self.total_reward = 0.0
        self.goal_reached = False
        self.running      = False
        self._rebuild()

    # ── controller / state management ────────────────────
    def _ctrl_kwargs(self):
        p = self.params
        return dict(
            H=p["H"], K=p["K"], lam=p["lam"], alpha=p["alpha"],
            sg_enabled=p["sg_enabled"], sg_win=p["sg_win"], sg_order=p["sg_order"],
            seed=self.seed,
        )

    def _rebuild(self):
        """Create / recreate controllers and reset agent states."""
        p  = self.params
        kw = self._ctrl_kwargs()
        self.ctrls = {
            "mppi": MPPIController(self.env, sigma=p["sigma"],  **kw),
            "dial": DIALController(self.env, N=p["N"],
                                   sigma_base=p["sigma_base"],
                                   beta1=p["beta1"], beta2=p["beta2"], **kw),
        }
        self._reset_agents()
        self.step_count   = 0
        self.total_reward = 0.0
        self.goal_reached = False
        self.running      = False

    def _reset_agents(self):
        self.agents = {
            k: {"state": self.env.init_state(), "traj": [],
                "done": False, "last_action": [0.0, 0.0]}
            for k in ("mppi", "dial")
        }

    # ── message handler ───────────────────────────────────
    def handle_message(self, msg: dict):
        t = msg.get("type", "")

        if t == "start":
            if self.goal_reached or self.step_count >= self.MAX_STEPS:
                self._rebuild()
            self.running = True

        elif t == "pause":
            self.running = False

        elif t == "reset":
            self._rebuild()

        elif t == "params":
            data = msg.get("data", {})
            self.params.update(data)
            self._rebuild()

        elif t == "set_algo":
            self.algo = msg.get("algo", "mppi")

        elif t == "set_obstacles":
            self.env.set_obstacles_from_list(msg["obstacles"])

        elif t == "set_goal":
            self.env.goal = np.array(msg["goal"], dtype=float)

        elif t == "set_seed":
            self.seed = int(msg.get("seed", 42))
            for ctrl in self.ctrls.values():
                ctrl.reset(seed=self.seed)

        elif t == "set_dyn_model":
            self.env.dyn_model = msg.get("model", "2nd")
            self._rebuild()

    # ── single simulation step ────────────────────────────
    def _do_step(self):
        keys     = ["mppi", "dial"] if self.algo == "both" else [self.algo]
        all_done = True

        for k in keys:
            ag = self.agents[k]
            if ag["done"]:
                continue
            all_done = False

            action         = self.ctrls[k].compute(ag["state"])
            ag["last_action"] = action.tolist()
            ag["state"]    = self.env.step(ag["state"], action)
            ag["traj"].append(ag["state"][:2].tolist())

            dist = float(np.linalg.norm(ag["state"][:2] - self.env.goal))
            self.total_reward -= dist * 0.1

            if dist < 0.4 or self.step_count >= self.MAX_STEPS:
                ag["done"] = True
                if dist < 0.4:
                    self.goal_reached = True

        self.step_count += 1
        if all_done or self.step_count >= self.MAX_STEPS:
            self.running = False

    # ── build JSON state message ──────────────────────────
    def build_state_msg(self) -> dict:
        keys   = ["mppi", "dial"] if self.algo == "both" else [self.algo]
        agents = {}

        for k in keys:
            ag   = self.agents[k]
            ctrl = self.ctrls[k]
            s    = ag["state"]
            trajs, costs, indices, best_pos = ctrl.get_rollout_trajs(s)

            # sigma history for mini-chart
            sigma_profile = (
                ctrl.sigma_schedule_list(1)
                if hasattr(ctrl, "sigma_schedule_list") else []
            )

            agents[k] = {
                "pos":    s[:2].tolist(),
                "vel":    s[2:].tolist() if self.env.dyn_model == "2nd" else [],
                "action": ag["last_action"],
                "traj":   ag["traj"][-400:],
                "done":   ag["done"],
                "cost_history": ctrl.cost_history[-200:],
                "sigma_profile": sigma_profile,
                "rollouts": {
                    "trajs":    trajs,
                    "costs":    costs,
                    "indices":  indices,
                    "best_pos": best_pos,
                },
            }

        return {
            "type":         "state",
            "step":          self.step_count,
            "algo":          self.algo,
            "running":       self.running,
            "goal_reached":  self.goal_reached,
            "total_reward":  round(self.total_reward, 2),
            "obstacles":     self.env.obstacles_to_list(),
            "goal":          self.env.goal.tolist(),
            "dyn_model":     self.env.dyn_model,
            "max_speed":     self.env.max_speed,
            "max_acc":       self.env.max_acc,
            "agents":        agents,
        }


# ═══════════════════════════════════════════════════════
#  WebSocket endpoint
# ═══════════════════════════════════════════════════════
@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    session = SimSession()

    # Send initial state immediately
    await websocket.send_json(session.build_state_msg())

    async def recv_loop():
        """Handle incoming messages from client."""
        async for data in websocket.iter_json():
            session.handle_message(data)
            # Always send back current state after any message
            await websocket.send_json(session.build_state_msg())

    async def sim_loop():
        """Push simulation steps when running."""
        while True:
            if session.running:
                # Run compute in thread to avoid blocking event loop
                await asyncio.to_thread(session._do_step)
                try:
                    await websocket.send_json(session.build_state_msg())
                except Exception:
                    break
                # ~20 ms between pushes (MPPI compute adds more latency)
                await asyncio.sleep(0.02)
            else:
                await asyncio.sleep(0.05)

    recv_task = asyncio.create_task(recv_loop())
    sim_task  = asyncio.create_task(sim_loop())
    try:
        done, pending = await asyncio.wait(
            [recv_task, sim_task],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for t in pending:
            t.cancel()
    except WebSocketDisconnect:
        recv_task.cancel()
        sim_task.cancel()


# ═══════════════════════════════════════════════════════
#  Static files
# ═══════════════════════════════════════════════════════
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
