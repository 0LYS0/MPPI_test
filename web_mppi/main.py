"""
main.py — FastAPI + WebSocket simulation server.

Run:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
Then open http://localhost:8000

WebSocket protocol
──────────────────
Client → Server:
    {"type": "start" | "pause" | "reset"}
    {"type": "params",       "data": {...}}
    {"type": "set_algos",    "algos": ["mppi"] | ["mppi","cem"]}
    {"type": "set_obstacles","obstacles": [...]}
    {"type": "reset_obstacles"}
    {"type": "set_goal",     "goal": [x, y]}
    {"type": "set_seed",     "seed": 42}
    {"type": "set_dyn_model","model": "1st"|"2nd"}
    {"type": "set_physics",  "max_speed": 4.0, "max_acc": 2.0}

Server → Client (every step):
    {"type":"state", "active_algos":[...], "step":n, "agents":{...}, ...}
"""
import asyncio
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from env import PointMass2DEnv, DEFAULT_OBSTACLES
from controllers import MPPIController, DIALController, CEMController, iCEMController

app = FastAPI(title="MPPI · DIAL-MPC · CEM · iCEM Simulator")

CTRL_CLASSES = {
    "mppi": MPPIController,
    "dial": DIALController,
    "cem":  CEMController,
    "icem": iCEMController,
}

# ═══════════════════════════════════════════════════════
#  Simulation session
# ═══════════════════════════════════════════════════════
class SimSession:
    MAX_STEPS = 1200

    def __init__(self):
        self.env          = PointMass2DEnv()
        self.active_algos = ["mppi"]          # 1 or 2 algo names
        self.seed         = 42
        self.params = dict(
            # common
            H=30, K=400, lam=0.05, alpha=1.0,
            sg_enabled=False, sg_win=7, sg_order=3,
            # MPPI
            sigma=0.9,
            # DIAL
            N=5, sigma_base=1.2, beta1=0.6, beta2=0.5,
            # CEM
            cem_N_iter=5, cem_elite_frac=0.1, cem_sigma=1.5,
            # iCEM
            icem_N_iter=5, icem_elite_frac=0.1, icem_sigma=1.5,
            icem_beta_color=1.0, icem_elite_reuse=0.3,
            # physics
            max_speed=4.0, max_acc=2.0,
        )
        self.ctrls: dict = {}
        self.agents: dict = {}
        self.step_count   = 0
        self.total_reward = 0.0
        self.goal_reached = False
        self.running      = False
        self._rebuild()

    # ── controller factory ────────────────────────────
    def _common_kw(self):
        p = self.params
        return dict(H=p["H"], K=p["K"], lam=p["lam"], alpha=p["alpha"],
                    sg_enabled=p["sg_enabled"], sg_win=p["sg_win"],
                    sg_order=p["sg_order"], seed=self.seed)

    def _build_ctrl(self, name: str):
        p  = self.params
        kw = self._common_kw()
        if name == "mppi":
            return MPPIController(self.env, sigma=p["sigma"], **kw)
        if name == "dial":
            return DIALController(self.env, N=p["N"], sigma_base=p["sigma_base"],
                                  beta1=p["beta1"], beta2=p["beta2"], **kw)
        if name == "cem":
            return CEMController(self.env, N_iter=p["cem_N_iter"],
                                 elite_frac=p["cem_elite_frac"],
                                 sigma=p["cem_sigma"], **kw)
        if name == "icem":
            return iCEMController(self.env, N_iter=p["icem_N_iter"],
                                  elite_frac=p["icem_elite_frac"],
                                  sigma=p["icem_sigma"],
                                  beta_color=p["icem_beta_color"],
                                  elite_reuse=p["icem_elite_reuse"], **kw)
        raise ValueError(f"Unknown algo: {name}")

    def _rebuild(self):
        p = self.params
        self.env.max_speed = float(p.get("max_speed", self.env.max_speed))
        self.env.max_acc   = float(p.get("max_acc",   self.env.max_acc))
        self.ctrls  = {name: self._build_ctrl(name) for name in CTRL_CLASSES}
        self.agents = {
            name: {"state": self.env.init_state(), "traj": [],
                   "done": False, "last_action": [0.0, 0.0]}
            for name in CTRL_CLASSES
        }
        self.step_count   = 0
        self.total_reward = 0.0
        self.goal_reached = False
        self.running      = False

    # ── message handler ───────────────────────────────
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
            self.params.update(msg.get("data", {}))
            self._rebuild()

        elif t == "set_algos":
            algos = msg.get("algos", ["mppi"])
            algos = [a for a in algos if a in CTRL_CLASSES][:2]
            if algos:
                self.active_algos = algos
                # Reset only the newly-active agents
                for name in algos:
                    self.agents[name]["state"] = self.env.init_state()
                    self.agents[name]["traj"]  = []
                    self.agents[name]["done"]  = False
                    self.ctrls[name].reset(seed=self.seed)
                self.step_count   = 0
                self.total_reward = 0.0
                self.goal_reached = False
                self.running      = False

        elif t == "set_obstacles":
            self.env.set_obstacles_from_list(msg["obstacles"])

        elif t == "reset_obstacles":
            self.env.set_obstacles_from_list(DEFAULT_OBSTACLES)

        elif t == "set_goal":
            self.env.goal = np.array(msg["goal"], dtype=float)

        elif t == "set_seed":
            self.seed = int(msg.get("seed", 42))
            for ctrl in self.ctrls.values():
                ctrl.reset(seed=self.seed)

        elif t == "set_dyn_model":
            self.env.dyn_model = msg.get("model", "2nd")
            self._rebuild()

        elif t == "set_physics":
            if "max_speed" in msg:
                self.env.max_speed = float(msg["max_speed"])
                self.params["max_speed"] = self.env.max_speed
            if "max_acc" in msg:
                self.env.max_acc = float(msg["max_acc"])
                self.params["max_acc"] = self.env.max_acc

    # ── simulation step ───────────────────────────────
    def _do_step(self):
        all_done = True
        for k in self.active_algos:
            ag = self.agents[k]
            if ag["done"]: continue
            all_done = False
            action = self.ctrls[k].compute(ag["state"])
            ag["last_action"] = action.tolist()
            ag["state"] = self.env.step(ag["state"], action)
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

    # ── state message ─────────────────────────────────
    def build_state_msg(self) -> dict:
        agents = {}
        for k in self.active_algos:
            ag   = self.agents[k]
            ctrl = self.ctrls[k]
            s    = ag["state"]
            trajs, costs, indices, best_pos, n_explore = ctrl.get_rollout_trajs(s)
            sigma_profile = (ctrl.sigma_schedule_list(1)
                             if hasattr(ctrl, "sigma_schedule_list") else [])
            agents[k] = {
                "pos":           s[:2].tolist(),
                "vel":           s[2:].tolist() if self.env.dyn_model == "2nd" else [],
                "action":        ag["last_action"],
                "traj":          ag["traj"][-400:],
                "done":          ag["done"],
                "cost_history":  ctrl.cost_history[-200:],
                "sigma_profile": sigma_profile,
                "n_explore":     n_explore,
                "rollouts": {
                    "trajs":    trajs,
                    "costs":    costs,
                    "indices":  indices,
                    "best_pos": best_pos,
                },
            }
        return {
            "type":          "state",
            "step":           self.step_count,
            "active_algos":   self.active_algos,
            "running":        self.running,
            "goal_reached":   self.goal_reached,
            "total_reward":   round(self.total_reward, 2),
            "obstacles":      self.env.obstacles_to_list(),
            "goal":           self.env.goal.tolist(),
            "dyn_model":      self.env.dyn_model,
            "max_speed":      self.env.max_speed,
            "max_acc":        self.env.max_acc,
            "agents":         agents,
        }


# ═══════════════════════════════════════════════════════
#  WebSocket endpoint
# ═══════════════════════════════════════════════════════
@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    session = SimSession()
    await websocket.send_json(session.build_state_msg())

    SILENT_WHILE_RUNNING = {"set_goal", "set_obstacles"}

    async def recv_loop():
        async for data in websocket.iter_json():
            session.handle_message(data)
            if session.running and data.get("type") in SILENT_WHILE_RUNNING:
                continue
            await websocket.send_json(session.build_state_msg())

    async def sim_loop():
        while True:
            if session.running:
                await asyncio.to_thread(session._do_step)
                try:
                    await websocket.send_json(session.build_state_msg())
                except Exception:
                    break
                await asyncio.sleep(0.02)
            else:
                await asyncio.sleep(0.05)

    recv_task = asyncio.create_task(recv_loop())
    sim_task  = asyncio.create_task(sim_loop())
    try:
        done, pending = await asyncio.wait(
            [recv_task, sim_task], return_when=asyncio.FIRST_COMPLETED)
        for t in pending:
            t.cancel()
    except WebSocketDisconnect:
        recv_task.cancel(); sim_task.cancel()


# ═══════════════════════════════════════════════════════
#  Static files
# ═══════════════════════════════════════════════════════
STATIC_DIR = Path(__file__).parent / "static"

if not STATIC_DIR.exists():
    raise RuntimeError(
        f"\n\n  Static directory not found: {STATIC_DIR}\n"
        "  Run:  mkdir -p static && mv index.html static/\n")

@app.get("/")
async def serve_index():
    return FileResponse(str(STATIC_DIR / "index.html"))

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static_assets")
