"""
robot_main.py — FastAPI + WebSocket server for Franka Panda MPPI simulation.

Run (port 8001 to coexist with 2D simulator on 8000):
    uvicorn robot_main:app --reload --host 0.0.0.0 --port 8001

WebSocket protocol
──────────────────
Client → Server:
    {"type": "start" | "pause" | "reset"}
    {"type": "params",       "data": {...}}
    {"type": "set_algos",    "algos": ["mppi"] | ["mppi","cem"]}
    {"type": "set_goal",     "goal": [x, y, z]}
    {"type": "set_seed",     "seed": 42}
    {"type": "set_dyn_model","model": "1st"|"2nd"}
    {"type": "set_physics",  "dq_scale": 1.0, "ddq_scale": 1.0}

Server → Client (every step):
    {"type":"state", "active_algos":[...], "step":n,
     "joint_positions": [[x,y,z]×9], "ee_pos":[x,y,z],
     "goal_ee":[x,y,z], "dist_to_goal":float,
     "agents":{algo: {"q":[], "dq":[], "cost_history":[],
                      "rollout_ee_trajs":[[...]], "rollout_costs":[],
                      "best_pos":int}}}
"""
import asyncio
import sys
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Reuse existing controllers unchanged
sys.path.insert(0, str(Path(__file__).parent.parent / "simulator"))
try:
    from controllers import MPPIController, CEMController, iCEMController
except ImportError:
    # If running standalone, expect controllers.py in same directory or parent
    sys.path.insert(0, str(Path(__file__).parent))
    from controllers import MPPIController, CEMController, iCEMController

from robot_env import PandaArmEnv, N_DOF, fk_all_joints, Q_MIN, Q_MAX

app = FastAPI(title="Panda MPPI Simulator")

CTRL_CLASSES = {
    "mppi": MPPIController,
    "cem":  CEMController,
    "icem": iCEMController,
}


# ═══════════════════════════════════════════════════════
#  Rollout EE trajectory helper
# ═══════════════════════════════════════════════════════
def compute_rollout_ee_trajs(env: PandaArmEnv, ctrl, start_state: np.ndarray,
                              n_disp: int = 20):
    """
    Compute EE Cartesian trajectories for a subset of rollout samples.
    Uses controller's stored last_samples and last_costs.
    Returns: (ee_trajs, costs, best_pos)
    """
    if ctrl.last_samples is None or ctrl.last_costs is None:
        return [], [], 0

    K       = ctrl.K
    H       = ctrl.H
    samples = ctrl.last_samples   # [K, H, n_dof]
    costs   = ctrl.last_costs     # [K]
    lim_per_joint = env.dq_max if env.dyn_model == "1st" else env.ddq_max  # [7]

    best_idx = int(np.argmin(costs))
    step     = max(1, K // n_disp)
    indices  = sorted(set(list(range(0, K, step))[:n_disp] + [best_idx]))

    ee_trajs   = []
    traj_costs = []

    for ki in indices:
        q  = start_state[:N_DOF].copy()
        dq = (start_state[N_DOF:].copy()
              if env.dyn_model == "2nd" and len(start_state) > N_DOF
              else np.zeros(N_DOF))

        traj = [fk_all_joints(q)[0][-1].tolist()]

        for t in range(H):
            a = np.clip(samples[ki, t, :], -lim_per_joint, lim_per_joint)
            if env.dyn_model == "1st":
                q = np.clip(q + a * env.DT, env.q_min, env.q_max)
            else:
                dq = np.clip(dq + a * env.DT, -env.dq_max, env.dq_max)
                q  = np.clip(q + dq * env.DT,  env.q_min,  env.q_max)
            traj.append(fk_all_joints(q)[0][-1].tolist())

        ee_trajs.append(traj)
        traj_costs.append(float(costs[ki]))

    best_pos = indices.index(best_idx) if best_idx in indices else 0
    return ee_trajs, traj_costs, best_pos


# ═══════════════════════════════════════════════════════
#  Simulation session
# ═══════════════════════════════════════════════════════
class RobotSimSession:
    MAX_STEPS  = 600
    GOAL_THRESH = 0.01   # metres — goal reached threshold

    def __init__(self):
        self.env          = PandaArmEnv()
        self.active_algos = ["mppi"]
        self.seed         = 42
        self.params = dict(
            # common
            H=20, K=200, lam=0.05, alpha=1.0,
            sg_enabled=False, sg_win=5, sg_order=2,
            # MPPI
            sigma=0.3,
            # CEM
            cem_N_iter=4, cem_elite_frac=0.1, cem_sigma=0.8,
            # iCEM
            icem_N_iter=4, icem_elite_frac=0.1, icem_sigma=0.8,
            icem_beta_color=1.0, icem_elite_reuse=0.3,
        )
        self.ctrls: dict = {}
        self.agents: dict = {}
        self.step_count   = 0
        self.total_reward = 0.0
        self.goal_reached = False
        self.running      = False
        self._rebuild()

    # ── controller factory ───────────────────────────────
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
        self.ctrls  = {name: self._build_ctrl(name) for name in CTRL_CLASSES}
        self.agents = {
            name: {"state": self.env.init_state(), "traj_ee": [],
                   "done": False, "last_action": np.zeros(N_DOF)}
            for name in CTRL_CLASSES
        }
        self.step_count   = 0
        self.total_reward = 0.0
        self.goal_reached = False
        self.running      = False

    # ── message handler ──────────────────────────────────
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
            algos = [a for a in msg.get("algos", ["mppi"]) if a in CTRL_CLASSES][:2]
            if algos:
                self.active_algos = algos
                for name in algos:
                    self.agents[name]["state"]  = self.env.init_state()
                    self.agents[name]["traj_ee"] = []
                    self.agents[name]["done"]   = False
                    self.ctrls[name].reset(seed=self.seed)
                self.step_count   = 0
                self.total_reward = 0.0
                self.goal_reached = False
                self.running      = False

        elif t == "set_goal":
            # Accept pos-only [3] or SE(3) {pos:[3], quat:[x,y,z,w]}
            if isinstance(msg.get("goal"), dict):
                pos  = msg["goal"].get("pos",  [0.5, 0.0, 0.5])
                quat = msg["goal"].get("quat", [0.0, 0.0, 0.0, 1.0])
                self.env.set_goal_pose(pos, quat)
            else:
                # legacy: plain [x,y,z] list
                self.env.goal_pos = np.array(msg["goal"], dtype=float)

        elif t == "set_goal_ori_only":
            quat = msg.get("quat", [0.0, 0.0, 0.0, 1.0])
            from robot_env import quat_xyzw_to_R
            self.env.goal_R = quat_xyzw_to_R(np.array(quat, dtype=float))

        elif t == "set_ori_weight":
            self.env.w_ori        = float(msg.get("w_ori", 1.5))
            self.env.use_ori_cost = bool(msg.get("use_ori", True))

        elif t == "set_joint_weights":
            if "w_joint_lim" in msg:
                self.env.w_joint_lim = float(msg["w_joint_lim"])

        elif t == "set_seed":
            self.seed = int(msg.get("seed", 42))
            for ctrl in self.ctrls.values():
                ctrl.reset(seed=self.seed)

        elif t == "set_dyn_model":
            self.env.dyn_model = msg.get("model", "2nd")
            self._rebuild()

        elif t == "set_physics":
            if "dq_scale" in msg:
                s = float(msg["dq_scale"])
                self.env.dq_max = s * np.array([2.175, 2.175, 2.175, 2.175,
                                                 2.61, 2.61, 2.61])
            if "ddq_scale" in msg:
                s = float(msg["ddq_scale"])
                self.env.ddq_max = s * np.array([15, 7.5, 10, 12.5, 15, 20, 20])

    # ── simulation step ──────────────────────────────────
    def _do_step(self):
        all_done = True
        for k in self.active_algos:
            ag = self.agents[k]
            if ag["done"]:
                continue
            all_done = False

            action = self.ctrls[k].compute(ag["state"])
            ag["last_action"] = action
            ag["state"] = self.env.step(ag["state"], action)

            ee_pos, ee_R = self.env.ee_pose(ag["state"])
            ag["traj_ee"].append(ee_pos.tolist())

            pos_err = float(np.linalg.norm(ee_pos - self.env.goal_pos))
            from robot_env import batch_ori_error
            ori_err = float(batch_ori_error(ee_R[None], self.env.goal_R)[0])
            self.total_reward -= (pos_err + ori_err * 0.5) * 0.1

            # Goal reached: position within threshold AND orientation within 0.2 rad (~11°)
            pos_ok = pos_err < self.GOAL_THRESH
            ori_ok = ori_err < 0.1 if self.env.use_ori_cost else True
            if (pos_ok and ori_ok) or self.step_count >= self.MAX_STEPS:
                ag["done"] = True
                if pos_ok and ori_ok:
                    self.goal_reached = True

        self.step_count += 1
        if all_done or self.step_count >= self.MAX_STEPS:
            self.running = False

    # ── state message ─────────────────────────────────────
    def build_state_msg(self) -> dict:
        from robot_env import R_to_quat_xyzw, batch_ori_error
        first_ag    = self.agents[self.active_algos[0]]
        first_state = first_ag["state"]

        agents_out = {}
        for k in self.active_algos:
            ag   = self.agents[k]
            ctrl = self.ctrls[k]
            s    = ag["state"]
            q    = s[:N_DOF]
            dq   = s[N_DOF:].tolist() if self.env.dyn_model == "2nd" else []

            ee_trajs, traj_costs, best_pos = compute_rollout_ee_trajs(
                self.env, ctrl, s, n_disp=18)

            agents_out[k] = {
                "q":                q.tolist(),
                "dq":               dq,
                "action":           ag["last_action"].tolist(),
                "traj_ee":          ag["traj_ee"][-200:],
                "done":             ag["done"],
                "cost_history":     ctrl.cost_history[-150:],
                "rollout_ee_trajs": ee_trajs,
                "rollout_costs":    traj_costs,
                "best_pos":         best_pos,
            }

        ee_pos, ee_R = self.env.ee_pose(first_state)
        ee_quat = R_to_quat_xyzw(ee_R).tolist()
        goal_quat = self.env.get_goal_quat().tolist()

        ori_err = float(batch_ori_error(ee_R[None], self.env.goal_R)[0])

        joint_positions_list, _ = self.env.joint_positions_list(first_state)

        return {
            "type":           "state",
            "step":            self.step_count,
            "active_algos":    self.active_algos,
            "running":         self.running,
            "goal_reached":    self.goal_reached,
            "total_reward":    round(self.total_reward, 2),
            "joint_positions": joint_positions_list,
            "ee_pos":          ee_pos.tolist(),
            "ee_quat":         ee_quat,          # [x,y,z,w]
            "goal_pos":        self.env.goal_pos.tolist(),
            "goal_quat":       goal_quat,         # [x,y,z,w]
            "dist_to_goal":    float(np.linalg.norm(ee_pos - self.env.goal_pos)),
            "ori_error":       round(ori_err, 4),
            "use_ori_cost":    self.env.use_ori_cost,
            "w_ori":           self.env.w_ori,
            "dyn_model":       self.env.dyn_model,
            "agents":          agents_out,
        }


# ═══════════════════════════════════════════════════════
#  WebSocket endpoint
# ═══════════════════════════════════════════════════════
@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    session = RobotSimSession()
    await websocket.send_json(session.build_state_msg())

    SILENT = {"set_goal"}

    async def recv_loop():
        async for data in websocket.iter_json():
            session.handle_message(data)
            if session.running and data.get("type") in SILENT:
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
        _, pending = await asyncio.wait(
            [recv_task, sim_task], return_when=asyncio.FIRST_COMPLETED)
        for t in pending:
            t.cancel()
    except WebSocketDisconnect:
        recv_task.cancel()
        sim_task.cancel()


# ═══════════════════════════════════════════════════════
#  URDF API
# ═══════════════════════════════════════════════════════
URDF_PATH     = Path(__file__).parent / "assets/franka_panda/franka_panda/model.urdf"
MESH_BASE_URL = "/assets/franka_panda/franka_panda"

# Ensure urdf_parser.py (same directory) is importable
_robot_dir = str(Path(__file__).parent)
if _robot_dir not in sys.path:
    sys.path.insert(0, _robot_dir)

# Parse once at startup (lazily on first request)
_urdf_cache = None   # type: dict or None

def get_urdf_data():
    global _urdf_cache
    if _urdf_cache is not None:
        return _urdf_cache
    if not URDF_PATH.exists():
        _urdf_cache = {"error": f"URDF not found at {URDF_PATH}"}
        return _urdf_cache
    try:
        from urdf_parser import parse_urdf
        _urdf_cache = parse_urdf(str(URDF_PATH), MESH_BASE_URL)
    except Exception as e:
        _urdf_cache = {"error": f"Parse failed: {e}"}
    return _urdf_cache

@app.get("/api/urdf")
async def api_urdf():
    """Return parsed URDF kinematic tree as JSON for Three.js scene graph."""
    return JSONResponse(content=get_urdf_data())

@app.get("/api/check-meshes")
async def api_check_meshes():
    """Check which mesh files exist on disk. Useful for diagnosing missing assets."""
    data = get_urdf_data()
    if "error" in data:
        return JSONResponse(content={"error": data["error"]})

    results = []
    for lname, ldata in data.get("links", {}).items():
        for vis in ldata.get("visuals", []):
            url = vis.get("mesh_url", "")
            if not url:
                continue
            # Convert URL to filesystem path
            rel  = url.lstrip("/")                          # "assets/franka_panda/..."
            path = Path(__file__).parent / rel
            results.append({
                "link":   lname,
                "url":    url,
                "path":   str(path),
                "exists": path.exists(),
            })

    missing = [r for r in results if not r["exists"]]
    return JSONResponse(content={
        "total":   len(results),
        "found":   len(results) - len(missing),
        "missing": len(missing),
        "files":   results,
        "hint": ("Put mesh files under: " + str(Path(__file__).parent / "assets"))
                 if missing else "All mesh files found ✓",
    })


# ═══════════════════════════════════════════════════════
#  Static files
# ═══════════════════════════════════════════════════════
STATIC_DIR = Path(__file__).parent / "static"
ASSET_DIR  = Path(__file__).parent / "assets"

if not STATIC_DIR.exists():
    raise RuntimeError(f"Static directory not found: {STATIC_DIR}")

@app.get("/")
async def serve_index():
    return FileResponse(str(STATIC_DIR / "robot.html"))

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
if ASSET_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(ASSET_DIR)), name="assets")
