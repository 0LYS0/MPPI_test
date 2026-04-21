"""
robot_env.py — Franka Panda 7-DOF arm environment  (SE(3) goal)

Goal is now T_goal ∈ SE(3): position [3] + quaternion [4] (xyzw).
Cost = w_pos * ||p_ee - p_goal|| + w_ori * orientation_error(R_ee, R_goal)

Orientation error metric: geodesic angle on SO(3)
  e_ori = || log(R_goal^T R_ee) ||_F  (approx via trace formula)
  = arccos( clip( (tr(R_goal^T R_ee) - 1) / 2, -1, 1) )

Dynamics (simplified, no M/C/g):
  1st-order: state=[q]      (7D),  action=[dq]  (7D)
  2nd-order: state=[q,dq]  (14D),  action=[ddq] (7D)
"""

import numpy as np

N_DOF  = 7
DT     = 0.05

Q_MIN   = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
Q_MAX   = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973])
DQ_MAX  = np.array([ 2.175,   2.175,   2.175,   2.175,   2.61,    2.61,    2.61  ])
DDQ_MAX = np.array([15.0,      7.5,    10.0,    12.5,    15.0,    20.0,    20.0  ])

Q_HOME  = np.array([0.0, -np.pi/4, 0.0, -3*np.pi/4, 0.0, np.pi/2, np.pi/4])

# ── URDF joint parameters ────────────────────────────────────────
_JOINT_XYZ = np.array([
    [ 0.000,  0.000,  0.333],
    [ 0.000,  0.000,  0.000],
    [ 0.000, -0.316,  0.000],
    [ 0.0825, 0.000,  0.000],
    [-0.0825, 0.384,  0.000],
    [ 0.000,  0.000,  0.000],
    [ 0.088,  0.000,  0.000],
])
_JOINT_RPY = np.array([
    [ 0.000,       0, 0],
    [-np.pi/2,     0, 0],
    [ np.pi/2,     0, 0],
    [ np.pi/2,     0, 0],
    [-np.pi/2,     0, 0],
    [ np.pi/2,     0, 0],
    [ np.pi/2,     0, 0],
])
_FLANGE_XYZ = np.array([0.0, 0.0, 0.107])


# ── FK helpers ───────────────────────────────────────────────────

def _rpy_to_R(rpy):
    r, p, y = rpy
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    return np.array([
        [cy*cp,  cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr],
        [sy*cp,  sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr],
        [-sp,    cp*sr,             cp*cr            ],
    ])


def _make_T(xyz, rpy):
    T = np.eye(4)
    T[:3, :3] = _rpy_to_R(rpy)
    T[:3,  3] = xyz
    return T


_T_FIXED  = np.array([_make_T(_JOINT_XYZ[i], _JOINT_RPY[i]) for i in range(N_DOF)])
_T_FLANGE = _make_T(_FLANGE_XYZ, [0, 0, 0])


# ── Quaternion helpers ───────────────────────────────────────────

def quat_xyzw_to_R(q):
    """Unit quaternion [x,y,z,w] → 3×3 rotation matrix."""
    x, y, z, w = q / (np.linalg.norm(q) + 1e-12)
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-z*w),     2*(x*z+y*w)  ],
        [2*(x*y+z*w),     1-2*(x*x+z*z),   2*(y*z-x*w)  ],
        [2*(x*z-y*w),     2*(y*z+x*w),     1-2*(x*x+y*y)],
    ])


def R_to_quat_xyzw(R):
    """3×3 rotation matrix → unit quaternion [x,y,z,w]."""
    t = R[0,0] + R[1,1] + R[2,2]
    if t > 0:
        s = 0.5 / np.sqrt(t + 1.0)
        return np.array([(R[2,1]-R[1,2])*s, (R[0,2]-R[2,0])*s,
                          (R[1,0]-R[0,1])*s, 0.25/s])
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        return np.array([0.25*s, (R[0,1]+R[1,0])/s,
                          (R[0,2]+R[2,0])/s, (R[2,1]-R[1,2])/s])
    elif R[1,1] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        return np.array([(R[0,1]+R[1,0])/s, 0.25*s,
                          (R[1,2]+R[2,1])/s, (R[0,2]-R[2,0])/s])
    else:
        s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        return np.array([(R[0,2]+R[2,0])/s, (R[1,2]+R[2,1])/s,
                          0.25*s, (R[1,0]-R[0,1])/s])


# ── Single-config FK returning full SE(3) ────────────────────────

def fk_ee_pose(q):
    """
    Single configuration FK → (pos [3], R [3,3]).
    """
    T = np.eye(4)
    for i in range(N_DOF):
        T = T @ _T_FIXED[i]
        c, s = np.cos(q[i]), np.sin(q[i])
        Rz = np.array([[c,-s,0,0],[s,c,0,0],[0,0,1,0],[0,0,0,1]])
        T  = T @ Rz
    T = T @ _T_FLANGE
    return T[:3, 3].copy(), T[:3, :3].copy()


def fk_all_joints(q):
    """
    Returns 9 positions (base + 7 joints + EE) and EE rotation matrix.
    """
    positions = [np.zeros(3)]
    T = np.eye(4)
    for i in range(N_DOF):
        T = T @ _T_FIXED[i]
        c, s = np.cos(q[i]), np.sin(q[i])
        Rz = np.array([[c,-s,0,0],[s,c,0,0],[0,0,1,0],[0,0,0,1]])
        T  = T @ Rz
        positions.append(T[:3, 3].copy())
    T = T @ _T_FLANGE
    positions.append(T[:3, 3].copy())
    return positions, T[:3, :3].copy()   # (9 positions, R_ee)


# ── Batched FK: position + rotation ─────────────────────────────

def batch_fk_ee(q_batch):
    """[K,7] → [K,3] EE positions."""
    K = q_batch.shape[0]
    T = np.tile(np.eye(4), (K, 1, 1))
    for i in range(N_DOF):
        T  = T @ _T_FIXED[i]
        c  = np.cos(q_batch[:, i]); s = np.sin(q_batch[:, i])
        Rz = np.zeros((K, 4, 4))
        Rz[:, 0, 0]=c; Rz[:, 0, 1]=-s
        Rz[:, 1, 0]=s; Rz[:, 1, 1]= c
        Rz[:, 2, 2]=1; Rz[:, 3, 3]=1
        T  = np.matmul(T, Rz)
    T = T @ _T_FLANGE
    return T[:, :3, 3]


def batch_fk_ee_pose(q_batch):
    """[K,7] → ([K,3] positions, [K,3,3] rotation matrices)."""
    K = q_batch.shape[0]
    T = np.tile(np.eye(4), (K, 1, 1))
    for i in range(N_DOF):
        T  = T @ _T_FIXED[i]
        c  = np.cos(q_batch[:, i]); s = np.sin(q_batch[:, i])
        Rz = np.zeros((K, 4, 4))
        Rz[:, 0, 0]=c; Rz[:, 0, 1]=-s
        Rz[:, 1, 0]=s; Rz[:, 1, 1]= c
        Rz[:, 2, 2]=1; Rz[:, 3, 3]=1
        T  = np.matmul(T, Rz)
    T = T @ _T_FLANGE
    return T[:, :3, 3], T[:, :3, :3]   # [K,3], [K,3,3]


# ── Geodesic orientation error ───────────────────────────────────

def batch_ori_error(R_ee_batch, R_goal):
    """
    Geodesic angle between each R_ee and R_goal.
    R_ee_batch : [K, 3, 3]
    R_goal     : [3, 3]
    Returns    : [K] angles in radians  (0 = identical)
    """
    # R_rel = R_goal^T @ R_ee  → [K,3,3]
    R_rel = np.matmul(R_goal.T, R_ee_batch)           # [K,3,3]
    trace = R_rel[:, 0, 0] + R_rel[:, 1, 1] + R_rel[:, 2, 2]  # [K]
    cos_a = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    return np.arccos(cos_a)                             # [K] radians


# ── Environment ──────────────────────────────────────────────────

class PandaArmEnv:
    """
    SE(3) goal: goal_pos [3] + goal_R [3,3] (stored as quat internally).
    Cost = w_pos*||p_ee - p_goal|| + w_ori*geodesic_angle(R_ee, R_goal)
    """
    n_dof = N_DOF
    DT    = DT

    def __init__(self):
        self.dyn_model = "2nd"
        # Default goal
        self.goal_pos  = np.array([0.5,  0.0,  0.5])
        self.goal_R    = np.eye(3)
        # Cost weights
        self.w_pos        = 2.0
        self.w_ori        = 1.5
        self.w_joint_lim  = 50.0   # joint limit barrier weight
        self.use_ori_cost = True
        # Joint limits
        self.q_min   = Q_MIN.copy()
        self.q_max   = Q_MAX.copy()
        self.dq_max  = DQ_MAX.copy()
        self.ddq_max = DDQ_MAX.copy()

    # ── Convenience: set goal from quaternion [x,y,z,w] ──────────
    def set_goal_pose(self, pos, quat_xyzw):
        self.goal_pos = np.array(pos, dtype=float)
        self.goal_R   = quat_xyzw_to_R(np.array(quat_xyzw, dtype=float))

    def get_goal_quat(self):
        """Returns goal orientation as [x,y,z,w] quaternion."""
        return R_to_quat_xyzw(self.goal_R)

    # ── Standard env interface ────────────────────────────────────
    def state_dim(self):
        return N_DOF * 2 if self.dyn_model == "2nd" else N_DOF

    def action_lim(self):
        return float(self.ddq_max.max() if self.dyn_model=="2nd" else self.dq_max.max())

    def action_dim(self):
        return N_DOF

    def init_state(self):
        s = np.zeros(self.state_dim())
        s[:N_DOF] = Q_HOME
        return s

    def get_q(self, state):   return state[:N_DOF]
    def get_dq(self, state):  return state[N_DOF:] if self.dyn_model=="2nd" else np.zeros(N_DOF)

    # ── Dynamics ──────────────────────────────────────────────────
    def step(self, state, action):
        lim = self.action_lim()
        a   = np.clip(action, -lim, lim)
        if self.dyn_model == "1st":
            return np.clip(state + a * self.DT, self.q_min, self.q_max)
        q, dq = state[:N_DOF].copy(), state[N_DOF:].copy()
        dq    = np.clip(dq + a * self.DT, -self.dq_max, self.dq_max)
        q_new = q + dq * self.DT
        # Hard limit: zero velocity when joint reaches limit (no overshoot)
        at_limit = (q_new < self.q_min) | (q_new > self.q_max)
        dq       = np.where(at_limit, 0.0, dq)
        q_new    = np.clip(q_new, self.q_min, self.q_max)
        return np.concatenate([q_new, dq])

    def batch_step(self, states, actions):
        lim = self.action_lim()
        a   = np.clip(actions, -lim, lim)
        if self.dyn_model == "1st":
            return np.clip(states + a * self.DT, self.q_min, self.q_max)
        q  = states[:, :N_DOF].copy()
        dq = states[:, N_DOF:].copy()
        dq    = np.clip(dq + a * self.DT, -self.dq_max, self.dq_max)
        q_new = q + dq * self.DT
        # Hard limit: zero velocity per joint when hitting limit  [K, 7]
        at_limit = (q_new < self.q_min) | (q_new > self.q_max)
        dq       = np.where(at_limit, 0.0, dq)
        q_new    = np.clip(q_new, self.q_min, self.q_max)
        return np.concatenate([q_new, dq], axis=1)

    # ── Cost functions ────────────────────────────────────────────
    def running_cost(self, states):
        q = states[:, :N_DOF]

        if self.use_ori_cost:
            pos_ee, R_ee = batch_fk_ee_pose(q)          # [K,3], [K,3,3]
            pos_cost = np.linalg.norm(pos_ee - self.goal_pos, axis=1) * self.w_pos
            ori_cost = batch_ori_error(R_ee, self.goal_R) * self.w_ori
            cost     = pos_cost + ori_cost
        else:
            pos_ee   = batch_fk_ee(q)
            cost     = np.linalg.norm(pos_ee - self.goal_pos, axis=1) * self.w_pos

        # Joint limit barrier — two-zone: soft (0.15 rad) + hard (0.05 rad)
        margin_soft = 0.15
        margin_hard = 0.05
        excess_lo_s = np.maximum(0.0, self.q_min + margin_soft - q)
        excess_hi_s = np.maximum(0.0, q - (self.q_max - margin_soft))
        excess_lo_h = np.maximum(0.0, self.q_min + margin_hard - q)
        excess_hi_h = np.maximum(0.0, q - (self.q_max - margin_hard))
        cost += (excess_lo_s**2 + excess_hi_s**2).sum(axis=1) * self.w_joint_lim
        cost += (excess_lo_h**2 + excess_hi_h**2).sum(axis=1) * self.w_joint_lim * 5.0

        # Velocity penalty (2nd-order)
        if self.dyn_model == "2nd":
            dq     = states[:, N_DOF:]
            exceed = np.maximum(0.0, np.abs(dq) - self.dq_max * 0.85)
            cost  += (exceed**2).sum(axis=1) * 5.0

        return cost

    def terminal_cost(self, states):
        q = states[:, :N_DOF]
        if self.use_ori_cost:
            pos_ee, R_ee = batch_fk_ee_pose(q)
            pos_err = np.linalg.norm(pos_ee - self.goal_pos, axis=1) * self.w_pos * 10.0
            ori_err = batch_ori_error(R_ee, self.goal_R) * self.w_ori * 10.0
            return pos_err + ori_err
        pos_ee = batch_fk_ee(q)
        return np.linalg.norm(pos_ee - self.goal_pos, axis=1) * 20.0

    # ── FK helpers ────────────────────────────────────────────────
    def ee_pose(self, state):
        """Returns (pos [3], R [3,3]) for a single state."""
        return fk_ee_pose(self.get_q(state))

    def ee_position(self, state):
        pos, _ = self.ee_pose(state)
        return pos

    def joint_positions_list(self, state):
        positions, R_ee = fk_all_joints(self.get_q(state))
        return [p.tolist() for p in positions], R_ee
