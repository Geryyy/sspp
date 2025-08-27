# hooks.py
# Concrete hooks for VP-STO benchmarking with MuJoCo
# - Body to move: "gripper_collision_with_block/"
# - q = [x, y, z, yaw] for that body's free joint
# - Collision penalty: averaged penetration depth across the trajectory (0 if collision-free)
# - Path length: measured at that body's world position (d.xpos[body_id])

import numpy as np
import mujoco as mj

# ---- Configuration ----
BODY_NAME = "gripper_collision_with_block/"

# If you want to switch to a site instead of a body for path length, set these:
USE_SITE_FOR_PATH = False
SITE_NAME = "ee_site"  # only used if USE_SITE_FOR_PATH=True


def _find_body_and_free_joint(model):
    """Return (body_id, jnt_id, qpos_adr) for the body's FREE joint."""
    body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, BODY_NAME)
    if body_id < 0:
        raise RuntimeError(f'Body "{BODY_NAME}" not found in model.')

    jadr = model.body_jntadr[body_id]
    nj = model.body_jntnum[body_id]
    if nj <= 0:
        raise RuntimeError(f'Body "{BODY_NAME}" has no joints.')
    # Find a free joint attached to this body
    for k in range(nj):
        j = jadr + k
        if model.jnt_type[j] == mj.mjtJoint.mjJNT_FREE:
            qadr = model.jnt_qposadr[j]
            return body_id, j, qadr
    raise RuntimeError(f'Body "{BODY_NAME}" has no FREE joint; found types: {model.jnt_type[jadr:jadr+nj]}')


def _yaw_to_quat(yaw: float):
    """Quaternion (qw,qx,qy,qz) for rotation about z-axis by yaw."""
    half = 0.5 * yaw
    c = np.cos(half)
    s = np.sin(half)
    return np.array([c, 0.0, 0.0, s], dtype=float)


def _set_free_body_qpos_from_xyzyaw(model, data, q_xyzyaw: np.ndarray):
    """
    Write the free joint state for BODY_NAME into data.qpos:
    q_xyzyaw = [x, y, z, yaw]
    """
    if q_xyzyaw.shape[0] != 4:
        raise ValueError(f"Expected q of shape (4,), got {q_xyzyaw.shape}")

    body_id, jnt_id, qadr = _find_body_and_free_joint(model)
    # Free joint occupies 7 qpos: [x, y, z, qw, qx, qy, qz]
    data.qpos[qadr + 0] = q_xyzyaw[0]
    data.qpos[qadr + 1] = q_xyzyaw[1]
    data.qpos[qadr + 2] = q_xyzyaw[2]
    qw, qx, qy, qz = _yaw_to_quat(q_xyzyaw[3])
    data.qpos[qadr + 3] = qw
    data.qpos[qadr + 4] = qx
    data.qpos[qadr + 5] = qy
    data.qpos[qadr + 6] = qz
    # Note: we leave other qpos as-is (assumed fixed or zero)


def fk_point(q: np.ndarray, mj_model) -> np.ndarray:
    """
    Return the 3D world position of the moving body (or site) for path-length measurement.
    q: (4,) -> [x, y, z, yaw]
    """
    d = mj.MjData(mj_model)
    _set_free_body_qpos_from_xyzyaw(mj_model, d, q)
    mj.mj_forward(mj_model, d)

    if USE_SITE_FOR_PATH:
        site_id = mj.mj_name2id(mj_model, mj.mjtObj.mjOBJ_SITE, SITE_NAME)
        if site_id < 0:
            raise RuntimeError(f'Site "{SITE_NAME}" not found.')
        return np.array(d.site_xpos[site_id], dtype=float)

    body_id = mj.mj_name2id(mj_model, mj.mjtObj.mjOBJ_BODY, BODY_NAME)
    if body_id < 0:
        raise RuntimeError(f'Body "{BODY_NAME}" not found.')
    return np.array(d.xpos[body_id], dtype=float)


def collision_penalty(candidates: dict, mj_model) -> np.ndarray:
    """
    Compute a scalar penalty per candidate trajectory.
    - Input `candidates['pos'][i]`: array (T, 4) with per-timestep [x,y,z,yaw]
    - Penalty: average penetration depth over the trajectory (sum of max(0,-dist) over contacts) / T
               -> 0 if no contacts at any timestep.
    """
    pop_size = len(candidates["T"])
    penalties = np.zeros(pop_size, dtype=float)

    # Reuse a single MjData per candidate for speed
    for i in range(pop_size):
        q_traj = np.asarray(candidates["pos"][i], dtype=float)  # (T, 4)
        if q_traj.ndim != 2 or q_traj.shape[1] != 4:
            raise ValueError(f"Each candidate pos must be (T,4) [x,y,z,yaw]; got {q_traj.shape}")

        d = mj.MjData(mj_model)
        total_pen = 0.0

        for t in range(q_traj.shape[0]):
            _set_free_body_qpos_from_xyzyaw(mj_model, d, q_traj[t])
            mj.mj_forward(mj_model, d)
            mj.mj_collision(mj_model, d)

            # Accumulate penetration depth across *all* contacts
            if d.ncon > 0:
                # contact.dist < 0 indicates penetration; sum depths
                depth_sum = 0.0
                for k in range(d.ncon):
                    dist = d.contact[k].dist
                    if dist < 0.0:
                        depth_sum += (-dist)
                total_pen += depth_sum  # 0 if all distances >= 0

        # Average per time step (meters); 0 if no penetrations at all
        penalties[i] = total_pen / max(1, q_traj.shape[0])

    return penalties
