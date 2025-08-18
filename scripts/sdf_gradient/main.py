import time 
import mujoco 
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import BSplines as bs


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from plot import *


k = 3
n_control_points = 5



def print_geom_names(mj_model):
    for i in range(mj_model.ngeom):
        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, i)
        print(f"{i}: {name}")

def set_body_free_joint(mj_model, mj_data, body_name, pos, quat):
    body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"Body '{body_name}' not found in model.")
    jnt_id = mj_model.body_jntadr[body_id]
    qpos_adr = mj_model.jnt_qposadr[jnt_id]
    
    mj_data.qpos[qpos_adr:qpos_adr+3] = pos
    mj_data.qpos[qpos_adr+3:qpos_adr+7] = quat


def get_closest_points(mj_model, mj_data, pos, quat, coll_geom_list, env_geom_list):
    min_dist = float("inf")
    min_fromto = np.zeros(6)
    coll_geom_id = -1
    env_geom_id = -1

    dist_max = 100.0

    closest_point = None
    for coll_geom_id in coll_geom_list:
        for env_geom_id in env_geom_list:
            if env_geom_id == coll_geom_id:
                continue
            # Compute distance from the point to the body
            fromto = np.zeros(6)
            dist = mujoco.mj_geomDistance(mj_model, mj_data, coll_geom_id, env_geom_id, dist_max, fromto)
            if dist < min_dist:
                min_dist = dist
                min_fromto = fromto
                min_coll_geom_id = coll_geom_id
                min_env_geom_id = env_geom_id

    grad_direction = (min_fromto[3:6] - min_fromto[0:3])
    grad = min_dist * grad_direction / np.linalg.norm(grad_direction) if np.linalg.norm(grad_direction) > 0 else np.zeros(3)

    return min_dist, grad, min_coll_geom_id, min_env_geom_id


def get_collision_gradient(mj_model, mj_data, u_list, knot_pts, ctrl_pts, coll_body_name, coll_geom_list, env_geom_list):
    col_grad_ls = []

    # collision gradient w.r.t to spline point
    for i in range(len(ctrl_pts)):
        col_grad_acc = np.zeros(3)
        for u in u_list:
            pos = bs.bspline(u, knot_pts, ctrl_pts, k)
            # set collision
            set_body_free_joint(mj_model, mj_data, coll_body_name, pos[:3], np.array([1, 0, 0, 0]))
            mujoco.mj_forward(mj_model, mj_data)

            min_dist, grad, min_coll_geom_id, min_env_geom_id = get_closest_points(mj_model, mj_data, pos, np.array([1, 0, 0, 0]), coll_geom_list, env_geom_list)
            col_grad_acc += grad * bs.B(u, k, i, knot_pts)
        col_grad_ls.append(col_grad_acc / len(u_list))

    return col_grad_ls


def main():
    # xml_file = "./../../mjcf/robocrane/robocrane.xml"
    xml_file = "/home/geraldebmer/repos/sspp/mjcf/robocrane/robocrane.xml"
    mj_model = mujoco.MjModel.from_xml_path(xml_file)
    mj_data = mujoco.MjData(mj_model)

    mujoco.mj_forward(mj_model, mj_data)

    start_name = "block_green/"
    end_name = "block_orange/"

    start_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, start_name)
    end_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, end_name)

    start_pos = mj_data.xpos[start_id] + np.array([0, 0, 0.1])  # Offset to avoid collision with the ground
    end_pos = mj_data.xpos[end_id] + np.array([0, 0, 0.1])  # Offset to avoid collision with the ground
    print("Start Position:", start_pos)
    print("End Position:", end_pos)

    # spline
    # middle_pos = (start_pos + end_pos) / 2
    via_pts = np.linspace(start_pos, end_pos, n_control_points)
    ctrl_pts, knot_pts = bs.compute_control_points(via_pts, k)
    u_list = np.linspace(0, 1, 11)

    for u in u_list:
        pos = bs.bspline(u, knot_pts, ctrl_pts, k)
        print(f"Position at u={u:.2f}: {pos}")


    print_geom_names(mj_model)    
    # exit()
    
    coll_body_name = "gripper_collision_with_block/"
    # coll_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, coll_body_name)
    env_geom_list = range(0,9)
    coll_geom_list = range(9,29)

    print("env_geom_list: ", env_geom_list)
    print("coll_geom_list: ", coll_geom_list)

    gradient_list = get_collision_gradient(mj_model, mj_data, u_list, knot_pts, ctrl_pts, coll_body_name, coll_geom_list, env_geom_list)
    print("gradient_list: ", [1e3*g for g in gradient_list])

    visualize_trajectory_optimization(knot_pts, ctrl_pts, gradient_list, k)
    exit()
    
    T_traj = 10

    framerate = 25
    frame_time = 1.0 / framerate
    dt = mj_model.opt.timestep 

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        # Close the viewer automatically after 30 wall-seconds.
        sim_time = 0
        last_frame_time = time.time()
        while viewer.is_running():
            step_start = time.time()
        
            
            mujoco.mj_forward(mj_model, mj_data)
            sim_time += dt
            
            time.sleep(0.8*dt)
            time_until_next_frame = frame_time - (time.time() - last_frame_time)
            if time_until_next_frame < 0:
                last_frame_time = time.time()
                viewer.sync()

                body_pos = bs.bspline(sim_time / T_traj, knot_pts, ctrl_pts, k)
                set_body_free_joint(mj_model, mj_data, coll_body_name, body_pos[:3], np.array([1, 0, 0, 0]))
                min_dist, grad, min_coll_geom_id, min_env_geom_id = get_closest_points(mj_model, mj_data, body_pos, np.array([1, 0, 0, 0]), coll_geom_list, env_geom_list)
                print("body pos: ", body_pos)
                print("Minimum Distance:", min_dist)
                print("Gradient (norm):", grad/np.linalg.norm(grad))
                print("Geom 1:", mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, min_coll_geom_id))
                print("Geom 2:", mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, min_env_geom_id))
                print("contact: ", min_dist < 0) 
                print("---")


            if sim_time >= T_traj:
                break

if __name__ == "__main__":
    main()