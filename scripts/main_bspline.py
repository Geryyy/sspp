from sspp import _sspp as sp
from sspp import CubicPath as cp
from sspp import BSplines as bs
import ctypes
import os
import sys
import numpy as np
# from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

pinutil_path = os.path.abspath(os.path.join("/home/ubuntu/", "python"))
# pinutil_path = os.path.abspath(os.path.join("/home/gebmer/repos/robocrane", "python"))
#print(pinutil_path)
sys.path.append(pinutil_path)
# import pinutil
import create_robocrane_env as cre


def load():
    mj_model, mj_data, env = cre.create_robocrane_mujoco_models()
    # pin_model, env, geom_model = cre.create_robocrane_pinocchio_models_with_collision()
    pin_model, env = cre.create_robocrane_pinocchio_models()
    pin_data = pin_model.createData()
    tool_frame_id = cre.get_gripper_point_frame_id(pin_model)
    xml_path = os.path.join(cre.mjenv_path, "presets", "robocrane", "robocrane.xml")
    return mj_model, mj_data, xml_path, pin_model, pin_data, env, tool_frame_id


def get_pointer(obj):
    """Convert a Python object to a raw pointer."""
    # Assumes the object is wrapped in a ctypes.Structure
    address = id(obj)
    return ctypes.cast(address, ctypes.c_void_p)



import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_triad(transformations, labels, ax=None, scale=0.1, block=False):
    """
    Plots triads (x, y, z axes) for a list of homogeneous transformations with labels.
    
    Parameters:
    - transformations: List of homogeneous transformation matrices (4x4 numpy arrays).
    - labels: List of labels corresponding to the transformations.
    - ax: (optional) Matplotlib 3D Axes object. Creates a new one if not provided.
    - scale: Scale of the triad axes (default 0.1).
    """
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    
    # Plot each transformation
    for i, (T, label) in enumerate(zip(transformations, labels)):
        # Origin of the frame
        origin = T[:3, 3]
        # Extract rotation matrix (upper 3x3 block)
        R = T[:3, :3]
        
        # Define axes vectors
        x_axis = origin + scale * R[:, 0]  # X-axis (red)
        y_axis = origin + scale * R[:, 1]  # Y-axis (green)
        z_axis = origin + scale * R[:, 2]  # Z-axis (blue)
        
        # Plot axes
        ax.quiver(*origin, *(x_axis - origin), color='r', label='X' if i == 0 else "", linewidth=1)
        ax.quiver(*origin, *(y_axis - origin), color='g', label='Y' if i == 0 else "", linewidth=1)
        ax.quiver(*origin, *(z_axis - origin), color='b', label='Z' if i == 0 else "", linewidth=1)
        
        # Add label at the origin
        ax.text(*origin, label, color='k', fontsize=10)
    
    # Set axis limits and labels
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio for 3D plot
    ax.legend()

    if block:
        plt.show(block=block)
        return None
    else:
        plt.show(block=False)
        return fig, ax



import time 
import mujoco 
import pinutil as pu
from sspp import SteadyState as ss
# import numpy as np
import pinocchio as pin

def main():
    mj_model, mj_data, xml_path, pin_model, pin_data, env, tool_frame_id = load()

    # site_names = ["wall/site_left_wall", "wall/site_right_wall"]
    # site1_xpos = mj_data.site(site_names[0]).xpos + np.array([0,0,0.0])
    # site2_xpos = mj_data.site(site_names[1]).xpos + np.array([0,0,0.0])

    site1_xpos = np.array([0.5, 0.2, 0.3]) # left wall
    site2_xpos = np.array([0.5, -0.2, 0.3]) # left wall

    via_xpos = (site1_xpos + site2_xpos)/2+ np.array([0,0,0.3])

    q_init = np.array([0, 0, 0, np.pi/2, 0, -np.pi/2, 0, 0, 0])

    iksolver = ss.SteadyState(pin_model, pin_data, tool_frame_id)

    joint_id = 7
    # print("site1_xpos: ", np.array(site1_xpos))
    # print("site2_xpos: ", np.array(site2_xpos))
    
    # path start
    oMdes1 = pu.create_se3_from_rpy_and_trans(np.array(site1_xpos), np.array([0,0,np.pi]))
    # q_ik, success = pu.inverse_kinematics_clik(pin_model, pin_data, q_init, joint_id, oMdes)
    print("oMdes1: ", oMdes1.homogeneous)
    qd1, success = iksolver.inverse_kinematics(oMdes1.homogeneous, q_init)
    print("q_ik: ", qd1.T)
    print("success: ", success)
    frot = iksolver.compute_frot(qd1, oMdes1.homogeneous)
    print("frot: ", frot)
    # if success:
    mj_data.qpos[0:9] = qd1

    # path end
    oMdes2 = pu.create_se3_from_rpy_and_trans(np.array(site2_xpos), np.array([0,0,np.pi]))
    # q_ik, success = pu.inverse_kinematics_clik(pin_model, pin_data, q_init, joint_id, oMdes)
    print("oMdes2: ", oMdes2.homogeneous)
    qd2, success = iksolver.inverse_kinematics(oMdes2.homogeneous, qd1)
    print("q_ik: ", qd2.T)
    print("success: ", success)
    frot = iksolver.compute_frot(qd2, oMdes2.homogeneous)
    print("frot: ", frot)

    # path end
    oMvia = pu.create_se3_from_rpy_and_trans(np.array(via_xpos), np.array([0,0,np.pi]))
    # q_ik, success = pu.inverse_kinematics_clik(pin_model, pin_data, q_init, joint_id, oMdes)
    print("oMdes2: ", oMvia.homogeneous)
    qvia, success = iksolver.inverse_kinematics(oMvia.homogeneous, qd1)
    print("q_ik: ", qvia.T)
    print("success: ", success)
    # frot = iksolver.compute_frot(qvia, oMdes2.homogeneous)
    # print("frot: ", frot)


    oMorigin = pu.create_se3_from_rpy_and_trans(np.array([0,0,0]), np.array([0,0,0]))
    oMstart = pu.get_frameSE3(pin_model, pin_data, qd1, tool_frame_id)
    oMend = pu.get_frameSE3(pin_model, pin_data, qd2, tool_frame_id)

    M_ls = [oMorigin.homogeneous, oMstart.homogeneous, oMend.homogeneous, oMdes1.homogeneous, oMvia.homogeneous, oMdes2.homogeneous]
    label_ls = ["origin", "start", "end", "des1", "via", "des2"]

    # plot_triad(M_ls, label_ls, block=True)

   
    cubic_planner = cp.CubicPath()

    start = qd1[0:7]
    end = qd2[0:7] #+ np.array([0,0,0,0,0,0,1])
    via = qvia[0:7] #+ np.array([0,0,0,0,0,0,1])
    sigma = 0.08
    limits = np.ones((7,1)) * np.pi
    
    t_start = time.time()
    succ_paths = []
    # success = planner.plan(start,end, sigma, limits, succ_paths, sample_count = 100, check_points = 100, init_points = 7)
    success = cubic_planner.plan(start, via, end)
    duration = time.time() - t_start
    print("Success: ", success)
    print("Duration: ", duration)

    # fit bspline based on cubic path
    n_ctr_pts = 7
    k = 2 # spline order
    u_ = np.linspace(0, 1, n_ctr_pts)
    via_pts = []
    for i in range(n_ctr_pts):
        v = cubic_planner.evaluate(u_[i])
        v = np.concatenate([v, np.zeros(2)])
        via_pts.append(v)

    # print("u_: ", u_)
    # print("len(via_pts): ", len(via_pts))
    # print("via_pts: ")
    # for v in via_pts:
    #     # 2 significant digits
    #     print(np.round(v, 2).T)

    via_pts = np.array(via_pts)
    # compute bspline
    ctr_pts, knot_vec = bs.compute_control_points(via_pts, k)

    # create dictionary with all the parameters
    bspline_params = {
        "knot_vec": knot_vec,
        "ctr_pts": ctr_pts,
        "k": k
    }
    # now save the dictionary to a file
    np.save("bspline_params.npy", bspline_params)

    # print("len(ctr_pts): ", len(ctr_pts))
    # print("ctr_pts: ")
    # for v in ctr_pts:
    #     # 2 significant digits
    #     print(np.round(v, 2).T)

    # plot bspline curve 
    theta = np.linspace(0, 1, 100)
    y = np.array([bs.bspline(x, knot_vec, ctr_pts, k) for x in theta])
    y_gt = np.array([cubic_planner.evaluate(x) for x in theta])

    import matplotlib.pyplot as plt
    plt.plot(theta, y)
    plt.plot(theta, y_gt, 'r--')
    plt.grid()
    plt.show()


    # exit()

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

            if sim_time <= T_traj:
                u = min((sim_time) / T_traj, 1)
                # q_act = cubic_planner.evaluate(u)
                # print("q_act: ", q_act.T)
                # mj_data.qpos[0:7] = cubic_planner.evaluate(u)
                mj_data.qpos[0:7] = bs.bspline(u, knot_vec, ctr_pts, k)[:7]
                # print("u: ", u)
                sim_time += dt
                print("u: ", u, end="\r")
            # else:
            #     break
        
            
            mujoco.mj_forward(mj_model, mj_data)

            time_until_next_frame = frame_time - (time.time() - last_frame_time)
            if time_until_next_frame < 0:
                last_frame_time = time.time()
                viewer.sync()
            
            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = mj_model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()