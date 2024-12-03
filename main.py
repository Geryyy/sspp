from sspp import _sspp as sp
import ctypes
import os
import sys
import numpy as np
from scipy.interpolate import splprep, splev
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
    # pin_model, env = cre.create_robocrane_pinocchio_models()
    # pin_data = pin_model.createData()
    # tool_frame_id = crimport ctypes
    xml_path = os.path.join(cre.mjenv_path, "presets", "robocrane", "robocrane.xml")
    return mj_model, mj_data, xml_path


def get_pointer(obj):
    """Convert a Python object to a raw pointer."""
    # Assumes the object is wrapped in a ctypes.Structure
    address = id(obj)
    return ctypes.cast(address, ctypes.c_void_p)



import time 
import mujoco 

def main():
    mj_model, mj_data, xml_path = load()

    planner = sp.SamplingPathPlanner7(xml_path)

    start = np.zeros((7,1))
    end = np.ones((7,1)) * np.pi/2
    sigma = 0.1
    limits = np.ones((7,1)) * np.pi
    
    t_start = time.time()

    success = planner.plan(start,end, sigma, limits, sample_count = 1000, check_points = 10)

    duration = time.time() - t_start
    print("Success: ", success)
    print("Duration: ", duration)
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

            if sim_time > T_traj:
                break

            u = min((sim_time) / T_traj, 1)
            q_act = planner.evaluate(u)
            print("q_act: ", q_act.T)
            mj_data.qpos[0:7] = planner.evaluate(u)
            # print("u: ", u)
            sim_time += dt
        
            
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