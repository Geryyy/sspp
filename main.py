from sspp import _sspp as sp
import ctypes
import os
import sys
import numpy as np
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# pinutil_path = os.path.abspath(os.path.join("/home/ubuntu/", "python"))
pinutil_path = os.path.abspath(os.path.join("/home/gebmer/repos/robocrane", "python"))
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
    return mj_model, mj_data


def get_pointer(obj):
    """Convert a Python object to a raw pointer."""
    # Assumes the object is wrapped in a ctypes.Structure
    address = id(obj)
    return ctypes.cast(address, ctypes.c_void_p)



def main():
    mj_model, mj_data = load()

    # Convert mjModel and mjData to void* pointers
    model_ptr = get_pointer(mj_model)
    data_ptr = get_pointer(mj_data)
    
    # Initialize the planner with void* pointers
    model_capsule = sp.create_model_capsule(mj_model)
    data_capsule = sp.create_data_capsule(mj_data)
    planner = sp.SamplingPathPlanner7(model_capsule, data_capsule)
    # planner = sp.SamplingPathPlanner7(mj_model, mj_data)

    start = np.zeros((7,1))
    end = np.ones((7,1)) * np.pi/2
    sigma = 0.1
    limits = np.ones((7,1)) * np.pi

    success = planner.plan(start,end, sigma, limits)
    print("Success: ", success)


if __name__ == "__main__":
    main()