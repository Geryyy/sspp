import numpy as np
import pinocchio as pin
from pinocchio import casadi as cpin
import casadi as ca
import arc_core as ac
from casadi import *
from scipy.spatial.transform import Rotation
from importlib.resources import files # for accessing casadi data files

pinutil_path = os.path.abspath(os.path.join("/home/ubuntu/", "python"))
sys.path.append(pinutil_path)
import create_robocrane_env as cre
import pinutil as pu
import pinocchio as pin


def Rx(q):
    rotmat = np.array([[1, 0, 0],
                        [0, cos(q), -sin(q)],
                        [0, sin(q), cos(q)]])
    return rotmat

def Ry(q):
    rotmat = np.array([[cos(q), 0, sin(q)],
                        [0, 1, 0],
                        [-sin(q), 0, cos(q)]])
    return rotmat

def Rz(q):
    rotmat = np.array([[cos(q), -sin(q), 0],
                        [sin(q), cos(q), 0],
                        [0, 0, 1]])
    return rotmat

class SteadyState: 

    def __init__(self, pin_model, pin_data, tool_frame_id, tol=1e-6, max_iter=100):
        self.tol = tol
        self.max_iter = max_iter

        self.pin_model = pin_model
        self.pin_data = pin_data
        print("pin_model.type: ", type(pin_model))
        self.cpin_model = cpin.Model(pin_model)
        self.cpin_data = self.cpin_model.createData()

        self.tool_frame_id = tool_frame_id

        # get transformation from tool frame to last joint frame 
        q_neutral = pin.neutral(pin_model)
        pin.forwardKinematics(pin_model, pin_data, q_neutral)
        pin.updateFramePlacements(pin_model, pin_data)
        # pin.framesForwardKinematics(pin_model, pin_data, q_neutral)
        H_0_tool = pin_data.oMf[self.tool_frame_id]
        H_0_9 = pin_data.oMi[-1]

        # print("H_0_tool: ", H_0_tool)
        # print("H_0_9: ", H_0_9)

        self.H_tool_9 = H_0_tool.inverse() * H_0_9
        # print("H_tool_9: ", self.H_tool_9)

        # intialize casadi functions
        self.q_in = ca.SX.sym('q_in', 9,1)
        self.q_u = self.q_in[7:]
        self.q_a = self.q_in[:7]
        
        self.J = ca.jacobian(self._gravitational_forces_ca(self.q_in), self.q_u)
        self.F_gg_u = ca.Function('F_gg_u', [self.q_in], [self._gravitational_forces_ca(self.q_in)], ['q0'], ['qpp_u'])
        self.F_J = ca.Function('F_J', [self.q_in], [self.J], ['q0'], ['J'])

        self.ik_solver = self.optProblemGeneratorInverseKinematics()

        self.cost_weight_parameters = np.array([1, 1])

        self.q_ub = np.array([170, 120, 170, 120, 170, 120, 175, 15, 15]) * np.pi / 180
        self.q_lb = -self.q_ub

        self.lbg = np.zeros(2)
        self.ubg = np.zeros(2)

    def compute_frot(self, q, Hd):
        pd = Hd[:3, 3]
        Rd = Hd[:3, :3]
        pin.forwardKinematics(self.pin_model, self.pin_data, q.T)
        pin.updateFramePlacements(self.pin_model, self.pin_data)

        oMdes = pin.SE3(Rd, pd.T)
        oMact = self.pin_data.oMf[self.tool_frame_id]
        # iMd = oMdes.actInv(oMact)
        oMerr = oMdes * oMact.inverse()
        # oRerr = oMdes.rotation @ oMact.rotation.T

        print("Rd: ", Rd)
        print("Ract: ", oMact.rotation)

        R_ed = oMerr.rotation
        frot = 2*(3-np.trace(R_ed))
        return frot


    def optProblemGeneratorInverseKinematics(self):
        dof = self.pin_model.nq
        q = SX.sym('q', 1, dof)
        pd = SX.sym('pd', 1, 3) # goal / desired position
        Rd = SX.sym('Rd', 3, 3)
        cost_weight_parameters = SX.sym('cost_weight_parameters', 1, 2)

        cpin.forwardKinematics(self.cpin_model, self.cpin_data, q.T)
        cpin.updateFramePlacements(self.cpin_model, self.cpin_data)

        oMdes = cpin.SE3(Rd, pd.T)
        oMact = self.cpin_data.oMf[self.tool_frame_id]
        # iMd = oMdes.actInv(oMact)
        oMerr = oMdes * oMact.inverse()
        
        epos = oMerr.translation
        R_ed = oMerr.rotation
        # frot = (dot(oMact.rotation[:,0], Rd[:,0]) - 1)**2
        
        # general orientation error
        frot = 2*(3-trace(R_ed))
        fpos = dot(epos, epos)

        # cost function weights
        fpos *= cost_weight_parameters[0]
        frot *= cost_weight_parameters[1]
        f = fpos + frot 
        # f = fpos

        # constraints
        g = cpin.computeGeneralizedGravity(self.cpin_model, self.cpin_data, q.T)[7:]        

        # optimization variables
        opt_var = vertcat(q[:])
        # params = horzcat(cost_weight_parameters, pd, Rd[:].T, q0)
        params = horzcat(cost_weight_parameters, pd, Rd[:].T)

        print('opt_var shape: ' + str(opt_var.shape))
        print('params shape: ' + str(params.shape))

        nlp = {'x' : opt_var,
                'f' : f,
                'g' : g,
                'p' : params}

        ipopt_options = {'tol' : 1e-6,
                        'max_iter' : 50,
                        'linear_solver' : 'ma57',
                        # 'linear_system_scaling' : 'none',
                        # 'ma57_automatic_scaling' : 'no',
                        # 'ma57_pre_alloc' : 10,
                        # 'mu_strategy' : 'monotone',
                        # 'fixed_mu_oracle' : 'probing',
                        # 'expect_infeasible_problem' : 'no',
                        # 'print_info_string' : 'no',
                        # 'fast_step_computation' : 'yes',
                        'print_level' : 0} # 5
        nlp_options = {'ipopt' : ipopt_options,
                        'print_time' : 0}
        F = nlpsol('F', 'ipopt', nlp, nlp_options)

        # export code for compiling
        codegen_options = {'cpp' : True,
                        'indent' : 2}
        #F.generate_dependencies('gen_ik_nlp_deps.cpp', codegen_options)

        # ik_solver_path = files('pympc.code_generation').joinpath('gen_ik_nlp_deps.so')

        # check if file exists maybe?
        #F = nlpsol('F', 'ipopt', str(ik_solver_path), nlp_options)

        return F
    

    def inverse_kinematics(self, H_0_tool, q0 = np.zeros(9)):
        oMd = H_0_tool
        pd = oMd[:3, 3]
        Rd = oMd[:3, :3]

        params = np.concatenate([self.cost_weight_parameters, pd, Rd[:].flatten('F')])
                
        r = self.ik_solver(x0 = q0, p = params.T, lbx = self.q_lb.T, ubx = self.q_ub.T, lbg = self.lbg.T, ubg = self.ubg.T)
        self.stats = self.ik_solver.stats()
        self.iterations = self.stats['iter_count']
        if(self.stats['return_status'] != 'Solve_Succeeded'):
            print('not converged with status: ' + str(self.stats['return_status']))
            result = r['x'].full()
            q_res = result[0]
            H_out = pu.get_frameSE3(self.pin_model, self.pin_data, q_res, self.tool_frame_id).homogeneous
            return q_res, False
        else:        
            result = r['x'].full()
            q_res = result[0]
            H_out = pu.get_frameSE3(self.pin_model, self.pin_data, q_res, self.tool_frame_id).homogeneous
            # compare position with desired position
            p_out = H_out[:3, 3]
            p_des = pd
            # print("H_ik: ", H_out)
            # print("H_des: ", oMd)
            pos_err = np.linalg.norm(p_out - p_des)
            if pos_err > 1e-3:
                print("position error: ", pos_err)
                return q_res, False
            
            # everything is fine :)   
            return q_res, True
    


    def _gravitational_forces_ca(self, q,):
        qpp_u = cpin.computeGeneralizedGravity(self.cpin_model, self.cpin_data, q)[7:]
        return qpp_u
    
    def find_steady_state(self, q_init):
        q0_a = q_init[0:7]
        print("q_init: ", q_init)
        
        q_ss = self._newton_raphson_casadi(q_init)
        try:
            q_ss = q_ss.flatten()
        except:
            q_ss = q_ss.full().flatten()
            
        return q_ss
    
    def _newton_raphson_casadi(self, q_init):
        # Newton-Raphson iteration
        q_current = q_init.copy()
        for i in range(self.max_iter):
            # qpp_val, J_val = self.F(q_current)
            qpp_val = self.F_gg_u(q_current)
            J_val = self.F_J(q_current)

            print("qpp_val: ", qpp_val)
            print("J_val: ", J_val)
            qpp_norm = ca.norm_2(qpp_val)
            
            if qpp_norm < self.tol:
                # print(f"Converged after {i+1} iterations.")
                return q_current
            
            # Solve for the update (delta_q)
            print("J_val.shape: ", J_val.shape)
            print("qpp_val.shape: ", qpp_val.shape)

            delta_q = -ca.solve(J_val, qpp_val)
            
            # Update current estimate of q
            q_current[7:] += delta_q.full().flatten()
        
        print("Warning: Maximum iterations reached without convergence.")
        return q_current
    
    def get_acceleration_and_gradient(self, q):
        qpp_val = self.F_gg_u(q)
        J_val = self.F_J(q)
        return qpp_val, J_val

    def random_steady_state(self, q0):        
        max_iterations = 1e2       
        x_lim = [-0.5, 0.5]
        y_lim = [-0.5, 0.5]
        z_lim = [0., 1.5]
        yaw_lim = [-np.pi, np.pi]

        iterations = 0
        while iterations < max_iterations:
            iterations += 1

            x = np.random.rand() * (x_lim[1] - x_lim[0]) + x_lim[0]
            y = np.random.rand() * (y_lim[1] - y_lim[0]) + y_lim[0]
            z = np.random.rand() * (z_lim[1] - z_lim[0]) + z_lim[0]
            yaw = np.random.rand() * (yaw_lim[1] - yaw_lim[0]) + yaw_lim[0]
            # Convert Euler angles (roll, pitch, yaw) to rotation matrix
            rotation_matrix = pin.rpy.rpyToMatrix(np.pi, 0, yaw)

            # Create translation vector
            translation_vector = np.array([x, y, z])

            # Create SE3 object from rotation and translation
            H_d = pin.SE3(rotation_matrix, translation_vector)

            # get initial solution 
            H_0_efiiwa = H_d @ np.linalg.inv(self.H_efiiwa_t)
            q_ik, success = self.robotmodel.inverse_kinematics(H_0_efiiwa, q0[:7], True, False)

            if success == False:
                continue
            q_init = np.concatenate([q_ik.flatten(), [0,0]])
            # print(q_init)
            
            q_ss, success = self.inverse_kinematics(H_d, q_init)

            # check if inverse kinematics was successful
            if success == False:
                print("inverse kinematics failed")
                continue

            pin.framesForwardKinematics(self.pin_model, self.pin_data, q_ss)
            H_0_t_ik = self.pin_data.oMf[9].homogeneous
            # d_H = H_d @ np.linalg.inv(H_0_t_ik)

            # err = np.linalg.norm(d_H - np.eye(4), 'fro')
            pos_err = np.linalg.norm(H_d[:3, 3] - H_0_t_ik[:3, 3])
            rot_err = np.linalg.norm(H_d[:3, :3] @ np.linalg.inv(H_0_t_ik[:3, :3]) - np.eye(3), 'fro')
            
            # if q_ss is not None:
            if pos_err < 1e-3:
                
                # print("q_ss: ", q_ss)

                q_ss_aik, success = self.analytic_inverse_kinematics(H_d, q_init)
                if success:
                    return q_ss, True
                else:
                    return None, False
            else:
                continue
        
        print("max iterations reached")
        return None, None
    



if __name__ == "__main__":

    import sspp.src.sspp.SteadyState as ss
    import numpy as np 

    # pinocchio
    pin_model, env = cre.create_robocrane_pinocchio_models()
    pin_data = pin_model.createData()
    tool_frame_id = cre.get_gripper_point_frame_id(pin_model)
    # ef_frame_id = cre.get_endeffector_frame_id(pin_model)
    print("tool_frame_id: ", tool_frame_id)
    # print("ef_frame_id: ", ef_frame_id)

    # pu.print_frame_names(pin_model)

    ss_obj = ss.SteadyState(pin_model, pin_data, tool_frame_id, ef_frame_id)
    # ss_obj.test_objective_function()
    # exit()


    # visualize with mujoco 
    import mujoco
    import time

    mj_model, mj_data, env = cre.create_robocrane_mujoco_models()
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:

        # config viewer
        viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_WIREFRAME] = True
        viewer.opt.label = mujoco.mjtLabel.mjLABEL_SITE # mjLABEL_BODY
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE # mjFRAME_BODY

        while viewer.is_running():
            q = pin.randomConfiguration(pin_model)
            print("q: ", q)
            mj_data.qpos[:9] = q
            mujoco.mj_forward(mj_model, mj_data)
            viewer.sync()

            time.sleep(1)

            # H = pu.get_frameSE3(pin_model, pin_data, q, tool_frame_id).homogeneous
            pin.forwardKinematics(pin_model, pin_data, q)
            pin.updateFramePlacements(pin_model, pin_data)

            H = pin_data.oMf[tool_frame_id].homogeneous
            q_init = q
            q_ik, success = ss_obj.inverse_kinematics(H, q_init)
            q_ik = q_ik
            print("q_ik: ", q_ik)
            
            # mj_data.mocap_pos[0] = pin_data.oMi[-1].translation
            # rotmat = pin_data.oMi[-1].rotation
            mj_data.mocap_pos[0] = pin_data.oMf[tool_frame_id].translation
            rotmat = pin_data.oMf[tool_frame_id].rotation
            quat = Rotation.from_matrix(rotmat).as_quat()
            # flip quaternion
            mj_data.mocap_quat[0] = quat[[3,0,1,2]]

            # print("mocap_pos: ", mj_data.mocap_pos[0]) 
            # print("mocap_quat: ", mj_data.mocap_quat[0])

            print("q_ik: ", q_ik)
            input("Press Enter to execute ik config...")

            if success:
                mj_data.qpos[:9] = q_ik
            mujoco.mj_forward(mj_model, mj_data)
            viewer.sync()
            
            input("Press Enter to continue...")



