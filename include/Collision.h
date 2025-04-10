//
// Created by geraldebmer on 10.04.25.
//

#ifndef SSPP_COLLISION_H
#define SSPP_COLLISION_H

#include "mujoco/mujoco.h"
#include "utility.h"
#include <Eigen/Core>
#include <omp.h>


struct BodyJointInfo {
    int body_id = -1;
    int jnt_id = -1;
    int qpos_adr = -1;
    int type = -1;
};

template<typename Point>
class Collision{
    mjModel *mj_model_;
    mjData *mj_data_;
    BodyJointInfo joint_info;

public:
    Collision(std::string coll_body_name, mjModel *mj_model) : mj_model_(mj_model){
        mj_data_ = mj_makeData(mj_model_);
        joint_info = get_free_body_joint_info(coll_body_name, mj_model);
    }

    ~Collision(){
    }

    void set_collision_body(std::string body_name){
        joint_info = get_free_body_joint_info(body_name, mj_model_);
    }

    
    BodyJointInfo get_free_body_joint_info(const std::string& body_name, mjModel* m){

        BodyJointInfo info;
    
        info.body_id = mj_name2id(m, mjOBJ_BODY, body_name.c_str());
        if (info.body_id == -1) {
            std::cerr << "Error: Body with name '" << body_name << "' not found." << std::endl;
            return info; // Return with invalid body_id
        }
    
        info.jnt_id = m->body_jntadr[info.body_id];
        if (info.jnt_id == -1) {
            std::cerr << "Error: Body '" << body_name << "' has no joint." << std::endl;
            return info; // Return with invalid jnt_id
        }
    
        if (m->jnt_type[info.jnt_id] != mjJNT_FREE) {
            std::cerr << "Error: Joint of body '" << body_name << "' is not a free joint." << std::endl;
            return info; // Return with invalid jnt_id
        }
    
        info.qpos_adr = m->jnt_qposadr[info.jnt_id];
        info.type = m->jnt_type[info.jnt_id];
    
        if (info.type != mjtJoint::mjJNT_FREE) {
            std::cerr << "Error: Free joint of body '" << body_name << "' does not have 6 degrees of freedom (pos + orientation)." << std::endl;
            return info; // Return with invalid dof
        }
    
        return info;
    }

    static double get_geom_center_distance(int contact_id, mjData *data) {
        int geom1_id = data->contact[contact_id].geom1;
        int geom2_id = data->contact[contact_id].geom2;
        Point geom1_center, geom2_center;
        geom1_center << data->geom_xpos[geom1_id * 3], data->geom_xpos[geom1_id * 3 + 1], data->geom_xpos[
                geom1_id * 3 + 2], 0.0;
        geom2_center << data->geom_xpos[geom2_id * 3], data->geom_xpos[geom2_id * 3 + 1], data->geom_xpos[
                geom2_id * 3 + 2], 0.0;
        return (geom2_center - geom1_center).norm();
    }


    void mj_set_point(const Point &point, mjData *mj_data){
        if (point.size() != 4) {
            std::cerr << "Error: Input point must have 4 elements (x, y, z, yaw) for a free joint." << std::endl;
            return;
        }

        if (joint_info.body_id == -1 || joint_info.jnt_id == -1 || joint_info.qpos_adr == -1 || joint_info.type != mjtJoint::mjJNT_FREE) {
            return; // An error occurred in get_free_body_joint_info
        }
    
        // Set position
        for (int i = 0; i < 3; ++i) {
            mj_data->qpos[joint_info.qpos_adr + i] = point(i);
        }
    
        // Set orientation (yaw to quaternion)
        Eigen::Quaterniond quat = yaw_to_quat(point(3));
        mj_data->qpos[joint_info.qpos_adr + 3] = quat.w();
        mj_data->qpos[joint_info.qpos_adr + 4] = quat.x();
        mj_data->qpos[joint_info.qpos_adr + 5] = quat.y();
        mj_data->qpos[joint_info.qpos_adr + 6] = quat.z();
    }


    bool check_collision_point(const Point &pt) {
        mjData *mj_data = mj_data_;
        // mjData *mj_data = data_copies_[omp_get_thread_num()];
        Point point = pt;

        mj_set_point(point, mj_data);
        mj_forward(mj_model_, mj_data);

        for (int i = 0; i < mj_data->ncon; i++) {
            auto col_dist = mj_data->contact[i].dist;
            if (col_dist < -1e-3) {
                return true;
            }
        }
        return false;
    }

    double collision_point_cost(const Point &point, bool use_center_dist = true){
        double cost = 0.0;

        mjData *mj_data = mj_data_;

        mj_set_point(point, mj_data);
        mj_forward(mj_model_, mj_data);

        for (int i = 0; i < mj_data->ncon; i++) {
            double col_dist = mj_data->contact[i].dist;
            double center_dist = get_geom_center_distance(i, mj_data);

            if (col_dist < -1e-3) {
                if (use_center_dist) {
                    constexpr double lambda = 1e-4;
                    cost += -1 / (center_dist + lambda);
                } else {
                    cost += -col_dist;
                }
            }
        }

        return cost;
    }

};
#endif //SSPP_COLLISION_H
