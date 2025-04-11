//
// Created by geraldebmer on 10.04.25.
//

#ifndef SSPP_COLLISION_H
#define SSPP_COLLISION_H

#include "mujoco/mujoco.h"
#include "utility.h" // Include the updated utility header
#include <Eigen/Core>
#include <omp.h>

template<typename Point>
class Collision{
    mjModel *mj_model_;
    mjData *mj_data_;
    Utility::BodyJointInfo joint_info; // Use the struct from the namespace

public:
    Collision(std::string coll_body_name, mjModel *mj_model) : mj_model_(mj_model){
        mj_data_ = mj_makeData(mj_model_);
        joint_info = Utility::get_free_body_joint_info(coll_body_name, mj_model); // Use the namespaced function
    }

    ~Collision(){
        mj_deleteData(mj_data_); // Clean up mjData in the destructor
    }

    void set_collision_body(std::string body_name){
        joint_info = Utility::get_free_body_joint_info(body_name, mj_model_); // Use the namespaced function
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


    void mj_set_point(const Point &point){
        Utility::mj_set_point(point, joint_info, mj_data_); // Use the namespaced function
    }


    bool check_collision_point(const Point &pt) {
        mj_set_point(pt);
        mj_forward(mj_model_, mj_data_);

        for (int i = 0; i < mj_data_->ncon; i++) {
            auto col_dist = mj_data_->contact[i].dist;
            if (col_dist < -1e-3) {

                int geom1_id = mj_data_->contact[i].geom1;
                int geom2_id = mj_data_->contact[i].geom2;

                const char* geom1_name = mj_id2name(mj_model_, mjOBJ_GEOM, geom1_id);
                const char* geom2_name = mj_id2name(mj_model_, mjOBJ_GEOM, geom2_id);

                std::cout << "check_collision_point(): Collision detected!" << std::endl;
                std::cout << "  Contact distance (col_dist): " << col_dist << std::endl;
                std::cout << "  Geom 1 ID: " << geom1_id;
                if (geom1_name) {
                    std::cout << " (Name: " << geom1_name << ")";
                }
                std::cout << std::endl;
                std::cout << "  Geom 2 ID: " << geom2_id;
                if (geom2_name) {
                    std::cout << " (Name: " << geom2_name << ")";
                }
                std::cout << std::endl;

                return true;
            }
        }
        return false;
    }

    double collision_point_cost(const Point &point, bool use_center_dist = true){
        mj_set_point(point);
        mj_forward(mj_model_, mj_data_);

        double cost = 0.0;
        for (int i = 0; i < mj_data_->ncon; i++) {
            double col_dist = mj_data_->contact[i].dist;
            double center_dist = get_geom_center_distance(i, mj_data_);

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