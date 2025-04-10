//
// Created by geraldebmer on 10.04.25.
//

#ifndef SSPP_COLLISION_H
#define SSPP_COLLISION_H

#include "mujoco/mujoco.h"
#include "utility.h"
#include <Eigen/Core>
#include <omp.h>

template<typename Point>
class Collision{
    mjModel *mj_model_;
    mjData *mj_data_;

public:
    Collision(mjModel *mj_model) : mj_model_(mj_model){
        mj_data_ = mj_makeData(mj_model_);
    }

    ~Collision(){
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
        for (int j = 0; j < 3; ++j) {
            mj_data->qpos[j] = point(j);
        }
        auto quat = yaw_to_quat(point[3], mj_data);
        mj_data->qpos[3] = quat.w();
        mj_data->qpos[4] = quat.x();
        mj_data->qpos[5] = quat.y();
        mj_data->qpos[6] = quat.z();
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
