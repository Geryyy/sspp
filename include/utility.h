//
// Created by geraldebmer on 26.02.25.
//

#ifndef SSPP_UTILITY_H
#define SSPP_UTILITY_H

#include <iostream>
#include <fstream>
#include <Eigen/Dense>

#include <iostream>
#include <Eigen/Geometry>

#include "tsp.h"
#include "mujoco/mujoco.h"


inline Eigen::Vector3d get_body_position(mjModel* m, mjData* d, const std::string& name){
    auto block_id = mj_name2id(m, mjtObj::mjOBJ_BODY, name.c_str());
    //    std::cout << block_name << " id: " << block_id << std::endl;
    Eigen::Vector3d body_pos;
    body_pos << d->xpos[block_id*3], d->xpos[block_id*3+1], d->xpos[block_id*3+2];
    return body_pos;
}

inline int get_body_id(mjModel* m, const std::string& name){
    return mj_name2id(m, mjtObj::mjOBJ_BODY, name.c_str());
}


// Function to get the yaw angle of a geom
inline double get_body_yaw(int geom_id, mjData *mj_data) {
    if (geom_id < 0) {
        std::cerr << "Invalid geom ID!" << std::endl;
        return 0.0;
    }

    // Get quaternion [w, x, y, z] from MuJoCo
    double *xquat = mj_data->xquat + 4 * geom_id;

    // Convert to Eigen quaternion
    Eigen::Quaterniond q(xquat[0], xquat[1], xquat[2], xquat[3]);

    // Convert to Euler angles (ZYX order)
    Eigen::Vector3d euler = q.toRotationMatrix().eulerAngles(2, 1, 0);

    double yaw = euler[0];// * 180.0 / M_PI; // Convert radians to degrees
    return yaw;
}

// Function to add a yaw angle to the existing orientation of a geom
inline Eigen::Quaterniond yaw_to_quat( double yaw_angle) {
    // Compute new yaw rotation (around Z-axis)
    double yaw_radians = yaw_angle;// * M_PI / 180.0; // Convert degrees to radians
    Eigen::AngleAxisd yaw_rotation(yaw_radians, Eigen::Vector3d::UnitZ());

    // Apply new yaw rotation to the current orientation
    Eigen::Quaterniond new_q(yaw_rotation);

    return new_q;
}


inline void exportToCSV(const std::string &filename, const Eigen::VectorXd &x, const Eigen::MatrixXd &y) {
    std::ofstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }

    // Write header
    file << "x";
    for (int j = 0; j < y.rows(); ++j) {
        file << ", y" << j; // Column names: y0, y1, ...
    }
    file << "\n";

    // Write data
    for (int i = 0; i < x.size(); ++i) {
        file << x(i); // X value
        for (int j = 0; j < y.rows(); ++j) {
            file << ", " << y(j, i); // Y values
        }
        file << "\n";
    }

    file.close();
    std::cout << "Data exported to " << filename << std::endl;
}


#endif //SSPP_UTILITY_H
