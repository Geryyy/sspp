//
// Created by geraldebmer on 26.02.25.
//

#ifndef SSPP_UTILITY_H
#define SSPP_UTILITY_H

#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include "tsp.h"
#include "mujoco/mujoco.h"

namespace Utility
{

    // --- MuJoCo Related Utilities ---

    struct BodyJointInfo
    {
        int body_id = -1;
        int jnt_id = -1;
        int qpos_adr = -1;
        int type = -1;
    };


    inline int get_body_id(mjModel *m, const std::string &name)
    {
        return mj_name2id(m, mjtObj::mjOBJ_BODY, name.c_str());
    }

    inline BodyJointInfo get_free_body_joint_info(const std::string &body_name, mjModel *m)
    {
        BodyJointInfo info;

        info.body_id = mj_name2id(m, mjOBJ_BODY, body_name.c_str());
        if (info.body_id == -1)
        {
            std::cerr << "Error: Body with name '" << body_name << "' not found." << std::endl;
            return info; // Return with invalid body_id
        }

        info.jnt_id = m->body_jntadr[info.body_id];
        if (info.jnt_id == -1)
        {
            std::cerr << "Error: Body '" << body_name << "' has no joint." << std::endl;
            return info; // Return with invalid jnt_id
        }

        if (m->jnt_type[info.jnt_id] != mjJNT_FREE)
        {
            std::cerr << "Error: Joint of body '" << body_name << "' is not a free joint." << std::endl;
            return info; // Return with invalid jnt_id
        }

        info.qpos_adr = m->jnt_qposadr[info.jnt_id];
        info.type = m->jnt_type[info.jnt_id];

        if (info.type != mjtJoint::mjJNT_FREE)
        {
            std::cerr << "Error: Joint of body '" << body_name << "' is not a free joint." << std::endl;
            return info; // Return with invalid dof
        }

        return info;
    }

    // Forward declaration of yaw_to_quat
    Eigen::Quaterniond yaw_to_quat(double yaw_angle);

    template <typename Point>
    void mj_set_point(const Point &point, const BodyJointInfo &joint_info, mjData *mj_data)
    {
        if (point.size() != 4)
        {
            std::cerr << "Error: Input point must have 4 elements (x, y, z, yaw) for a free joint." << std::endl;
            return;
        }

        if (joint_info.body_id == -1 || joint_info.jnt_id == -1 || joint_info.qpos_adr == -1 || joint_info.type != mjtJoint::mjJNT_FREE)
        {
            return; // An error occurred in get_free_body_joint_info
        }

        // Set position
        for (int i = 0; i < 3; ++i)
        {
            mj_data->qpos[joint_info.qpos_adr + i] = point(i);
        }

        // Set orientation (yaw to quaternion)
        Eigen::Quaterniond quat = yaw_to_quat(point(3));
        mj_data->qpos[joint_info.qpos_adr + 3] = quat.w();
        mj_data->qpos[joint_info.qpos_adr + 4] = quat.x();
        mj_data->qpos[joint_info.qpos_adr + 5] = quat.y();
        mj_data->qpos[joint_info.qpos_adr + 6] = quat.z();
    }

    inline double get_body_yaw(int geom_id, mjData *mj_data)
    {
        if (geom_id < 0)
        {
            std::cerr << "Invalid geom ID!" << std::endl;
            return 0.0;
        }

        // Get quaternion [w, x, y, z] from MuJoCo
        double *xquat = mj_data->xquat + 4 * geom_id;

        // Convert to Eigen quaternion
        Eigen::Quaterniond q(xquat[0], xquat[1], xquat[2], xquat[3]);

        // Convert to Euler angles (ZYX order)
        Eigen::Vector3d euler = q.toRotationMatrix().eulerAngles(2, 1, 0);

        double yaw = euler[0];
        return yaw;
    }

    inline Eigen::Quaterniond yaw_to_quat(double yaw)
    {
        Eigen::AngleAxisd rollAngle(0.0, Eigen::Vector3d::UnitX());
        Eigen::AngleAxisd pitchAngle(0.0, Eigen::Vector3d::UnitY());
        Eigen::AngleAxisd yawAngle(yaw, Eigen::Vector3d::UnitZ());

        Eigen::Quaterniond q = yawAngle * pitchAngle * rollAngle;
        return q;
    }

    template <typename Point>
    inline Point get_body_point(mjModel *m, mjData *d, const std::string &body_name)
    {
        BodyJointInfo joint_info = get_free_body_joint_info(body_name, m);
        Point body_point;

        if (joint_info.type == mjtJoint::mjJNT_FREE && joint_info.qpos_adr != -1)
        {
            body_point.resize(4);
            body_point(0) = d->qpos[joint_info.qpos_adr + 0]; // x
            body_point(1) = d->qpos[joint_info.qpos_adr + 1]; // y
            body_point(2) = d->qpos[joint_info.qpos_adr + 2]; // z

            // Get quaternion [w, x, y, z]
            Eigen::Quaterniond quat(d->qpos[joint_info.qpos_adr + 3],
                                    d->qpos[joint_info.qpos_adr + 4],
                                    d->qpos[joint_info.qpos_adr + 5],
                                    d->qpos[joint_info.qpos_adr + 6]);

            // Extract yaw angle (ZYX Euler angles)
            Eigen::Vector3d euler = quat.toRotationMatrix().eulerAngles(2, 1, 0);
            body_point(3) = euler(0); // yaw
        }
        else
        {
            std::cerr << "Error: Body '" << body_name << "' is not a free joint or has invalid qpos address." << std::endl;
            body_point.resize(4);
            body_point.setZero(); // Return a zero point to indicate an error
        }

        return body_point;
    }

    // --- General Utilities ---

    inline void exportToCSV(const std::string &filename, const Eigen::VectorXd &x, const Eigen::MatrixXd &y)
    {
        std::ofstream file(filename);

        if (!file.is_open())
        {
            std::cerr << "Error: Could not open file " << filename << std::endl;
            return;
        }

        // Write header
        file << "x";
        for (int j = 0; j < y.rows(); ++j)
        {
            file << ", y" << j; // Column names: y0, y1, ...
        }
        file << "\n";

        // Write data
        for (int i = 0; i < x.size(); ++i)
        {
            file << x(i); // X value
            for (int j = 0; j < y.rows(); ++j)
            {
                file << ", " << y(j, i); // Y values
            }
            file << "\n";
        }

        file.close();
        std::cout << "Data exported to " << filename << std::endl;
    }

} // namespace Utility

#endif // SSPP_UTILITY_H