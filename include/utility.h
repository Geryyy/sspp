//
// Created by geraldebmer on 26.02.25.
//

#ifndef SSPP_UTILITY_H
#define SSPP_UTILITY_H

#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <Eigen/Geometry>
// #include "tsp.h"
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

    inline void set_body_free_joint(const std::string& body_name, mjModel* m, mjData* d,
                                const Eigen::Vector3d& pos, const Eigen::Quaterniond& quat)
    {
        BodyJointInfo info = get_free_body_joint_info(body_name, m);

        if (info.body_id == -1)
        {
            std::cerr << "Error: Body '" << body_name << "' not found." << std::endl;
            return;
        }

        if (info.qpos_adr + 6 >= m->nq)
        {
            std::cerr << "Error: Invalid qpos address for body '" << body_name << "'." << std::endl;
            return;
        }

        // Set position
        d->qpos[info.qpos_adr + 0] = pos.x();
        d->qpos[info.qpos_adr + 1] = pos.y();
        d->qpos[info.qpos_adr + 2] = pos.z();

        // Set orientation (quaternion)
        d->qpos[info.qpos_adr + 3] = quat.w();
        d->qpos[info.qpos_adr + 4] = quat.x();
        d->qpos[info.qpos_adr + 5] = quat.y();
        d->qpos[info.qpos_adr + 6] = quat.z();
    }

    inline void print_body_info(const mjModel* m) {
        if (!m) {
            std::cerr << "Error: MuJoCo model is null." << std::endl;
            return;
        }

        int n_bodies = m->nbody;
        std::cout << "Number of bodies: " << n_bodies << std::endl;

        for (int i = 0; i < n_bodies; ++i) {
            int body_id = i;
            const char* body_name = mj_id2name(m, mjOBJ_BODY, body_id);
            int joint_id = m->body_jntadr[body_id];
            mjtJoint joint_type = mjJNT_FREE;
            const char* joint_type_str = "none";

            if (joint_id != -1) {
                joint_type = static_cast<mjtJoint>(m->jnt_type[joint_id]);
                switch (joint_type) {
                    case mjJNT_FREE:
                        joint_type_str = "free";
                        break;
                    case mjJNT_HINGE:
                        joint_type_str = "hinge";
                        break;
                    case mjJNT_SLIDE:
                        joint_type_str = "slide";
                        break;
                    case mjJNT_BALL:
                        joint_type_str = "ball";
                        break;
                    default:
                        joint_type_str = "unknown";
                        break;
                }
            }

            std::cout << "Body ID: " << body_id << ", Name: ";
            if (body_name) {
                std::cout << body_name;
            } else {
                std::cout << "(unnamed)";
            }
            std::cout << ", Joint Type: " << joint_type_str << std::endl;
        }
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


    inline double quat_to_yaw(const Eigen::Quaterniond& q) {
        Eigen::Matrix3d rotation_matrix = q.toRotationMatrix();
        // Extract Euler angles (roll, pitch, yaw) from the rotation matrix
        // Using the convention ZYX for Euler angles.
        // yaw (around Z-axis) is the angle we are interested in.
    
        // Check for singularities (gimbal lock)
        if (std::abs(rotation_matrix(2, 0)) >= 1.0) {
            // Handle the case where pitch is +/- 90 degrees.
            // In this case, roll and yaw are coupled.
            // We can arbitrarily set roll to 0 and calculate yaw.
            return std::atan2(rotation_matrix(0, 1), rotation_matrix(0, 0));
        } else {
            // General case:
            return std::atan2(rotation_matrix(1, 0), rotation_matrix(0, 0));
        }
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