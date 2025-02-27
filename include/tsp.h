//
// Created by geraldebmer on 26.02.25
//

#ifndef spp_SAMPLING_PATH_PLANNER_H
#define spp_SAMPLING_PATH_PLANNER_H

#include <vector>
#include <Eigen/Core>
#include <unsupported/Eigen/Splines>
#include <iostream>
#include <random>
#include <omp.h>
#include "mujoco/mujoco.h"
#include "Timer.h"

namespace tsp
{
    static constexpr int kSplineDegree = 2;
    static constexpr int kDOF = 3;
    using Point = Eigen::Matrix<double, kDOF, 1>;
    using Spline = Eigen::Spline<double, kDOF, kSplineDegree>;

    struct CollisionPoint{
        double spline_param;
        double collision_distance;
        Point coll_point;

        CollisionPoint(double spline_param, double coll_dist, Point coll_point)
            : spline_param(spline_param), collision_distance(coll_dist), coll_point(coll_point) {}
    };

    class TaskSpacePlanner
    {
    private:
        using SplineFitter = Eigen::SplineFitting<Spline>;

        Spline path_spline_;
        mjModel *model_;
        mjData *data_;
        char error_buffer_[1000]; // Buffer for storing MuJoCo error messages
        std::vector<double> param_vec;
        std::vector<Point> via_points;
        std::vector<CollisionPoint> coll_pts;

    public:
        TaskSpacePlanner(mjModel *model)
            : model_(model), via_points(), param_vec(), coll_pts()
        {
            data_ = mj_makeData(model_);
            if (!data_)
            {
                throw std::runtime_error("Failed to create MuJoCo data structure.");
            }
        }

        TaskSpacePlanner(const std::string &xml_string) : via_points(), param_vec(), coll_pts()
        {
            // Parse the model from the XML string
            model_ = mj_loadXML(xml_string.c_str(), nullptr, error_buffer_, sizeof(error_buffer_));
            if (!model_)
            {
                throw std::runtime_error("Failed to load MuJoCo model from XML: " + std::string(error_buffer_));
            }

            // Create the mjData structure associated with the model
            data_ = mj_makeData(model_);
            if (!data_)
            {
                throw std::runtime_error("Failed to create MuJoCo data structure.");
            }

        }


        ~TaskSpacePlanner()
        {
            mj_deleteModel(model_);
        }

        int initializePath(const Point &start, const Point &end, const Point &end_derivative, int num_points = 3)
        {
            param_vec.clear();
            via_points.clear();

            // linear placement of via points from start to end
            for (int i = 0; i < num_points; ++i)
            {
                double t = static_cast<double>(i) / (num_points - 1);
                Point point = (1 - t) * start + t * end;
                param_vec.push_back(t);
                via_points.push_back(point);
            }

            Eigen::Map<Eigen::VectorXd> u_knots(param_vec.data(), num_points);
            Eigen::MatrixXd via_mat(kDOF, num_points);
            for(size_t i = 0; i < via_points.size(); i++){
                via_mat.block<3,1>(0,i) = via_points[i];
            }

            Eigen::MatrixXd derivatives = end_derivative;
            Eigen::Vector<int, 1> deriv_ind(num_points -1);
            path_spline_ = SplineFitter::InterpolateWithDerivatives(via_mat, derivatives, deriv_ind, kSplineDegree, u_knots);
            return 0;
        }

        Point evaluate(double u) const
        {
            return path_spline_(u);
        }


        Spline::ControlPointVectorType get_ctrl_pts() const
        {
            return path_spline_.ctrls();
        }

        void perturb(){
            return;
        }

        bool checkCollision(const Spline &spline, int num_samples, mjData *data)
        {
            for (int i = 0; i <= num_samples; ++i)
            {
                double u = static_cast<double>(i) / num_samples;
                Point point = spline(u);

                for (int j = 0; j < kDOF; ++j)
                {
                    data->qpos[j] = point(j);
                }
                mj_forward(model_, data);

                // iterate over all contacts
                for(int i=0; i<data->ncon; i++) {
//                    std::cout << " Collision at sample " << i << " with depth " << data->contact[i].dist << std::endl;
                    auto col_dist = data->contact[i].dist;
                    if (col_dist < -1e-3)
                    {
                        auto nv = model_->nv;
                        auto nJ = data_->nJ;

                        if(nJ>0){
                            int nc = nJ/nv;
                            std::cout << "dimension of constrained jacobian nc: " << nc <<" x nv: " << nv << std::endl;
                            Eigen::Map<Eigen::MatrixXd> J_constr(data->efc_J, nv, nc);
                            std::cout << "J_constr:\n" << J_constr.transpose() << std::endl;
                            std::cout << "ncon: " << data->ncon << std::endl;
                        }

                        CollisionPoint col_pt(u, col_dist, point);
                        coll_pts.push_back(col_pt);
                        std::cout << " Collision at sample " << i << " with depth " << data->contact[i].dist << std::endl;
                        return true;
                    }
                }


            }
            return false;
        }

        double computeArcLength(const Spline &spline, int check_points) const
        {
            double total_length = 0.0;

#pragma omp parallel for reduction(+ : total_length)
            for (int i = 1; i < check_points; ++i)
            {
                double u1 = static_cast<double>(i - 1) / (check_points - 1);
                double u2 = static_cast<double>(i) / (check_points - 1);

                Eigen::VectorXd p1 = spline(u1);
                Eigen::VectorXd p2 = spline(u2);

                total_length += (p2 - p1).norm();
            }

            return total_length;
        }

        bool findBestPath(const std::vector<Spline> &successful_paths, Spline &best_spline, int check_points = 10)
        {
            double min_cost = std::numeric_limits<double>::infinity();
            bool found = false;

//#pragma omp parallel for
            for (size_t i = 0; i < successful_paths.size(); ++i)
            {
                double cost = computeArcLength(successful_paths[i], check_points);
//#pragma omp critical
                {
                    if (cost < min_cost)
                    {
                        min_cost = cost;
                        best_spline = successful_paths[i];
                        found = true;
                    }
                }
            }

            return found;
        }

        bool plan(const Point &start, const Point &end, const Point &end_derivative, double sigma, const Point &limits, int sample_count = 50,
                  int check_points = 50, int init_points = 10)
        {
            coll_pts.clear();
            initializePath(start, end, end_derivative, init_points);

            {

                for (int i = 0; i < sample_count; ++i)
                {
                    perturb();

                    if (!checkCollision(path_spline_, check_points, data_))
                    {
                        std::cout << "ready!" << std::endl;
                    }
                }
            }

            return false;
        }

    };

} // namespace sspp

#endif // spp_SAMPLING_PATH_PLANNER_H
