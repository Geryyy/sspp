//
// Created by geraldebmer on 26.02.25
//

#ifndef spp_SAMPLING_PATH_PLANNER_H
#define spp_SAMPLING_PATH_PLANNER_H

#include "Gradient.h"
#include <vector>
#include <Eigen/Core>
#include <unsupported/Eigen/Splines>
#include <iostream>
#include <random>
#include <omp.h>
#include "mujoco/mujoco.h"
#include "Timer.h"
#include <random>

namespace tsp
{
    static constexpr int kSplineDegree = 2;
    static constexpr int kDOF = 3;
    using Point = Eigen::Matrix<double, kDOF, 1>;
    using Spline = Eigen::Spline<double, kDOF, kSplineDegree>;

    struct CollisionPoint
    {
        double spline_param;
        double collision_distance;
        Point coll_point;

        CollisionPoint(double spline_param, double coll_dist, Point coll_point)
            : spline_param(spline_param), collision_distance(coll_dist), coll_point(coll_point)
        {
        }
    };



    struct PathCandidate
    {
        Point via_point;
        std::vector<GradientStep> gradient_steps;
        SolverStatus status;
    };

    /* TODO: add start_derivative_ */
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
        // std::vector<CollisionPoint> coll_pts;
        Point end_derivative_;

        // statistics
        std::vector<PathCandidate> successful_candidates_;
        std::vector<PathCandidate> failed_candidates_;

    public:
        TaskSpacePlanner(mjModel *model)
            : model_(model), via_points(), param_vec()
        {
            data_ = mj_makeData(model_);
            if (!data_)
            {
                throw std::runtime_error("Failed to create MuJoCo data structure.");
            }
        }

        TaskSpacePlanner(const std::string &xml_string) : via_points(), param_vec()
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

        Spline initializePath(const Point &start, const Point &end, const Point &end_derivative, int num_points = 3)
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
            end_derivative_ = end_derivative;
            Eigen::Map<Eigen::VectorXd> u_knots(param_vec.data(), num_points);
            Eigen::MatrixXd via_mat(kDOF, num_points);
            for (size_t i = 0; i < via_points.size(); i++)
            {
                via_mat.block<3, 1>(0, i) = via_points[i];
            }

            Eigen::MatrixXd derivatives = end_derivative;
            Eigen::Vector<int, 1> deriv_ind(num_points - 1);
            Spline spline = SplineFitter::InterpolateWithDerivatives(via_mat, derivatives, deriv_ind, kSplineDegree,
                                                                     u_knots);
            return spline;
        }

        Spline path_from_via_pt(const Point &via_pt)
        {
            Eigen::Map<Eigen::VectorXd> u_knots(param_vec.data(), param_vec.size());
            Eigen::Map<Eigen::MatrixXd> via_mat(reinterpret_cast<double *>(via_points.data()),
                                                kDOF, via_points.size());

            Eigen::MatrixXd via_mat_copy = via_mat;
            via_mat_copy.block<3, 1>(0, 1) = via_pt;

            Eigen::MatrixXd derivatives = end_derivative_;
            Eigen::Vector<int, 1> deriv_ind(via_points.size() - 1);
            auto spline = SplineFitter::InterpolateWithDerivatives(via_mat_copy, derivatives, deriv_ind, kSplineDegree,
                                                                   u_knots);
            return spline;
        }

        Point get_random_point(const Point &mean, const Point &stddev)
        {
            static std::random_device rd;
            static std::mt19937 gen(rd());

            std::normal_distribution<double> dist_x(mean.x(), stddev.x());
            std::normal_distribution<double> dist_y(mean.y(), stddev.y());
            std::normal_distribution<double> dist_z(mean.z(), stddev.z());

            return Point(dist_x(gen), dist_y(gen), dist_z(gen));
        }

        Point evaluate(double u, const Spline &spline)
        {
            return spline(u);
        }

        Point evaluate(double u) const
        {
            return path_spline_(u);
        }

        std::vector<Point> get_via_pts() const
        {
            return via_points;
        }

        Spline::ControlPointVectorType get_ctrl_pts() const
        {
            return path_spline_.ctrls();
        }

        double get_geom_center_distance(int contact_id, mjData *data)
        {
            int geom1_id = data->contact[contact_id].geom1;
            int geom2_id = data->contact[contact_id].geom2;
            Point geom1_center, geom2_center;
            geom1_center << data->geom_xpos[geom1_id * 3], data->geom_xpos[geom1_id * 3 + 1], data->geom_xpos[geom1_id * 3 + 2];
            geom2_center << data->geom_xpos[geom2_id * 3], data->geom_xpos[geom2_id * 3 + 1], data->geom_xpos[geom2_id * 3 + 2];
            return (geom2_center - geom1_center).norm();
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
                for (int i = 0; i < data->ncon; i++)
                {
                    //                    std::cout << " Collision at sample " << i << " with depth " << data->contact[i].dist << std::endl;
                    auto col_dist = data->contact[i].dist;
                    if (col_dist < -1e-3)
                    {
                        auto nv = model_->nv;
                        auto nJ = data_->nJ;

                        if (nJ > 0)
                        {
                            int nc = nJ / nv;
                            std::cout << "dimension of constrained jacobian nc: " << nc << " x nv: " << nv << std::endl;
                            Eigen::Map<Eigen::MatrixXd> J_constr(data->efc_J, nv, nc);
                            std::cout << "J_constr:\n"
                                      << J_constr.transpose() << std::endl;
                            std::cout << "ncon: " << data->ncon << std::endl;
                        }

                        CollisionPoint col_pt(u, col_dist, point);
                        // coll_pts.push_back(col_pt);
                        std::cout << " Collision at sample " << i << " with depth " << data->contact[i].dist << std::endl;
                        return true;
                    }
                }
            }
            return false;
        }

        /* TODO: make moveable object selectable --> adapt qpos range to update */

        double collision_cost(const Point via_pt, int eval_cnt, mjData *data, bool use_center_dist = true)
        {
            Spline spline = path_from_via_pt(via_pt);
            double cost = 0.0;

            for (int i = 0; i <= eval_cnt; ++i)
            {
                double u = static_cast<double>(i) / eval_cnt;
                Point point = spline(u);

                for (int j = 0; j < kDOF; ++j)
                {
                    data->qpos[j] = point(j);
                }
                mj_forward(model_, data);

                for (int i = 0; i < data->ncon; i++)
                {
                    double col_dist = data->contact[i].dist;
                    double center_dist = get_geom_center_distance(i, data);
                    if (col_dist < -1e-3)
                    {
                        if (use_center_dist)
                        {
                            cost += -center_dist;
                        }
                        else
                        {
                            cost += -col_dist;
                        }
                    }
                }
            }
            return cost;
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

        bool findBestPath(const std::vector<PathCandidate> &candidates, Spline &best_spline, int check_points = 10)
        {
            double min_cost = std::numeric_limits<double>::infinity();
            bool found = false;

            // #pragma omp parallel for
            for(const auto& candidate : candidates) {
                auto path_candidate = path_from_via_pt(candidate.via_point);
                double cost = computeArcLength(path_candidate, check_points);
                if (cost < min_cost)
                {
                    min_cost = cost;
                    best_spline = path_candidate;
                    found = true;
                }
            }
            return found;
        }


        void collision_optimization(const std::vector<Point> &via_point_candidates,
            std::vector<PathCandidate> &successful_candidates,
            std::vector<PathCandidate> &failed_candidates,
            mjData* mj_data,
            int sample_count, int check_points, int gd_iterations) {
            {
                for (int i = 0; i < sample_count; ++i)
                {
                    const Point via_candidate = via_point_candidates[i];
                    // Lambda function for collision cost
                    auto collision_cost_lambda = [&](const Eigen::Vector3d &via_pt)
                    {
                        return collision_cost(via_pt, check_points, mj_data);
                    };

                    /* gradient descent steps */
                    Point diff_delta = Point::Ones() * 1e-2;
                    constexpr double step_size = 1e-3;
                    GradientDescent graddesc(step_size, gd_iterations, collision_cost_lambda, diff_delta);
                    auto solver_status = graddesc.optimize(via_candidate);
                    const auto via_pt_opt = graddesc.get_result();

                    std::cout << "solver status: " << SolverStatustoString(solver_status) << std::endl;

                    if(solver_status == SolverStatus::Converged){
                        PathCandidate candidate(via_pt_opt, graddesc.get_gradient_descent_steps(), solver_status);
                        successful_candidates.push_back(candidate);
                    }
                    else {
                        PathCandidate candidate(via_pt_opt, graddesc.get_gradient_descent_steps(), solver_status);
                        failed_candidates.push_back(candidate);
                    }
                }
            }
        }

        void arclength_optimization(std::vector<PathCandidate> &path_candidates, std::vector<PathCandidate> &optimized_candidates,
            mjData* mj_data, int check_points, int gd_iterations)
            {
                for(const auto& candidate : path_candidates) {
                    const Point via_candidate = candidate.via_point;
                    // Lambda function for collision cost
                    auto arc_lengthcost_lambda = [&](const Eigen::Vector3d &via_pt)
                    {
                        auto spline = path_from_via_pt(via_pt);
                        double col_cost = collision_cost(via_pt, check_points, mj_data);
                        double len_cost = computeArcLength(spline, check_points);
                        return len_cost;// + col_cost*col_cost;
                    };

                    /* gradient descent steps */
                    Point diff_delta = Point::Ones() * 1e-2;
                    constexpr double step_size = 1e-3;
                    GradientDescent graddesc(step_size, gd_iterations, arc_lengthcost_lambda, diff_delta);
                    auto solver_status = graddesc.optimize(via_candidate);
                    const auto via_pt_opt = graddesc.get_result();

                    std::cout << "solver status: " << SolverStatustoString(solver_status) << std::endl;

                    if(solver_status == SolverStatus::Converged){
                        PathCandidate cand(via_pt_opt, graddesc.get_gradient_descent_steps(), solver_status);
                        optimized_candidates.push_back(cand);
                    }
                    else {
                        PathCandidate cand(via_pt_opt, graddesc.get_gradient_descent_steps(), solver_status);
                        optimized_candidates.push_back(cand);
                    }
                }
        }

        std::vector<PathCandidate> plan(const Point &start,
                                        const Point &end, const Point &end_derivative, double sigma, const Point &limits,
                                        const int sample_count = 50,
                                        const int check_points = 50,
                                        const int gd_iterations = 10,
                                        const int init_points = 3)
        {
            // coll_pts.clear();
            /* initialize straight line */
            Spline init_spline = initializePath(start, end, end_derivative, init_points);
            path_spline_ = init_spline;
            Point init_via_pt = via_points[1];

            /* create random ensemble */
            std::vector<Point> via_point_candidates;
            via_point_candidates.push_back(init_via_pt);


            for (int i = 0; i < sample_count - 1; i++)
            {
                Point stddev = Point::Ones() * sigma;
                Point noisy_via_pt = get_random_point(init_via_pt, stddev);
                via_point_candidates.push_back(noisy_via_pt);
                // std::cout << "random via point[" << i << "]: " << noisy_via_pt.transpose() << std::endl;
            }

            /* optimize candidates for collision */
            collision_optimization(via_point_candidates, successful_candidates_, failed_candidates_,
                data_, sample_count, check_points, gd_iterations);


            /* tighten succesful paths */
            std::vector<PathCandidate> opt_candidates;
            arclength_optimization(successful_candidates_, opt_candidates, data_, check_points, gd_iterations);

            // test!!
            failed_candidates_ = opt_candidates;

            /* find best path */
            findBestPath(successful_candidates_, path_spline_, check_points);

            return successful_candidates_;
        }

        std::vector<PathCandidate> get_succesful_path_candidates() {
            return successful_candidates_;
        }

        std::vector<PathCandidate> get_failed_path_candidates() {
            return failed_candidates_;
        }

        std::vector<Point> get_path_pts(const Spline &spline, const int pts_cnt = 10)
        {
            std::vector<Point> pts;
            for (int i = 0; i < pts_cnt; i++)
            {
                double u = static_cast<double>(i) / (pts_cnt-1);
                auto pt = evaluate(u, spline);
                pts.push_back(pt);
                //        std::cout << "pt("<<u<<") " << pt.transpose() << std::endl;
            }
            return pts;
        }

        std::vector<Point> get_path_pts(const int pts_cnt = 10)
        {
            std::vector<Point> pts;
            for (int i = 0; i < pts_cnt; i++)
            {
                double u = static_cast<double>(i) / (pts_cnt-1);
                auto pt = evaluate(u);
                pts.push_back(pt);
                //        std::cout << "pt("<<u<<") " << pt.transpose() << std::endl;
            }
            return pts;
        }
    };
} // namespace sspp

#endif // spp_SAMPLING_PATH_PLANNER_H
