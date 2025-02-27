//
// Created by geraldebmer on 26.02.25
//

#ifndef spp_SAMPLING_PATH_PLANNER_H
#define spp_SAMPLING_PATH_PLANNER_H

#include <Eigen/Dense>
#include <functional>
#include <vector>
#include <Eigen/Core>
#include <unsupported/Eigen/Splines>
#include <iostream>
#include <random>
#include <omp.h>
#include "mujoco/mujoco.h"
#include "Timer.h"

namespace tsp {
    static constexpr int kSplineDegree = 2;
    static constexpr int kDOF = 3;
    using Point = Eigen::Matrix<double, kDOF, 1>;
    using Spline = Eigen::Spline<double, kDOF, kSplineDegree>;

    struct CollisionPoint {
        double spline_param;
        double collision_distance;
        Point coll_point;

        CollisionPoint(double spline_param, double coll_dist, Point coll_point)
            : spline_param(spline_param), collision_distance(coll_dist), coll_point(coll_point) {
        }
    };


    class Gradient {
    public:
        using CostFunction = std::function<double(const Eigen::Vector3d &)>;

        Gradient(CostFunction L, const Eigen::Vector3d &step, const Eigen::Vector3d &point)
            : L(L), step(step), point(point) {
        }

        Eigen::Vector3d compute() const {
            Eigen::Vector3d grad;
            for (int i = 0; i < 3; ++i) {
                Eigen::Vector3d step_vec = Eigen::Vector3d::Zero();
                step_vec[i] = step[i];
                grad[i] = (L(point + step_vec) - L(point - step_vec)) / (2.0 * step[i]);
            }
            return grad;
        }

    private:
        CostFunction L;
        Eigen::Vector3d step;
        Eigen::Vector3d point;
    };


    class TaskSpacePlanner {
    private:
        using SplineFitter = Eigen::SplineFitting<Spline>;

        Spline path_spline_;
        mjModel *model_;
        mjData *data_;
        char error_buffer_[1000]; // Buffer for storing MuJoCo error messages
        std::vector<double> param_vec;
        std::vector<Point> via_points;
        std::vector<CollisionPoint> coll_pts;
        Point end_derivative_;
        Point coll_gradient;

    public:
        TaskSpacePlanner(mjModel *model)
            : model_(model), via_points(), param_vec(), coll_pts() {
            data_ = mj_makeData(model_);
            if (!data_) {
                throw std::runtime_error("Failed to create MuJoCo data structure.");
            }
        }

        TaskSpacePlanner(const std::string &xml_string) : via_points(), param_vec(), coll_pts() {
            // Parse the model from the XML string
            model_ = mj_loadXML(xml_string.c_str(), nullptr, error_buffer_, sizeof(error_buffer_));
            if (!model_) {
                throw std::runtime_error("Failed to load MuJoCo model from XML: " + std::string(error_buffer_));
            }

            // Create the mjData structure associated with the model
            data_ = mj_makeData(model_);
            if (!data_) {
                throw std::runtime_error("Failed to create MuJoCo data structure.");
            }
        }

        ~TaskSpacePlanner() {
            mj_deleteModel(model_);
        }

        int initializePath(const Point &start, const Point &end, const Point &end_derivative, int num_points = 3) {
            param_vec.clear();
            via_points.clear();

            // linear placement of via points from start to end
            for (int i = 0; i < num_points; ++i) {
                double t = static_cast<double>(i) / (num_points - 1);
                Point point = (1 - t) * start + t * end;
                param_vec.push_back(t);
                via_points.push_back(point);
            }
            end_derivative_ = end_derivative;
            Eigen::Map<Eigen::VectorXd> u_knots(param_vec.data(), num_points);
            Eigen::MatrixXd via_mat(kDOF, num_points);
            for (size_t i = 0; i < via_points.size(); i++) {
                via_mat.block<3, 1>(0, i) = via_points[i];
            }

            Eigen::MatrixXd derivatives = end_derivative;
            Eigen::Vector<int, 1> deriv_ind(num_points - 1);
            path_spline_ = SplineFitter::InterpolateWithDerivatives(via_mat, derivatives, deriv_ind, kSplineDegree,
                                                                    u_knots);
            return 0;
        }

        Spline path_from_via_pt(const Point &via_pt) {
            Eigen::Map<Eigen::VectorXd> u_knots(param_vec.data(), param_vec.size());
            Eigen::Map<Eigen::MatrixXd> via_mat(reinterpret_cast<double*>(via_points.data()),
                                        kDOF, via_points.size());

            Eigen::MatrixXd via_mat_copy = via_mat;
            via_mat_copy.block<3,1>(0,1) = via_pt;

            Eigen::MatrixXd derivatives = end_derivative_;
            Eigen::Vector<int, 1> deriv_ind(via_points.size() - 1);
            auto spline = SplineFitter::InterpolateWithDerivatives(via_mat_copy, derivatives, deriv_ind, kSplineDegree,
                                                                    u_knots);
            return spline;
        }

        Point evaluate(double u, const Spline& spline) {
            return spline(u);
        }

        Point evaluate(double u) const {
            return path_spline_(u);
        }

        std::vector<Point> get_via_pts() const {
            return via_points;
        }

        Spline::ControlPointVectorType get_ctrl_pts() const {
            return path_spline_.ctrls();
        }

        Point get_coll_gradient() const {
            return coll_gradient;
        }

        void perturb() {
            return;
        }

        bool checkCollision(const Spline &spline, int num_samples, mjData *data) {
            for (int i = 0; i <= num_samples; ++i) {
                double u = static_cast<double>(i) / num_samples;
                Point point = spline(u);

                for (int j = 0; j < kDOF; ++j) {
                    data->qpos[j] = point(j);
                }
                mj_forward(model_, data);

                // iterate over all contacts
                for (int i = 0; i < data->ncon; i++) {
                    //                    std::cout << " Collision at sample " << i << " with depth " << data->contact[i].dist << std::endl;
                    auto col_dist = data->contact[i].dist;
                    if (col_dist < -1e-3) {
                        auto nv = model_->nv;
                        auto nJ = data_->nJ;

                        if (nJ > 0) {
                            int nc = nJ / nv;
                            std::cout << "dimension of constrained jacobian nc: " << nc << " x nv: " << nv << std::endl;
                            Eigen::Map<Eigen::MatrixXd> J_constr(data->efc_J, nv, nc);
                            std::cout << "J_constr:\n" << J_constr.transpose() << std::endl;
                            std::cout << "ncon: " << data->ncon << std::endl;
                        }

                        CollisionPoint col_pt(u, col_dist, point);
                        coll_pts.push_back(col_pt);
                        std::cout << " Collision at sample " << i << " with depth " << data->contact[i].dist <<
                                std::endl;
                        return true;
                    }
                }
            }
            return false;
        }


        double collision_cost(const Point& via_pt, int eval_cnt, mjData* data) {
            Spline spline = path_from_via_pt(via_pt);
            double cost = 0.0;

            for (int i = 0; i <= eval_cnt; ++i) {
                double u = static_cast<double>(i) / eval_cnt;
                Point point = spline(u);

                for (int j = 0; j < kDOF; ++j) {
                    data->qpos[j] = point(j);
                }
                mj_forward(model_, data);

                for (int i = 0; i < data->ncon; i++) {
                    double col_dist = data->contact[i].dist;
                    if (col_dist < -1e-3) {
                        cost += -col_dist;
                    }
                }
            }
            return cost;
        }

        double computeArcLength(const Spline &spline, int check_points) const {
            double total_length = 0.0;

#pragma omp parallel for reduction(+ : total_length)
            for (int i = 1; i < check_points; ++i) {
                double u1 = static_cast<double>(i - 1) / (check_points - 1);
                double u2 = static_cast<double>(i) / (check_points - 1);

                Eigen::VectorXd p1 = spline(u1);
                Eigen::VectorXd p2 = spline(u2);

                total_length += (p2 - p1).norm();
            }

            return total_length;
        }

        bool findBestPath(const std::vector<Spline> &successful_paths, Spline &best_spline, int check_points = 10) {
            double min_cost = std::numeric_limits<double>::infinity();
            bool found = false;

            //#pragma omp parallel for
            for (size_t i = 0; i < successful_paths.size(); ++i) {
                double cost = computeArcLength(successful_paths[i], check_points);
                //#pragma omp critical
                {
                    if (cost < min_cost) {
                        min_cost = cost;
                        best_spline = successful_paths[i];
                        found = true;
                    }
                }
            }

            return found;
        }

        bool plan(const Point &start, const Point &end, const Point &end_derivative, double sigma, const Point &limits,
                  int sample_count = 50,
                  int check_points = 50, int init_points = 10) {
            coll_pts.clear();
            initializePath(start, end, end_derivative, init_points); {
                for (int i = 0; i < sample_count; ++i) {
                    perturb();

                    // Lambda function for collision cost
                    auto collision_cost_lambda = [&](const Eigen::Vector3d& via_pt) {
                        return collision_cost(via_pt, check_points, data_);
                    };

                    auto via_pts = get_via_pts();
                    auto pt = via_pts[1];
                    Point step = Point::Ones()*1e-2;
                    Gradient grad(collision_cost_lambda, step, pt);
                    coll_gradient = grad.compute();


                    if (!checkCollision(path_spline_, check_points, data_)) {
                        std::cout << "ready!" << std::endl;
                    }
                }
            }

            return false;
        }
    };
} // namespace sspp

#endif // spp_SAMPLING_PATH_PLANNER_H
