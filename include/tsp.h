//
// Created by geraldebmer on 26.02.25
//

#ifndef TASK_SPACE_PLANNER_H
#define TASK_SPACE_PLANNER_H

#include "Gradient.h"
#include <utility>
#include <vector>
#include <Eigen/Core>
#include <unsupported/Eigen/Splines>
#include <iostream>
#include <random>
#include <omp.h>
#include "mujoco/mujoco.h"
#include "Timer.h"
#include "utility.h"
#include <optional>
#include "Collision.h"
#include <memory>

#define PROFILE_TIME 0

namespace tsp {
    constexpr int kSplineDegree = 2;
    constexpr int kDOF = 4;
    using Point = Eigen::Matrix<double, kDOF, 1>;
    using Spline = Eigen::Spline<double, kDOF, kSplineDegree>;
    using GradientStepType = GradientStep<kDOF>;
    using GradientDescentType = GradientDescent<kDOF>;

    struct CollisionPoint {
        double spline_param, collision_distance;
        Point coll_point;
        CollisionPoint(double sp, double cd, Point cp) : spline_param(sp), collision_distance(cd), coll_point(std::move(cp)) {}
    };

    struct PathCandidate {
        Point via_point;
        std::vector<GradientStepType> gradient_steps;
        SolverStatus status;
        PathCandidate(Point vp, std::vector<GradientStepType> gs, SolverStatus s)
                : via_point(std::move(vp)), gradient_steps(std::move(gs)), status(s) {}
    };

    class TaskSpacePlanner {
    private:
        using SplineFitter = Eigen::SplineFitting<Spline>;

        Spline path_spline_;
        mjModel *model_;
        char error_buffer_[1000]{};
        std::vector<double> param_vec;
        std::vector<Point> via_points;
        Point end_derivative_;
        std::vector<std::shared_ptr<Collision<Point>>> collision_env_vec;

        // Statistics
        std::vector<PathCandidate> successful_candidates_, failed_candidates_;
        std::vector<Point> sampled_via_pts_;
        bool flag_endderivatives = false;

        // Sampling parameters
        Point mean_, stddev_;
        double stddev_min = 0.01, stddev_max = 1.0;

    public:
        explicit TaskSpacePlanner(mjModel *model, std::string body_name) : model_(model) {
            init_collision_env(std::move(body_name));
        }

        explicit TaskSpacePlanner(const std::string &xml_string, std::string body_name) {
            model_ = mj_loadXML(xml_string.c_str(), nullptr, error_buffer_, sizeof(error_buffer_));
            if (!model_) throw std::runtime_error("Failed to load MuJoCo model from XML: " + std::string(error_buffer_));
            init_collision_env(std::move(body_name));
        }

        ~TaskSpacePlanner() = default;

        void init_collision_env(std::string body_name) {
            collision_env_vec.reserve(omp_get_max_threads());
            for(int i = 0; i < omp_get_max_threads(); i++) {
                collision_env_vec.push_back(std::make_shared<Collision<Point>>(body_name, model_));
            }
        }

        void reset() {
            successful_candidates_.clear();
            failed_candidates_.clear();
            sampled_via_pts_.clear();
        }

        Spline initializePath(const Point &start, const Point &end, const Point &end_derivative, int num_points = 3) {
            setupLinearPath(start, end, num_points);
            end_derivative_ = end_derivative;

            auto [via_mat, u_knots] = createSplineData();
            Eigen::MatrixXd derivatives = end_derivative;
            Eigen::Vector<int, 1> deriv_ind(num_points - 1);

            return SplineFitter::InterpolateWithDerivatives(via_mat, derivatives, deriv_ind, kSplineDegree, u_knots);
        }

        Spline initializePath(const Point &start, const Point &end, int num_points = 3) {
            setupLinearPath(start, end, num_points);
            auto [via_mat, u_knots] = createSplineData();
            return SplineFitter::Interpolate(via_mat, kSplineDegree, u_knots);
        }

        Spline path_from_via_pt(const Point &via_pt) {
            auto [via_mat, u_knots] = createSplineData();
            via_mat.block<kDOF, 1>(0, 1) = via_pt;

            if(flag_endderivatives) {
                Eigen::MatrixXd derivatives = end_derivative_;
                Eigen::Vector<int, 1> deriv_ind(via_points.size() - 1);
                return SplineFitter::InterpolateWithDerivatives(via_mat, derivatives, deriv_ind, kSplineDegree, u_knots);
            }
            return SplineFitter::Interpolate(via_mat, kSplineDegree, u_knots);
        }

        static Point get_random_point(const Point &mean, const Point &stddev) {
            static thread_local std::mt19937 gen(std::random_device{}());
            Point pt;
            for(int i = 0; i < 3; ++i) {
                std::normal_distribution<double> dist(mean[i], stddev[i]);
                pt[i] = dist(gen);
            }
            pt[3] = 0.0;
            return pt;
        }

        static Point evaluate(double u, const Spline &spline) { return spline(u); }
        [[nodiscard]] Point evaluate(double u) const { return path_spline_(u); }
        [[nodiscard]] std::vector<Point> get_via_pts() const { return via_points; }
        [[nodiscard]] Spline::ControlPointVectorType get_ctrl_pts() const { return path_spline_.ctrls(); }
        [[nodiscard]] Spline::KnotVectorType get_knot_vector() const { return path_spline_.knots(); }

        double collision_cost(const Point &via_pt, int eval_cnt, bool use_center_dist = true) {
            Spline spline = path_from_via_pt(via_pt);
            double cost = 0.0;

            for (int i = 0; i <= eval_cnt; ++i) {
                double u = static_cast<double>(i) / eval_cnt;
                cost += collision_env_vec[omp_get_thread_num()]->collision_point_cost(spline(u), use_center_dist);
            }
            return cost;
        }

        void collision_optimization(const std::vector<Point> &via_point_candidates,
                                    std::vector<PathCandidate> &successful_candidates,
                                    std::vector<PathCandidate> &failed_candidates,
                                    int sample_count, int check_points, int gd_iterations) {
#pragma omp parallel default(none) shared(via_point_candidates, check_points, gd_iterations, successful_candidates, failed_candidates, sample_count)
            {
                std::vector<PathCandidate> successful_thread, failed_thread;

#pragma omp for schedule(dynamic, 1) nowait
                for (int i = 0; i < sample_count; ++i) {
                    auto collision_cost_lambda = [&](const Point &via_pt) {
                        return collision_cost(via_pt, check_points);
                    };

                    GradientDescentType graddesc(1e-3, gd_iterations, collision_cost_lambda, Point::Ones() * 1e-2);
                    auto solver_status = graddesc.optimize(via_point_candidates[i]);

                    PathCandidate candidate(graddesc.get_result(), graddesc.get_gradient_descent_steps(), solver_status);
                    (solver_status == SolverStatus::Converged ? successful_thread : failed_thread).push_back(candidate);
                }

#pragma omp critical
                {
                    successful_candidates.insert(successful_candidates.end(), successful_thread.begin(), successful_thread.end());
                    failed_candidates.insert(failed_candidates.end(), failed_thread.begin(), failed_thread.end());
                }
            }
        }

        double computeArcLength(const Spline &spline, int check_points) const {
            double total_length = 0.0;
            const double step = 1.0 / (check_points - 1);

            for (int i = 1; i < check_points; ++i) {
                const Point p1 = spline((i - 1) * step);
                const Point p2 = spline(i * step);
                total_length += (p2 - p1).norm();
            }
            return total_length;
        }

        bool findBestPath(const std::vector<PathCandidate> &candidates, Spline &best_spline, int check_points = 10) {
            if (candidates.empty()) return false;

            double min_cost = std::numeric_limits<double>::infinity();
            size_t best_idx = 0;

            for (size_t i = 0; i < candidates.size(); ++i) {
                double cost = computeArcLength(path_from_via_pt(candidates[i].via_point), check_points);
                if (cost < min_cost) {
                    min_cost = cost;
                    best_idx = i;
                }
            }

            best_spline = path_from_via_pt(candidates[best_idx].via_point);
            return true;
        }

        std::vector<PathCandidate> plan(const Point &start, const Point &end, double sigma, const Point &limits,
                                        int sample_count = 50, int check_points = 50, int gd_iterations = 10,
                                        int init_points = 3, double z_min = 0.0) {
#if PROFILE_TIME
            auto start_time = std::chrono::high_resolution_clock::now();
#endif

            path_spline_ = initializePath(start, end, init_points);
            Point init_via_pt = start + 0.5 * (end - start);
            init_via_pt[2] = std::max(init_via_pt[2], z_min);

            // Check for collision at endpoints
            if (collision_env_vec[0]->check_collision_point(start) || collision_env_vec[0]->check_collision_point(end)) {
                std::cerr << "Start or end position is in collision!" << std::endl;
                return {};
            }

            // Generate candidate via points
            std::vector<Point> via_point_candidates = {init_via_pt};
            Point stddev = Point::Ones() * sigma;

            for (int i = 1; i < sample_count; i++) {
                via_point_candidates.push_back(get_random_point(init_via_pt, stddev));
            }
            sampled_via_pts_ = via_point_candidates;

            // Optimize and find best path
            collision_optimization(via_point_candidates, successful_candidates_, failed_candidates_,
                                   sample_count, check_points, gd_iterations);
            findBestPath(successful_candidates_, path_spline_, check_points);

#if PROFILE_TIME
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            std::cerr << "Total plan duration [ms]: " << duration.count() / 1e3 << std::endl;
#endif

            return successful_candidates_;
        }

        // Getters
        std::vector<PathCandidate> get_succesful_path_candidates() { return successful_candidates_; }
        std::vector<PathCandidate> get_failed_path_candidates() { return failed_candidates_; }
        std::vector<Point> get_sampled_via_pts() { return sampled_via_pts_; }

        static std::vector<Point> get_path_pts(const Spline &spline, int pts_cnt = 10) {
            std::vector<Point> pts;
            pts.reserve(pts_cnt);
            for (int i = 0; i < pts_cnt; i++) {
                pts.push_back(evaluate(static_cast<double>(i) / (pts_cnt - 1), spline));
            }
            return pts;
        }

        [[nodiscard]] std::vector<Point> get_path_pts(int pts_cnt = 10) const {
            std::vector<Point> pts;
            pts.reserve(pts_cnt);
            for (int i = 0; i < pts_cnt; i++) {
                pts.push_back(evaluate(static_cast<double>(i) / (pts_cnt - 1)));
            }
            return pts;
        }

    private:
        void setupLinearPath(const Point &start, const Point &end, int num_points) {
            param_vec.clear();
            via_points.clear();
            param_vec.reserve(num_points);
            via_points.reserve(num_points);

            for (int i = 0; i < num_points; ++i) {
                double t = static_cast<double>(i) / (num_points - 1);
                param_vec.push_back(t);
                via_points.push_back((1 - t) * start + t * end);
            }
        }

        std::pair<Eigen::MatrixXd, Eigen::Map<Eigen::VectorXd>> createSplineData() {
            Eigen::MatrixXd via_mat(kDOF, via_points.size());
            for (size_t i = 0; i < via_points.size(); i++) {
                via_mat.block<kDOF, 1>(0, i) = via_points[i];
            }
            return {via_mat, Eigen::Map<Eigen::VectorXd>(param_vec.data(), param_vec.size())};
        }
    };
} // namespace tsp

#endif // TASK_SPACE_PLANNER_H