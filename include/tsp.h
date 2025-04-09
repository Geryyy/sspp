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
#include <random>
#include "utility.h"
#include <optional>

#define PROFILE_TIME 0

namespace tsp {
    constexpr int kSplineDegree = 2;
    constexpr int kDOF = 4;
    using Point = Eigen::Matrix<double, kDOF, 1>;
    using Spline = Eigen::Spline<double, kDOF, kSplineDegree>;
    //    using GradientType = Gradient<kDOF>;
    using GradientStepType = GradientStep<kDOF>;
    using GradientDescentType = GradientDescent<kDOF>;

    struct CollisionPoint {
        double spline_param;
        double collision_distance;
        Point coll_point;

        CollisionPoint(double spline_param, double coll_dist, Point coll_point)
            : spline_param(spline_param), collision_distance(coll_dist), coll_point(std::move(coll_point)) {
        }
    };


    struct PathCandidate {
        Point via_point;
        std::vector<GradientStepType> gradient_steps;
        SolverStatus status;

        PathCandidate(Point via_point, std::vector<GradientStepType> gradient_steps,
                      SolverStatus status) : via_point(std::move(via_point)), gradient_steps(std::move(gradient_steps)),
                                             status(status) {
        }
    };

    /* TODO: add start_derivative_ */
    class TaskSpacePlanner {
    private:
        using SplineFitter = Eigen::SplineFitting<Spline>;

        Spline path_spline_;
        mjModel *model_;
        std::vector<mjData *> data_copies_;
        std::vector<mjData *> data_copies2_;
        char error_buffer_[1000]{}; // Buffer for storing MuJoCo error messages
        std::vector<double> param_vec;
        std::vector<Point> via_points;
        // std::vector<CollisionPoint> coll_pts;
        Point end_derivative_;

        // statistics
        std::vector<PathCandidate> successful_candidates_;
        std::vector<PathCandidate> failed_candidates_;
        std::vector<Point> sampled_via_pts_;
        bool flag_endderivatives = false;

    public:
        explicit TaskSpacePlanner(mjModel *model)
            : model_(model), via_points(), param_vec() {
            mjData *data_ = mj_makeData(model_);
            initializeDataCopies(data_);
            if (!data_) {
                throw std::runtime_error("Failed to create MuJoCo data structure.");
            }
        }

        explicit TaskSpacePlanner(const std::string &xml_string) : via_points(), param_vec() {
            // Parse the model from the XML string
            model_ = mj_loadXML(xml_string.c_str(), nullptr, error_buffer_, sizeof(error_buffer_));
            if (!model_) {
                throw std::runtime_error("Failed to load MuJoCo model from XML: " + std::string(error_buffer_));
            }

            // Create the mjData structure associated with the model
            mjData *data_ = mj_makeData(model_);
            if (!data_) {
                throw std::runtime_error("Failed to create MuJoCo data structure.");
            }

            initializeDataCopies(data_);
        }

        ~TaskSpacePlanner() {
            for (auto &data_copy: data_copies_) {
                mj_deleteData(data_copy);
            }

            for (auto &data_copy: data_copies2_) {
                mj_deleteData(data_copy);
            }
        }

        void reset() {
            successful_candidates_.clear();
            failed_candidates_.clear();
            sampled_via_pts_.clear();
        }


        Spline initializePath(const Point &start, const Point &end, const Point &end_derivative, int num_points = 3) {
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
                via_mat.block<kDOF, 1>(0, i) = via_points[i];
            }

            Eigen::MatrixXd derivatives = end_derivative;
            Eigen::Vector<int, 1> deriv_ind(num_points - 1);
            Spline spline = SplineFitter::InterpolateWithDerivatives(via_mat, derivatives, deriv_ind, kSplineDegree,
                                                                     u_knots);
            return spline;
        }


        Spline initializePath(const Point &start, const Point &end, int num_points = 3) {
            param_vec.clear();
            via_points.clear();

            // linear placement of via points from start to end
            for (int i = 0; i < num_points; ++i) {
                double t = static_cast<double>(i) / (num_points - 1);
                Point point = (1 - t) * start + t * end;
                param_vec.push_back(t);
                via_points.push_back(point);
            }
            Eigen::Map<Eigen::VectorXd> u_knots(param_vec.data(), num_points);
            Eigen::MatrixXd via_mat(kDOF, num_points);
            for (size_t i = 0; i < via_points.size(); i++) {
                via_mat.block<kDOF, 1>(0, i) = via_points[i];
            }

            Spline spline = SplineFitter::Interpolate(via_mat, kSplineDegree, u_knots);
            return spline;
        }

        Spline initializePath(const std::vector<Point> &via_pts) {
            param_vec.clear();
            via_points.clear();

            // linear placement of via points from start to end
            size_t num_points = via_pts.size();
            for (int i = 0; i < num_points; ++i) {
                double t = static_cast<double>(i) / (num_points - 1);
                param_vec.push_back(t);
                via_points.push_back(via_pts[i]);
            }
            Eigen::Map<Eigen::VectorXd> u_knots(param_vec.data(), num_points);
            Eigen::MatrixXd via_mat(kDOF, num_points);
            for (size_t i = 0; i < via_points.size(); i++) {
                via_mat.block<kDOF, 1>(0, i) = via_points[i];
            }

            Spline spline = SplineFitter::Interpolate(via_mat, kSplineDegree, u_knots);
            return spline;
        }


        Spline path_from_via_pt(const Point &via_pt) {
            Eigen::Map<Eigen::VectorXd> u_knots(param_vec.data(), param_vec.size());
            Eigen::Map<Eigen::MatrixXd> via_mat(reinterpret_cast<double *>(via_points.data()),
                                                kDOF, via_points.size());

            Eigen::MatrixXd via_mat_copy = via_mat;
            via_mat_copy.block<kDOF, 1>(0, 1) = via_pt;

            Eigen::MatrixXd derivatives = end_derivative_;
            Eigen::Vector<int, 1> deriv_ind(via_points.size() - 1);

            Spline spline;
            if(flag_endderivatives) {
                spline = SplineFitter::InterpolateWithDerivatives(via_mat_copy, derivatives, deriv_ind, kSplineDegree,
                                                                       u_knots);
            }
            else {
                spline = SplineFitter::Interpolate(via_mat_copy, kSplineDegree, u_knots);
            }
            return spline;
        }

        static Point get_random_point(const Point &mean, const Point &stddev) {
            static std::random_device rd;
            static std::mt19937 gen(rd());

            std::normal_distribution<double> dist_x(mean.x(), stddev.x());
            std::normal_distribution<double> dist_y(mean.y(), stddev.y());
            std::normal_distribution<double> dist_z(mean.z(), stddev.z());
            Point pt;
            pt << dist_x(gen), dist_y(gen), dist_z(gen), 0.0;
            return pt;
        }

        static Point evaluate(double u, const Spline &spline) {
            return spline(u);
        }

        [[nodiscard]] Point evaluate(double u) const {
            return path_spline_(u);
        }

        [[nodiscard]] std::vector<Point> get_via_pts() const {
            return via_points;
        }

        [[nodiscard]] Spline::ControlPointVectorType get_ctrl_pts() const {
            return path_spline_.ctrls();
        }

        [[nodiscard]] Spline::KnotVectorType get_knot_vector() const {
            return path_spline_.knots();
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

        bool check_collision_point(const Point &pt) {
            mjData *mj_data = this->data_copies_[0];
            Point point = pt;

            for (int j = 0; j < 3; ++j) {
                mj_data->qpos[j] = point(j);
            }
            mj_forward(model_, mj_data);

            for (int i = 0; i < mj_data->ncon; i++) {
                auto col_dist = mj_data->contact[i].dist;
                if (col_dist < -1e-3) {
                    return true;
                }
            }
            return false;
        }

        bool check_collision(const Spline &spline, int num_samples) {
            bool collision = false;
#pragma omp parallel
            {
#pragma omp for schedule(dynamic, 1)
                for (int i = 0; i <= num_samples; ++i) {
                    mjData *mj_data = data_copies_[omp_get_thread_num()];
                    double u = static_cast<double>(i) / num_samples;
                    Point point = spline(u);

                    for (int j = 0; j < kDOF; ++j) {
                        mj_data->qpos[j] = point(j);
                    }
                    mj_forward(model_, mj_data);

                    // iterate over all contacts
                    for (int i = 0; i < mj_data->ncon; i++) {
                        //                    std::cout << " Collision at sample " << i << " with depth " << data->contact[i].dist << std::endl;
                        auto col_dist = mj_data->contact[i].dist;
                        if (col_dist < -1e-3) {
                            auto nv = model_->nv;
                            auto nJ = mj_data->nJ;

                            if (nJ > 0) {
                                int nc = nJ / nv;
                                std::cout << "dimension of constrained jacobian nc: " << nc << " x nv: " << nv <<
                                        std::endl;
                                Eigen::Map<Eigen::MatrixXd> J_constr(mj_data->efc_J, nv, nc);
                                std::cout << "J_constr:\n"
                                        << J_constr.transpose() << std::endl;
                                std::cout << "ncon: " << mj_data->ncon << std::endl;
                            }

                            CollisionPoint col_pt(u, col_dist, point);
                            // coll_pts.push_back(col_pt);
                            std::cout << " Collision at sample " << i << " with depth " << mj_data->contact[i].dist <<
                                    std::endl;
                            collision = true;
                        }
                    }
                }
            }
            return collision;
        }

        /* TODO: make moveable object selectable --> adapt qpos range to update */

        double collision_cost(const Point &via_pt, int eval_cnt, mjData *mj_data, bool use_center_dist = true) {
            Spline spline = path_from_via_pt(via_pt);
            double cost = 0.0;

            // Use OpenMP to parallelize evaluation
#pragma omp parallel
            {
                //                mjData *mj_data = data_copies2_[omp_get_thread_num()]; // Each thread gets a separate mjData copy
                double thread_cost = 0.0; // Local cost for each thread
                //#pragma omp for nowait
                for (int i = 0; i <= eval_cnt; ++i) {
                    double u = static_cast<double>(i) / eval_cnt;
                    Point point = spline(u);

                    for (int j = 0; j < 3; ++j) {
                        mj_data->qpos[j] = point(j);
                    }
                    auto quat = yaw_to_quat(point[3], mj_data);
                    mj_data->qpos[3] = quat.w();
                    mj_data->qpos[4] = quat.x();
                    mj_data->qpos[5] = quat.y();
                    mj_data->qpos[6] = quat.z();

                    mj_forward(model_, mj_data);

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
                }
                // Accumulate the thread-local costs safely
#pragma omp atomic
                cost += thread_cost;
            }
            return cost;
        }

        double computeArcLength(const Spline &spline, int check_points) const {
            double total_length = 0.0;
            const double step = 1.0 / (check_points - 1); // Precompute step size

#pragma omp parallel for reduction(+ : total_length)
            for (int i = 1; i < check_points; ++i) {
                double u1 = (i - 1) * step;
                double u2 = i * step;

                const Eigen::VectorXd p1 = spline(u1); // Call once
                const Eigen::VectorXd p2 = spline(u2); // Call once

                total_length += (p2 - p1).norm();
            }

            return total_length;
        }

        bool findBestPath(const std::vector<PathCandidate> &candidates, Spline &best_spline, int check_points = 10) {
            double min_cost = std::numeric_limits<double>::infinity();
            bool found = false;
            Spline best_spline_local; // Local best spline (will be assigned globally later)

#pragma omp parallel
            {
                double thread_min_cost = std::numeric_limits<double>::infinity();
                Spline thread_best_spline;
                bool thread_found = false;

#pragma omp for nowait
                for (size_t i = 0; i < candidates.size(); ++i) {
                    auto path_candidate = path_from_via_pt(candidates[i].via_point);
                    double cost = computeArcLength(path_candidate, check_points);

                    if (cost < thread_min_cost) {
                        thread_min_cost = cost;
                        thread_best_spline = path_candidate;
                        thread_found = true;
                    }
                }

                // Merge thread-local results into global variables safely
#pragma omp critical
                {
                    if (thread_found && thread_min_cost < min_cost) {
                        min_cost = thread_min_cost;
                        best_spline_local = thread_best_spline;
                        found = true;
                    }
                }
            }

            if (found) {
                best_spline = best_spline_local;
            }

            return found;
        }


        void collision_optimization(const std::vector<Point> &via_point_candidates,
                                    std::vector<PathCandidate> &successful_candidates,
                                    std::vector<PathCandidate> &failed_candidates,
                                    int sample_count, int check_points, int gd_iterations) {
            // Thread-local storage
            std::vector<PathCandidate> successful_local;
            std::vector<PathCandidate> failed_local;

#pragma omp parallel
            {
                mjData *mj_data = data_copies_[omp_get_thread_num()]; // Each thread gets its own mjData
                std::vector<PathCandidate> successful_thread;
                std::vector<PathCandidate> failed_thread;

#pragma omp for schedule(dynamic, 1) nowait
                for (int i = 0; i < sample_count; ++i) {
                    const Point &via_candidate = via_point_candidates[i];

                    // Lambda function for collision cost
                    auto collision_cost_lambda = [&](const Point &via_pt) {
                        return collision_cost(via_pt, check_points, mj_data);
                    };

                    /* Gradient descent optimization */
                    Point diff_delta = Point::Ones() * 1e-2;
                    constexpr double step_size = 1e-3;
                    GradientDescentType graddesc(step_size, gd_iterations, collision_cost_lambda, diff_delta);
                    auto solver_status = graddesc.optimize(via_candidate);
                    const auto via_pt_opt = graddesc.get_result();

                    // Store results in thread-local vectors
                    PathCandidate candidate(via_pt_opt, graddesc.get_gradient_descent_steps(), solver_status);
                    if (solver_status == SolverStatus::Converged) {
                        successful_thread.push_back(candidate);
                    } else {
                        failed_thread.push_back(candidate);
                    }
                }

                // Merge thread-local results into global vectors safely
#pragma omp critical
                {
                    successful_candidates.insert(successful_candidates.end(), successful_thread.begin(),
                                                 successful_thread.end());
                    failed_candidates.insert(failed_candidates.end(), failed_thread.begin(), failed_thread.end());
                }
            }
        }


        void arclength_optimization(std::vector<PathCandidate> &path_candidates,
                                    std::vector<PathCandidate> &optimized_candidates,
                                    int check_points, int gd_iterations) {
#pragma omp parallel
            {
#pragma omp for schedule(dynamic, 1)
                for (const auto &candidate: path_candidates) {
                    mjData *mj_data = data_copies_[omp_get_thread_num()];
                    const Point via_candidate = candidate.via_point;
                    // Lambda function for collision cost
                    auto arc_lengthcost_lambda = [&](const Point &via_pt) {
                        auto spline = path_from_via_pt(via_pt);
                        double col_cost = collision_cost(via_pt, check_points, mj_data);
                        double len_cost = (computeArcLength(spline, check_points));
                        return len_cost; // + col_cost*col_cost;
                    };

                    /* gradient descent steps */
                    Point diff_delta = Point::Ones() * 1e-2;
                    constexpr double step_size = 1e-3;
                    GradientDescentType graddesc(step_size, gd_iterations, arc_lengthcost_lambda, diff_delta);
                    auto solver_status = graddesc.optimize(via_candidate);
                    const auto via_pt_opt = graddesc.get_result();

                    std::cout << "solver status: " << SolverStatustoString(solver_status) << std::endl;
#pragma omp critical
                    if (solver_status == SolverStatus::Converged) {
                        PathCandidate cand(via_pt_opt, graddesc.get_gradient_descent_steps(), solver_status);
                        optimized_candidates.push_back(cand);
                    } else {
                        PathCandidate cand(via_pt_opt, graddesc.get_gradient_descent_steps(), solver_status);
                        optimized_candidates.push_back(cand);
                    }
                }
            }
        }

        // Helper function containing the core planning logic
        std::vector<PathCandidate> core_plan(const std::vector<Point> &via_pts,
                                             const std::optional<Point> &end_derivative_opt,
                                             double sigma, const Point &limits,
                                             int sample_count,
                                             int check_points,
                                             int gd_iterations,
                                             int init_points) {
#ifdef PROFILE_TIME
            auto start_time = std::chrono::high_resolution_clock::now();
#endif
            // coll_pts.clear();
            /* initialize straight line */
#ifdef PROFILE_TIME
            auto init_start_time = std::chrono::high_resolution_clock::now();
#endif
            if (via_pts.size() < 2) {
                throw std::runtime_error("Not enough via points");
            }

            const Point start = via_pts.front();
            const Point end  = via_pts.back();
            Spline init_spline;
            if (via_pts.size() > 2){
                init_spline = initializePath(via_pts);
            }
            else {
                if (end_derivative_opt.has_value()) {
                    init_spline = initializePath(start, end, end_derivative_opt.value(), init_points);
                } else {
                    init_spline = initializePath(start, end, init_points);
                }
            }
            path_spline_ = init_spline;
            Point init_via_pt = via_points[1];

            // check if start or end are in collission
            std::vector<PathCandidate> no_candidates;
            if (check_collision_point(start)) {
                std::cerr << "start position is in collision!" << std::endl;
                return no_candidates;
            }

            if (check_collision_point(end)) {
                std::cerr << "end position is in collision!" << std::endl;
                return no_candidates;
            }

            /* create random ensemble */
            std::vector<Point> via_point_candidates;
            via_point_candidates.push_back(init_via_pt);

            for (int i = 0; i < sample_count - 1; i++) {
                Point stddev = Point::Ones() * sigma;
                Point noisy_via_pt = get_random_point(init_via_pt, stddev);
                via_point_candidates.push_back(noisy_via_pt);
                // std::cout << "random via point[" << i << "]: " << noisy_via_pt.transpose() << std::endl;
            }

            sampled_via_pts_ = via_point_candidates;

#ifdef PROFILE_TIME
            auto init_end_time = std::chrono::high_resolution_clock::now();
            auto init_duration = std::chrono::duration_cast<std::chrono::microseconds>(init_end_time - init_start_time);
            std::cerr << "plan.Initialize duration [ms]: " << init_duration.count() / 1e3 << std::endl;
#endif

            /* optimize candidates for collision */
#ifdef PROFILE_TIME
            auto opt_start_time = std::chrono::high_resolution_clock::now();
#endif
            collision_optimization(via_point_candidates, successful_candidates_, failed_candidates_,
                                   sample_count, check_points, gd_iterations);

#ifdef PROFILE_TIME
            auto opt_end_time = std::chrono::high_resolution_clock::now();
            auto opt_duration = std::chrono::duration_cast<std::chrono::microseconds>(opt_end_time - opt_start_time);
            std::cerr << "Collision Optimization duration [ms]: " << opt_duration.count() / 1e3 << std::endl;
#endif


            /* tighten succesful paths */
            // TODO: needs fix -> moves below ground
            // std::vector<PathCandidate> opt_candidates;
            // arclength_optimization(successful_candidates_, opt_candidates, data_, check_points, gd_iterations);
            //
            // // test!!
            // failed_candidates_ = opt_candidates;

            /* find best path */
#ifdef PROFILE_TIME
            auto find_best_start_time = std::chrono::high_resolution_clock::now();
#endif
            findBestPath(successful_candidates_, path_spline_, check_points);
#ifdef PROFILE_TIME
            auto find_best_end_time = std::chrono::high_resolution_clock::now();
            auto find_best_duration = std::chrono::duration_cast<std::chrono::microseconds>(
                find_best_end_time - find_best_start_time);
            std::cerr << "findBestPath duration [ms]: " << find_best_duration.count() / 1e3 << std::endl;

            auto end_time = std::chrono::high_resolution_clock::now();
            auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            std::cerr << "Total plan duration [ms]: " << total_duration.count() / 1e3 << std::endl;
#endif

            return successful_candidates_;
        }

    public:
        std::vector<PathCandidate> plan_with_end_derivatives(const Point &start,
                                                            const Point &end, const Point &end_derivative, double sigma,
                                                            const Point &limits,
                                                            const int sample_count = 50,
                                                            const int check_points = 50,
                                                            const int gd_iterations = 10,
                                                            const int init_points = 3) {
            flag_endderivatives = true;
            std::vector<Point> via_pts = {start, end};
            return core_plan(via_pts, end_derivative, sigma, limits, sample_count, check_points, gd_iterations,
                             init_points);
        }

        std::vector<PathCandidate> plan_with_via_pts(const std::vector<Point> via_pts,
                                        double sigma, const Point &limits,
                                        const int sample_count = 50,
                                        const int check_points = 50,
                                        const int gd_iterations = 10,
                                        const int init_points = 3) {
            flag_endderivatives = false;
            return core_plan(via_pts, std::nullopt, sigma, limits, sample_count, check_points, gd_iterations,
                             init_points);
        }

        std::vector<PathCandidate> plan(const Point &start,
                                        const Point &end, double sigma, const Point &limits,
                                        const int sample_count = 50,
                                        const int check_points = 50,
                                        const int gd_iterations = 10,
                                        const int init_points = 3) {
            flag_endderivatives = false;
            std::vector<Point> via_pts = {start, end};
            return core_plan(via_pts, std::nullopt, sigma, limits, sample_count, check_points, gd_iterations,
                             init_points);
        }

        std::vector<PathCandidate> get_succesful_path_candidates() {
            return successful_candidates_;
        }

        std::vector<PathCandidate> get_failed_path_candidates() {
            return failed_candidates_;
        }

        std::vector<Point> get_sampled_via_pts() {
            return sampled_via_pts_;
        }

        static std::vector<Point> get_path_pts(const Spline &spline, const int pts_cnt = 10) {
            std::vector<Point> pts;
            for (int i = 0; i < pts_cnt; i++) {
                double u = static_cast<double>(i) / (pts_cnt - 1);
                auto pt = evaluate(u, spline);
                pts.push_back(pt);
                //        std::cout << "pt("<<u<<") " << pt.transpose() << std::endl;
            }
            return pts;
        }

        [[nodiscard]] std::vector<Point> get_path_pts(const int pts_cnt = 10) const {
            std::vector<Point> pts;
            for (int i = 0; i < pts_cnt; i++) {
                double u = static_cast<double>(i) / (pts_cnt - 1);
                auto pt = evaluate(u);
                pts.push_back(pt);
                //        std::cout << "pt("<<u<<") " << pt.transpose() << std::endl;
            }
            return pts;
        }

    private:
        void initializeDataCopies(const mjData *data) {
            Timer exectimer;
            exectimer.tic();
            int max_threads = omp_get_max_threads();
            data_copies_.resize(max_threads, nullptr);

            for (auto &data_copy: data_copies_) {
                data_copy = mj_copyData(nullptr, model_, data);
            }
            std::cout << "data copies duration: " << exectimer.toc() / 1e3 << " us" << std::endl;

            data_copies2_.resize(max_threads, nullptr);

            for (auto &data_copy: data_copies2_) {
                data_copy = mj_copyData(nullptr, model_, data);
            }
        }
    };
} // namespace sspp

#endif // TASK_SPACE_PLANNER_H
