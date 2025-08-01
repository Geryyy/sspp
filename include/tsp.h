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
#include <algorithm>
#include <omp.h>
#include "mujoco/mujoco.h"
#include "Timer.h"
#include "utility.h"
#include <optional>
#include "Collision.h"
#include <memory>
#include <cmath>

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

        // Algorithm parameters (set in constructor)
        double stddev_initial_, stddev_min_, stddev_max_;
        double stddev_increase_factor_, stddev_decay_factor_;
        double elite_fraction_;
        int sample_count_, check_points_, gd_iterations_, init_points_;
        double collision_weight_;
        double z_min_;
        Point limits_;  // Sampling bounds for each dimension
        bool enable_gradient_descent_;

        // Algorithm state (updated each plan call)
        Point mean_, stddev_;
        std::vector<PathCandidate> successful_candidates_, failed_candidates_;
        std::vector<Point> sampled_via_pts_;
        bool flag_endderivatives = false;

    public:
        explicit TaskSpacePlanner(mjModel *model, std::string body_name,
                                  double stddev_initial = 0.3,
                                  double stddev_min = 0.01,
                                  double stddev_max = 2.0,
                                  double stddev_increase_factor = 1.5,
                                  double stddev_decay_factor = 0.95,
                                  double elite_fraction = 0.3,
                                  int sample_count = 50,
                                  int check_points = 50,
                                  int gd_iterations = 10,
                                  int init_points = 3,
                                  double collision_weight = 1.0,
                                  double z_min = 0.0,
                                  Point limits = Point::Ones() * 2.0,
                                  bool enable_gradient_descent = true)
                : model_(model),
                  stddev_initial_(stddev_initial), stddev_min_(stddev_min), stddev_max_(stddev_max),
                  stddev_increase_factor_(stddev_increase_factor), stddev_decay_factor_(stddev_decay_factor),
                  elite_fraction_(elite_fraction), sample_count_(sample_count), check_points_(check_points),
                  gd_iterations_(gd_iterations), init_points_(init_points),
                  collision_weight_(collision_weight), z_min_(z_min), limits_(limits),
                  enable_gradient_descent_(enable_gradient_descent) {
            init_collision_env(std::move(body_name));
        }

        explicit TaskSpacePlanner(const std::string &xml_string, std::string body_name,
                                  double stddev_initial = 0.3,
                                  double stddev_min = 0.01,
                                  double stddev_max = 2.0,
                                  double stddev_increase_factor = 1.5,
                                  double stddev_decay_factor = 0.95,
                                  double elite_fraction = 0.3,
                                  int sample_count = 50,
                                  int check_points = 50,
                                  int gd_iterations = 10,
                                  int init_points = 3,
                                  double collision_weight = 1.0,
                                  double z_min = 0.0,
                                  Point limits = Point::Ones() * 2.0,
                                  bool enable_gradient_descent = true)
                : stddev_initial_(stddev_initial), stddev_min_(stddev_min), stddev_max_(stddev_max),
                  stddev_increase_factor_(stddev_increase_factor), stddev_decay_factor_(stddev_decay_factor),
                  elite_fraction_(elite_fraction), sample_count_(sample_count), check_points_(check_points),
                  gd_iterations_(gd_iterations), init_points_(init_points),
                  collision_weight_(collision_weight), z_min_(z_min), limits_(limits),
                  enable_gradient_descent_(enable_gradient_descent) {
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

        std::vector<PathCandidate> plan(const Point &start, const Point &end, bool iterate_flag = false) {
#if PROFILE_TIME
            auto start_time = std::chrono::high_resolution_clock::now();
#endif

            // Initialize distribution
            if (!iterate_flag) {
                mean_ = 0.5 * (start + end);  // midpoint for new paths
                mean_[2] = std::max(mean_[2], z_min_);  // ensure above ground

                // Ensure initial mean is within limits
                for (int i = 0; i < 3; ++i) {
                    mean_[i] = std::clamp(mean_[i], -limits_[i], limits_[i]);
                }

                stddev_ = Point::Ones() * stddev_initial_;
            }

            // Check for collision at endpoints
            if (collision_env_vec[0]->check_collision_point(start) || collision_env_vec[0]->check_collision_point(end)) {
                std::cerr << "Start or end position is in collision!" << std::endl;
                return {};
            }

            // Initialize path spline
            path_spline_ = initializePath(start, end, init_points_);

            // Generate candidate via points using current distribution
            std::vector<Point> via_point_candidates;
            via_point_candidates.reserve(sample_count_ + 1);

            // Always include the mean as a candidate (shortest path if collision-free)
            Point mean_candidate = mean_;
            mean_candidate[2] = std::max(mean_candidate[2], z_min_);
            via_point_candidates.push_back(mean_candidate);

            for (int i = 0; i < sample_count_; i++) {
                Point candidate = get_random_point(mean_, stddev_, limits_);
                candidate[2] = std::max(candidate[2], z_min_);  // ensure above ground
                via_point_candidates.push_back(candidate);
            }
            sampled_via_pts_ = via_point_candidates;

            // Optimize candidates for collision avoidance
            collision_optimization(via_point_candidates, successful_candidates_, failed_candidates_,
                                   sample_count_, check_points_, gd_iterations_);

            // Update distribution based on results
            if (!successful_candidates_.empty()) {
                updateDistributionFromElites();
                findBestPath(successful_candidates_, path_spline_, check_points_);
            } else {
                adaptStddev(false);  // increase stddev for exploration
            }

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
        Point get_current_mean() const { return mean_; }
        Point get_current_stddev() const { return stddev_; }
        Point get_limits() const { return limits_; }

        static Point evaluate(double u, const Spline &spline) { return spline(u); }
        [[nodiscard]] Point evaluate(double u) const { return path_spline_(u); }
        [[nodiscard]] std::vector<Point> get_via_pts() const { return via_points; }
        [[nodiscard]] Spline::ControlPointVectorType get_ctrl_pts() const { return path_spline_.ctrls(); }
        [[nodiscard]] Spline::KnotVectorType get_knot_vector() const { return path_spline_.knots(); }

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

        Spline initializePath(const Point &start, const Point &end, int num_points = 3) {
            setupLinearPath(start, end, num_points);
            auto [via_mat, u_knots] = createSplineData();
            return SplineFitter::Interpolate(via_mat, kSplineDegree, u_knots);
        }

        Spline path_from_via_pt(const Point &via_pt) {
            auto [via_mat, u_knots] = createSplineData();
            via_mat.block<kDOF, 1>(0, 1) = via_pt;
            return SplineFitter::Interpolate(via_mat, kSplineDegree, u_knots);
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

        static Point get_random_point(const Point &mean, const Point &stddev, const Point &limits) {
            static thread_local std::mt19937 gen(std::random_device{}());
            Point pt;

            for(int i = 0; i < 3; ++i) {
                // Truncated normal distribution within bounds [-limits[i], +limits[i]]
                std::normal_distribution<double> dist(mean[i], stddev[i]);
                double lower_bound = -limits[i];
                double upper_bound = limits[i];

                double sample;
                int attempts = 0;
                const int max_attempts = 100;  // Prevent infinite loop

                do {
                    sample = dist(gen);
                    attempts++;

                    // If too many attempts, fall back to uniform sampling within bounds
                    if (attempts >= max_attempts) {
                        std::uniform_real_distribution<double> uniform_dist(lower_bound, upper_bound);
                        sample = uniform_dist(gen);
                        break;
                    }
                } while (sample < lower_bound || sample > upper_bound);

                pt[i] = sample;
            }
            pt[3] = 0.0;  // Fourth dimension always zero
            return pt;
        }

        double computePathCost(const PathCandidate& candidate) {
            auto spline = path_from_via_pt(candidate.via_point);
            double arc_length = computeArcLength(spline, check_points_);
            double collision_cost_val = collision_cost(candidate.via_point, check_points_);
            return arc_length + collision_weight_ * collision_cost_val;
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

            successful_candidates.clear();
            failed_candidates.clear();

#ifdef DEBUG_SINGLE_THREAD
            // Single-threaded debug version
            for (int i = 0; i < sample_count; ++i) {
                if (enable_gradient_descent_ && gd_iterations > 0) {
                    auto collision_cost_lambda = [&](const Point &via_pt) {
                        return collision_cost(via_pt, check_points);
                    };

                    GradientDescentType graddesc(1e-3, gd_iterations, collision_cost_lambda, Point::Ones() * 1e-2);
                    auto solver_status = graddesc.optimize(via_point_candidates[i]);

                    PathCandidate candidate(graddesc.get_result(), graddesc.get_gradient_descent_steps(), solver_status);
                    (solver_status == SolverStatus::Converged ? successful_candidates : failed_candidates).push_back(candidate);
                } else {
                    Point candidate_via_pt = via_point_candidates[i];
                    bool is_collision_free = (collision_cost(candidate_via_pt, check_points) == 0.0);

                    PathCandidate candidate(candidate_via_pt, {},
                                          is_collision_free ? SolverStatus::Converged : SolverStatus::Failed);

                    (is_collision_free ? successful_candidates : failed_candidates).push_back(candidate);
                }
            }
#else
            // Original OpenMP version
#pragma omp parallel default(none) shared(via_point_candidates, check_points, gd_iterations, successful_candidates, failed_candidates, sample_count)
            {
                std::vector<PathCandidate> successful_thread, failed_thread;

#pragma omp for schedule(dynamic, 1) nowait
                for (int i = 0; i < sample_count; ++i) {
                    if (enable_gradient_descent_ && gd_iterations > 0) {
                        // With gradient descent refinement
                        auto collision_cost_lambda = [&](const Point &via_pt) {
                            return collision_cost(via_pt, check_points);
                        };

                        GradientDescentType graddesc(1e-3, gd_iterations, collision_cost_lambda, Point::Ones() * 1e-2);
                        auto solver_status = graddesc.optimize(via_point_candidates[i]);

                        PathCandidate candidate(graddesc.get_result(), graddesc.get_gradient_descent_steps(), solver_status);
                        (solver_status == SolverStatus::Converged ? successful_thread : failed_thread).push_back(candidate);
                    } else {
                        // Pure sampling - no gradient descent refinement
                        Point candidate_via_pt = via_point_candidates[i];

                        // Check if via point leads to collision-free path
                        bool is_collision_free = (collision_cost(candidate_via_pt, check_points) == 0.0);

                        // Create candidate with original via point and empty gradient steps
                        PathCandidate candidate(candidate_via_pt, {},
                                                is_collision_free ? SolverStatus::Converged : SolverStatus::Failed);

                        (is_collision_free ? successful_thread : failed_thread).push_back(candidate);
                    }
                }

#pragma omp critical
                {
                    successful_candidates.insert(successful_candidates.end(), successful_thread.begin(), successful_thread.end());
                    failed_candidates.insert(failed_candidates.end(), failed_thread.begin(), failed_thread.end());
                }
            }
#endif
        }

        void updateDistributionFromElites() {
            if (successful_candidates_.empty()) return;

            // Sort by path quality (lower cost = better)
            std::sort(successful_candidates_.begin(), successful_candidates_.end(),
                      [this](const PathCandidate& a, const PathCandidate& b) {
                          return computePathCost(a) < computePathCost(b);
                      });

            // Select top elites
            int num_elites = std::max(1, static_cast<int>(successful_candidates_.size() * elite_fraction_));

            // Compute log-based weights (CMA-ES style)
            std::vector<double> weights(num_elites);
            double weight_sum = 0.0;
            for (int i = 0; i < num_elites; ++i) {
                weights[i] = std::log(num_elites + 0.5) - std::log(i + 1.0);
                weight_sum += weights[i];
            }

            // Normalize weights
            for (auto& w : weights) w /= weight_sum;

            // Update mean (weighted average of elite via points)
            Point new_mean = Point::Zero();
            for (int i = 0; i < num_elites; ++i) {
                new_mean += weights[i] * successful_candidates_[i].via_point;
            }
            mean_ = new_mean;
            mean_[2] = std::max(mean_[2], z_min_);  // ensure above ground

            // Ensure mean stays within limits
            for (int i = 0; i < 3; ++i) {
                mean_[i] = std::clamp(mean_[i], -limits_[i], limits_[i]);
            }

            // Update stddev (weighted standard deviation)
            Point variance = Point::Zero();
            for (int i = 0; i < num_elites; ++i) {
                Point diff = successful_candidates_[i].via_point - mean_;
                variance += weights[i] * diff.cwiseProduct(diff);
            }
            stddev_ = variance.cwiseSqrt();

            // Apply bounds and adaptive decay
            adaptStddev(true);
        }

        void adaptStddev(bool successful) {
            if (successful) {
                stddev_ *= stddev_decay_factor_;  // decay for exploitation
            } else {
                stddev_ *= stddev_increase_factor_;  // increase for exploration
            }

            // Apply bounds
            stddev_ = stddev_.cwiseMax(Point::Ones() * stddev_min_)
                    .cwiseMin(Point::Ones() * stddev_max_);
        }

        bool findBestPath(const std::vector<PathCandidate> &candidates, Spline &best_spline, int check_points = 10) {
            if (candidates.empty()) return false;

            double min_cost = std::numeric_limits<double>::infinity();
            size_t best_idx = 0;

            for (size_t i = 0; i < candidates.size(); ++i) {
                double cost = computePathCost(candidates[i]);
                if (cost < min_cost) {
                    min_cost = cost;
                    best_idx = i;
                }
            }

            best_spline = path_from_via_pt(candidates[best_idx].via_point);
            return true;
        }
    };
} // namespace tsp

#endif // TASK_SPACE_PLANNER_H