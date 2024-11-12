//
// Created by geraldebmer on 11.11.24.
//

#ifndef SSPP_SSPP_H
#define SSPP_SSPP_H

#include <vector>
#include <Eigen/Core>
#include <unsupported/Eigen/Splines>
#include <iostream>
#include <random>
#include <omp.h>  // OpenMP header
#include "mujoco/mujoco.h"

#include "Timer.h"

namespace sspp {

#include <omp.h>  // For OpenMP

    template<int DOF>
    class SSPP {
    public:
        using Point = Eigen::Matrix<double, DOF, 1>;
        static const int spline_deg = 3;
        typedef Eigen::Spline<double, DOF, spline_deg> Spline_t;

    private:
        typedef Eigen::SplineFitting<Spline_t> SplineFitting_t;

        // Spline_t init_spline;
        Spline_t path_spline;
        mjModel* m;
        std::vector<mjData*> d_copies;

    public:
        SSPP(mjModel* m, mjData* d) : m(m) {
            int max_threads = omp_get_max_threads();
            d_copies.resize(max_threads, nullptr);
            for (auto& d_copy : d_copies) {
                d_copy = mj_copyData(nullptr, m, d);  // Initialize data copies once in the constructor
            }
        }

        ~SSPP() {
            for (auto& d_copy : d_copies) {
                mj_deleteData(d_copy);  // Cleanup data copies in the destructor
            }
        }

        bool initialize(Point start, Point end, Spline_t& init_spline, int num_pts=10) {
            Eigen::VectorXd u_knots(num_pts, 1);
            Eigen::MatrixXd via_points(DOF, num_pts);

            for (int i = 0; i < num_pts; ++i) {
                double t = static_cast<double>(i) / (num_pts - 1);
                Point point = (1 - t) * start + t * end;  // Linear interpolation
                via_points.col(i) = point;
                u_knots(i) = t;
            }

            init_spline = SplineFitting_t::Interpolate(via_points, spline_deg, u_knots);
            return true;
        }

        Point evaluate(double u) {
            return path_spline(u);
        }

        Point evaluate(Spline_t spline, double u) {
            return spline(u);
        }

        Spline_t sample(Spline_t init_spline, double sigma, Point limits) {
            auto ctrl_pts = init_spline.ctrls();
            int p = spline_deg;

            std::default_random_engine generator;
            std::normal_distribution<double> distribution(0.0, sigma);

            // Add noise to the control points
            for (int i = 0; i < ctrl_pts.rows(); ++i) {
                for (int j = p; j < ctrl_pts.cols() - p; ++j) {
                    double noise = distribution(generator);
                    ctrl_pts(i, j) += noise * limits(i);
                }
            }

            Spline_t sampled_spline(init_spline.knots(), ctrl_pts);
            return sampled_spline;
        }

        Spline_t sample_with_generator(Spline_t init_spline, double sigma, const Point& limits,
                                       std::default_random_engine& generator,
                                       std::normal_distribution<double>& distribution) {
            auto ctrl_pts = init_spline.ctrls();
            int p = spline_deg;

            // Adjust control points with noise
            for (int i = 0; i < ctrl_pts.rows(); ++i) {
                for (int j = p; j < ctrl_pts.cols() - p; ++j) {
                    double noise = distribution(generator);
                    ctrl_pts(i, j) += noise * limits(i);
                }
            }

            // Update sampled_spline with the modified control points
            Spline_t sampled_spline(init_spline.knots(), ctrl_pts);
            return sampled_spline;
        }


        bool check_collision(Spline_t spline, const int num_samples, mjData* d) {
            for (int i = 0; i <= num_samples; ++i) {
                double u = static_cast<double>(i) / num_samples;
                Point point = spline(u);

                for (int j = 0; j < DOF; ++j) {
                    d->qpos[j] = point(j);
                }
//                mj_collision(m, d);
                mj_forward(m, d);
                if (d->ncon > 0) {
                    return true;
                }
            }
            return false;
        }

        // Define the Metric Function (Arc Length) with OpenMP parallelization
        double metric(const Spline_t& spline, int check_pts) {
            double total_length = 0.0;

            // Discretize the spline and sum the distances between consecutive points
#pragma omp parallel for reduction(+:total_length)
            for (int i = 1; i < check_pts; ++i) {
                // Evaluate the spline at two points: u_i and u_(i+1)
                double u1 = static_cast<double>(i - 1) / (check_pts - 1);
                double u2 = static_cast<double>(i) / (check_pts - 1);

                Eigen::VectorXd p1 = spline(u1);
                Eigen::VectorXd p2 = spline(u2);

                // Compute the Euclidean distance between the points
                double dist = (p2 - p1).norm();
                total_length += dist;
            }

            return total_length;  // The arc length is the total distance
        }

        // Define the eval_cost function to find the spline with the lowest cost using OpenMP
        bool eval_cost(const std::vector<Spline_t>& successful_paths, Spline_t& result_spline, int check_pts=10) {
            double min_cost = std::numeric_limits<double>::infinity();
//            Spline_t best_spline;
            bool success = false;

            // Use parallel for to compute the cost (arc length) for each spline concurrently
#pragma omp parallel for
            for (size_t i = 0; i < successful_paths.size(); ++i) {
                double cost = metric(successful_paths[i], check_pts);

                // Critical section to update the best spline safely
#pragma omp critical
                {
                    if (cost < min_cost) {
                        min_cost = cost;
                        result_spline = successful_paths[i];
                        success = true;
                    }
                }
            }

            return success;
        }



        bool plan(Point start, Point end, double sigma, Point limits,
                  const size_t sample_count = 50, const size_t check_pts = 50, const size_t init_pts = 10) {
            Spline_t init_spline;
            initialize(start, end, init_spline, init_pts);

            std::default_random_engine generator(omp_get_thread_num());  // Seeded by thread ID for unique generators
            std::normal_distribution<double> distribution(0.0, sigma);

            std::vector<Spline_t> successful_paths;  // Vector to store successful splines

#pragma omp parallel for schedule(dynamic, 1)
            for (size_t i = 0; i < sample_count; ++i) {
                auto sampled_spline = sample_with_generator(init_spline, sigma, limits, generator, distribution);
                int thread_id = omp_get_thread_num();
                mjData* d_thread = d_copies[thread_id];

                if (!check_collision(sampled_spline, check_pts, d_thread)) {
                    // Critical section to safely add successful splines to the vector
#pragma omp critical
                    {
                        successful_paths.push_back(sampled_spline);
                    }
                }
            }

            // Print the number of successful paths found
            std::cout << "Sampled " << sample_count << " splines. Successful paths found: " << successful_paths.size() << std::endl;

            Spline_t result_spline;
            auto success = eval_cost(successful_paths, result_spline, check_pts);

            if (success) {
                path_spline = result_spline;
            }

            return success;
        }
    };


} // sspp

#endif //SSPP_SSPP_H
