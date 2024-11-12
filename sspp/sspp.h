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

    template<int DOF>
    class SSPP {
    public:
        using Point = Eigen::Matrix<double, DOF, 1>;
    private:
        static const int spline_deg = 3;
        typedef Eigen::Spline<double, DOF, spline_deg> Spline_t;
        typedef Eigen::SplineFitting<Spline_t> SplineFitting_t;

        Spline_t init_spline;
        Spline_t path_spline;

    public:
        SSPP() = default;
        ~SSPP() = default;

        int initialize(Point start, Point end, int num_pts){
            Eigen::VectorXd u_knots(num_pts, 1);
            Eigen::MatrixXd via_points(DOF, num_pts);

            for (int i = 0; i < num_pts; ++i) {
                double t = static_cast<double>(i) / (num_pts - 1);
                Point point = (1 - t) * start + t * end;  // Linear interpolation
                via_points.col(i) = point;
                u_knots(i) = t;
            }

            init_spline = SplineFitting_t::Interpolate(via_points, spline_deg, u_knots);
            return 0;
        }

        Point evaluate(double u) {
            return path_spline(u);
        }

        Point evaluate(Spline_t spline, double u) {
            return spline(u);
        }

        Spline_t sample(double sigma, Point limits) {
            auto ctrl_pts = init_spline.ctrls();
            int p = spline_deg;

            std::default_random_engine generator;
            std::normal_distribution<double> distribution(0.0,sigma);

            // Add noise to the control points
            for (int i = 0; i < ctrl_pts.rows(); ++i) {
                for (int j = p; j < ctrl_pts.cols() - p; ++j) {
                    double noise = distribution(generator);
                    ctrl_pts(i, j) += noise * limits(i);
                }
            }

            Spline_t sampled_spline(init_spline.knots(), ctrl_pts);
//            std::cout << "ctrl pts rows: " << ctrl_pts.rows() << " cols: " << ctrl_pts.cols() << std::endl;
            return sampled_spline;
        }


        bool check_collision(Spline_t spline, const int num_samples, mjModel* m, mjData* d) {
            for (int i = 0; i <= num_samples; ++i) {
                double u = static_cast<double>(i) / num_samples;
                Point point = spline(u);

                for (int j = 0; j < DOF; ++j) {
                    d->qpos[j] = point(j);
                }
                mj_collision(m, d);
                if (d->ncon > 0) {
                    // collision detected
                    return true;
                }
            }
            return false;
        }


        bool plan(Point start, Point end, double sigma, Point limits, mjModel* m, mjData* d) {
            constexpr int init_pts = 10;
            constexpr int check_pts = 100;
            constexpr size_t sample_count = 1000000;

            Timer exec_timer;

            std::cout << "plan path with max openMP threads: " << omp_get_max_threads() << std::endl;

            exec_timer.tic();
            initialize(start, end, init_pts);
            std::cout << "Initialization time: " << static_cast<double>(exec_timer.toc())*1e-3 << " us" << std::endl;

            // Creating separate mjData copies for each thread
            exec_timer.tic();
            std::vector<mjData*> d_copies(omp_get_max_threads(), nullptr);
            for (auto& d_copy : d_copies) {
                d_copy = mj_copyData(d_copy, m, d);
            }
            std::cout << "Data copy time: " << static_cast<double>(exec_timer.toc())*1e-3 << " us" << std::endl;

            bool success = false;
            size_t i = 0;

            exec_timer.tic();
#pragma omp parallel for shared(success, i)
            for (i = 0; i < sample_count; i++) {
                if (success) continue;  // Exit other threads if a path is already found

                auto sampled_spline = sample(sigma, limits);

                // Get the thread ID and its specific mjData instance
                int thread_id = omp_get_thread_num();
                mjData* d_thread = d_copies[thread_id];

                if (!check_collision(sampled_spline, check_pts, m, d_thread)) {
#pragma omp critical  // Protect access to path_spline and success
                    {
                        if (!success) {
                            path_spline = sampled_spline;
                            success = true;
                        }
                    }
                }
            }
            std::cout << "Planning time: " << static_cast<double>(exec_timer.toc())*1e-3 << " us" << std::endl;

            exec_timer.tic();
            // Cleanup mjData copies
            for (auto& d_copy : d_copies) {
                mj_deleteData(d_copy);
            }
            std::cout << "Data cleanup time: " << static_cast<double>(exec_timer.toc())*1e-3 << " us" << std::endl;

            std::cout << "Sampled " << i << " splines. Path found: " << success << std::endl;
            return success;
        }



    };

} // sspp

#endif //SSPP_SSPP_H
