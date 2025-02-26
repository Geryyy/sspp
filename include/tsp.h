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
    struct CollisionPoint{
        double spline_param;
        double collision_distance;
    };

    class TaskSpacePlanner
    {
    public:
        static constexpr int kDOF = 3;
        static constexpr int kSplineDegree = 3;
        using Point = Eigen::Matrix<double, kDOF, 1>;
        using Spline = Eigen::Spline<double, kDOF, kSplineDegree>;

    private:
        using SplineFitter = Eigen::SplineFitting<Spline>;

        Spline path_spline_;
        mjModel *model_;
        char error_buffer_[1000]; // Buffer for storing MuJoCo error messages
        std::vector<mjData *> data_copies_;
        std::default_random_engine generator_;

    public:
        TaskSpacePlanner(mjModel *model, mjData *data)
            : model_(model)
        {
            initializeDataCopies(data);
        }

        TaskSpacePlanner(const std::string &xml_string)
        {
            // Parse the model from the XML string
            model_ = mj_loadXML(xml_string.c_str(), nullptr, error_buffer_, sizeof(error_buffer_));
            if (!model_)
            {
                throw std::runtime_error("Failed to load MuJoCo model from XML: " + std::string(error_buffer_));
            }

            // Create the mjData structure associated with the model
            auto data_ = mj_makeData(model_);
            if (!data_)
            {
                mj_deleteModel(model_);
                throw std::runtime_error("Failed to create MuJoCo data structure.");
            }

            // Initialize data copies
            initializeDataCopies(data_);
        }

        // work around for python bindings
        TaskSpacePlanner(void *model_ptr, void *data_ptr)
        {
            // Cast void* to mjModel* and mjData*
            model_ = static_cast<mjModel *>(model_ptr);
            mjData *data_ = static_cast<mjData *>(data_ptr);
            initializeDataCopies(data_);
        }

        ~TaskSpacePlanner()
        {
            for (auto &data_copy : data_copies_)
            {
                mj_deleteData(data_copy);
            }
        }

        int initializePath(const Point &start, const Point &end, const Point &end_derivative, Spline &init_spline, int num_points = 10)
        {
            Eigen::VectorXd u_knots(num_points, 1);
            Eigen::MatrixXd via_points(kDOF, num_points);
            
            // linear placement of via points from start to end
            for (int i = 0; i < num_points; ++i)
            {
                double t = static_cast<double>(i) / (num_points - 1);
                Point point = (1 - t) * start + t * end;
                via_points.col(i) = point;
                u_knots(i) = t;
            }

            // init_spline = SplineFitter::Interpolate(via_points, kSplineDegree, u_knots);
            Eigen::MatrixXd derivatives = end_derivative;
            // derivates.push_back(end_derivates);
            Eigen::Vector<int, 1> deriv_ind(num_points -1);
//            deriv_ind << num_points -1;
//            std::cout << "via_points: " << via_points << std::endl;
//            std::cout << "derivatives: " << derivatives << std::endl;
//            std::cout << "deriv_ind: " << deriv_ind << std::endl;
//            std::cout << "u_knots: " << u_knots << std::endl;
//            init_spline = SplineFitter::Interpolate(via_points, kSplineDegree, u_knots);
            init_spline = SplineFitter::InterpolateWithDerivatives(via_points, derivatives, deriv_ind, kSplineDegree, u_knots);
            path_spline_ = init_spline;
            return 0;
        }

        Point evaluate(double u) const
        {
            return path_spline_(u);
        }

        Point evaluate(const Spline &spline, double u) const
        {
            return spline(u);
        }

        Spline::ControlPointVectorType get_ctrl_pts() const
        {
            return path_spline_.ctrls();
        }

        Spline sampleWithNoise(const Spline &init_spline, double sigma, const Point &limits, std::default_random_engine &generator)
        {
            std::normal_distribution<double> distribution(0.0, sigma);
            auto control_points = init_spline.ctrls();
            int p = kSplineDegree;

            // Apply noise to control points except boundaries

            for (int i = 0; i < control_points.rows(); ++i)
            {
//                std::cout << "ctr_pt no   noise: " << control_points.row(i) << std::endl;
                for (int j = p; j < control_points.cols() - p; ++j)
                {
                    control_points(i, j) += distribution(generator) * limits(i);
                }
//                std::cout << "ctr_pt with noise: " << control_points.row(i) << std::endl;
            }

            return Spline(init_spline.knots(), control_points);
        }

        bool checkCollision(const Spline &spline, int num_samples, mjData *data) const
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
                    if (data->contact[i].dist < -1e-3)
                    {
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

#pragma omp parallel for
            for (size_t i = 0; i < successful_paths.size(); ++i)
            {
                double cost = computeArcLength(successful_paths[i], check_points);
#pragma omp critical
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

        bool plan(const Point &start, const Point &end, const Point &end_derivative, double sigma, const Point &limits,
                  std::vector<Spline> &ret_success_paths, int sample_count = 50,
                  int check_points = 50, int init_points = 10)
        {

            Spline init_spline;
            initializePath(start, end, end_derivative, init_spline, init_points);
            std::vector<Spline> successful_paths;

#pragma omp parallel
            {
                std::default_random_engine thread_generator(omp_get_thread_num());

#pragma omp for schedule(dynamic, 1)
                for (int i = 0; i < sample_count; ++i)
                {
                    Spline sampled_spline = sampleWithNoise(init_spline, sigma, limits, thread_generator);
                    mjData *thread_data = data_copies_[omp_get_thread_num()];

                    if (!checkCollision(sampled_spline, check_points, thread_data))
                    {
#pragma omp critical
                        successful_paths.push_back(sampled_spline);
                    }
                }
            }

            std::cout << "Sampled " << sample_count << " splines. Successful paths found: " << successful_paths.size() << std::endl;

            ret_success_paths = successful_paths;
            return findBestPath(successful_paths, path_spline_, check_points);
        }

        bool plan(const Point &start, const Point &end, const Point &end_derivative, double sigma, const Point &limits, int sample_count = 50,
                  int check_points = 50, int init_points = 10)
        {
            std::vector<Spline> successful_paths;
            return plan(start, end, end_derivative, sigma, limits, successful_paths, sample_count, check_points, init_points);
        }

    private:
        void initializeDataCopies(mjData *data)
        {
            int max_threads = omp_get_max_threads();
            data_copies_.resize(max_threads, nullptr);

            for (auto &data_copy : data_copies_)
            {
                data_copy = mj_copyData(nullptr, model_, data);
            }
        }
    };

} // namespace sspp

#endif // spp_SAMPLING_PATH_PLANNER_H
