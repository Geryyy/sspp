//
// Created by gebmer on 28.02.25.
//

#ifndef GRADIENT_H
#define GRADIENT_H

#include <Eigen/Dense>
#include <functional>
#include <vector>

class Gradient {
    public:
        using CostFunction = std::function<double(const Eigen::Vector3d &)>;

        Gradient(CostFunction L, const Eigen::Vector3d &delta, const Eigen::Vector3d &point)
            : L(L), delta(delta), point(point) {
        }

        Eigen::Vector3d compute() const {
            Eigen::Vector3d grad;
            for (int i = 0; i < 3; ++i) {
                Eigen::Vector3d delta_vec = Eigen::Vector3d::Zero();
                delta_vec[i] = delta[i];
                grad[i] = (L(point + delta_vec) - L(point - delta_vec)) / (2.0 * delta[i]);
            }
            return grad;
        }

    private:
        CostFunction L;
        Eigen::Vector3d delta;
        Eigen::Vector3d point;
    };



    struct GradientStep
    {
        Eigen::Vector3d via_point;
        Eigen::Vector3d gradient;
    };



    class GradientDescent
    {
        std::vector<GradientStep> steps;

    public:
        using CostFunction = std::function<double(const Eigen::Vector3d &)>;

        GradientDescent(double step_size, int iterations, CostFunction cost_function, const Eigen::Vector3d &delta)
            : step_size_(step_size), iterations_(iterations), cost_function_(cost_function), delta_(delta), steps() {}

        Eigen::Vector3d optimize(Eigen::Vector3d via_candidate)
        {
            for (int k = 0; k < iterations_; ++k)
            {
                Gradient grad(cost_function_, delta_, via_candidate);
                Eigen::Vector3d coll_gradient = grad.compute();
                double adaptive_step = computeStepSize(coll_gradient, step_size_);

                GradientStep step(via_candidate, coll_gradient);
                steps.push_back(step);

                via_candidate -= adaptive_step * coll_gradient;               

                std::cout << "gd step " << k << " via_pt: " << via_candidate.transpose() << " gradient: " << coll_gradient.transpose() << std::endl;
            }
            return via_candidate;
        }

        std::vector<GradientStep> get_gradient_descent_steps(){
            return steps;
        }

    private:
        double step_size_;
        int iterations_;
        CostFunction cost_function_;
        Eigen::Vector3d delta_;

        double computeStepSize(const Eigen::Vector3d &gradient, double base_step = 1e-2)
        {
            double norm = gradient.norm();
            return (norm > 1e-6) ? base_step / norm : base_step;
            // return base_step;
        }
    };

#endif //GRADIENT_H
