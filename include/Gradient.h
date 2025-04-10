//
// Created by gebmer on 28.02.25.
//

#ifndef GRADIENT_H
#define GRADIENT_H

#include <Eigen/Dense>
#include <functional>
#include <vector>

namespace tsp {

    template <int kDOF>
    class Gradient {
        using Point = Eigen::Matrix<double, kDOF, 1>;
    public:
        using CostFunction = std::function<double(const Point &)>;

        Gradient(CostFunction L, const Point &delta, const Point &point)
                : L(L), delta(delta), point(point) {
        }

        Point compute() const {
            Point grad;

            for (int i = 0; i < 3; ++i) {
                Point delta_vec = Point::Zero();
                delta_vec[i] = delta[i];
                grad[i] = (L(point + delta_vec) - L(point - delta_vec)) / (2.0 * delta[i]);
            }
            return grad;
        }

    private:
        CostFunction L;
        Point delta;
        Point point;
    };

    template <int kDOF>
    struct GradientStep {
        using Point = Eigen::Matrix<double, kDOF, 1>;
        Point via_point;
        Point gradient;
        double cost;

        GradientStep(Point via_point, Point gradient, double cost) : via_point(via_point), gradient(gradient),
                                                                     cost(cost) {}
    };

    enum class SolverStatus {
        Converged = 0,             // Gradient vanishes
        MaxIterationsExceeded = 1, // Maximum iterations reached
        BelowFloor = 2,             // Via point below floor
        Invalid = 3                   // default case
    };

    inline std::string SolverStatustoString(SolverStatus status) {
        switch (status) {
            case SolverStatus::Converged:
                return "Converged";
            case SolverStatus::MaxIterationsExceeded:
                return "MaxIterationsExceeded";
            case SolverStatus::BelowFloor:
                return "BelowFloor";
            default:
                return "Invalid";
        }
    }

// Gradient Descent class with Barzilai-Borwein step size
    template <int kDOF>
    class GradientDescent {
        using Point = Eigen::Matrix<double, kDOF, 1>;
        using GradientStepType = GradientStep<kDOF>;
    public:
        using CostFunction = std::function<double(const Point &)>;

        GradientDescent(double step_size, int iterations, CostFunction cost_function, const Point &delta)
                : iterations_(iterations), cost_function_(cost_function), delta_(delta), steps(),
                  init_step_size_(step_size) {}

        SolverStatus optimize(Point via_candidate) {
            SolverStatus status = SolverStatus::Invalid;

            Point prev_x = via_candidate;
            Gradient prev_grad(cost_function_, delta_, via_candidate);
            Point prev_gradient = prev_grad.compute();
            double step_size = init_step_size_; // Initial step size

            for (int k = 0; k < iterations_; ++k) {
                Gradient grad(cost_function_, delta_, via_candidate);
                Point gradient = grad.compute();

                // Compute BB step size after the first iteration
                if (k > 0) {
                    Point s = via_candidate - prev_x;   // s_k = x_k - x_{k-1}
                    Point y = gradient - prev_gradient; // y_k = ∇f_k - ∇f_{k-1}

                    double denom = y.dot(y);
                    if (denom > 1e-10) {                                 // Avoid division by zero
                        step_size = s.dot(y) / denom; // Barzilai-Borwein step size
                    }
                }

                // Store step data
                GradientStep step(via_candidate, gradient, cost_function_(via_candidate));
                steps.push_back(step);

                // std::cout << "gd step " << k << " via_pt: " << via_candidate.transpose()
                // << " gradient: " << gradient.transpose()
                // << " step_size: " << step_size << std::endl;

                // abort if gradient vanishes --> no collision
                if (gradient.norm() < 1e-6) {
                    // no collision
                    // std::cerr << "gradient vanishes - abort" << std::endl;
                    // std::cerr << "gradient.norm: " << gradient.norm() << std::endl;
                    // std::cout << "graddesc converged!" << std::endl;
                    status = SolverStatus::Converged;
                    break;
                }

                // abort if below ground
                if (via_candidate[2] < 0) {
                    // std::cout << "graddesc below floor!" << std::endl;
                    status = SolverStatus::BelowFloor;
                    break;
                }

                // Gradient descent update
                prev_x = via_candidate;
                prev_gradient = gradient;
                via_candidate -= step_size * gradient;

                if (k == iterations_ - 1) {
                    // std::cout << "graddesc max steps!" << std::endl;
                    status = SolverStatus::MaxIterationsExceeded;
                }
            }
            result = via_candidate;
            return status;
        }

        Point get_result() {
            return result;
        }

        std::vector<GradientStepType> get_gradient_descent_steps() {
            return steps;
        }

    private:
        int iterations_;
        double init_step_size_;
        CostFunction cost_function_;
        Point delta_;
        std::vector<GradientStepType> steps;
        Point result;
    };
} // namespace tsp
#endif // GRADIENT_H
