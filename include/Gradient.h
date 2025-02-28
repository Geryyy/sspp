//
// Created by gebmer on 28.02.25.
//

#ifndef GRADIENT_H
#define GRADIENT_H

#include <Eigen/Dense>
#include <functional>
#include <vector>

class Gradient
{
public:
    using CostFunction = std::function<double(const Eigen::Vector3d &)>;

    Gradient(CostFunction L, const Eigen::Vector3d &delta, const Eigen::Vector3d &point)
        : L(L), delta(delta), point(point)
    {
    }

    Eigen::Vector3d compute() const
    {
        Eigen::Vector3d grad;
        for (int i = 0; i < 3; ++i)
        {
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

// Gradient Descent class with Barzilai-Borwein step size
class GradientDescent
{
public:
    using CostFunction = std::function<double(const Eigen::Vector3d &)>;

    GradientDescent(double step_size, int iterations, CostFunction cost_function, const Eigen::Vector3d &delta)
        : iterations_(iterations), cost_function_(cost_function), delta_(delta), steps(), init_step_size_(step_size) {}

    Eigen::Vector3d optimize(Eigen::Vector3d via_candidate)
    {
        Eigen::Vector3d prev_x = via_candidate;
        Gradient prev_grad(cost_function_, delta_, via_candidate);
        Eigen::Vector3d prev_gradient = prev_grad.compute();
        double step_size = init_step_size_; // Initial step size

        for (int k = 0; k < iterations_; ++k)
        {
            Gradient grad(cost_function_, delta_, via_candidate);
            Eigen::Vector3d coll_gradient = grad.compute();


            // Compute BB step size after the first iteration
            if (k > 0)
            {
                Eigen::Vector3d s = via_candidate - prev_x;        // s_k = x_k - x_{k-1}
                Eigen::Vector3d y = coll_gradient - prev_gradient; // y_k = ∇f_k - ∇f_{k-1}

                double denom = y.dot(y);
                if (denom > 1e-10)
                {                                 // Avoid division by zero
                    step_size = s.dot(y) / denom; // Barzilai-Borwein step size
                }
            }

            // Store step data
            GradientStep step(via_candidate, coll_gradient);
            steps.push_back(step);
            
            // std::cout << "gd step " << k << " via_pt: " << via_candidate.transpose()
            // << " gradient: " << coll_gradient.transpose()
            // << " step_size: " << step_size << std::endl;
            
            // abort if gradient vanishes --> no collision
            if(coll_gradient.norm() < 1e-6){
                // no collision
                // std::cerr << "gradient vanishes - abort" << std::endl;
                // std::cerr << "coll_gradient.norm: " << coll_gradient.norm() << std::endl;
                break;
            }

            // abort if below ground
            if(via_candidate[2] < 0){
                break;
            }
            

            // Gradient descent update
            prev_x = via_candidate;
            prev_gradient = coll_gradient;
            via_candidate += step_size * coll_gradient;
        }
        return via_candidate;
    }

    std::vector<GradientStep> get_gradient_descent_steps()
    {
        return steps;
    }

private:
    int iterations_;
    double init_step_size_;
    CostFunction cost_function_;
    Eigen::Vector3d delta_;
    std::vector<GradientStep> steps;

};

#endif // GRADIENT_H
