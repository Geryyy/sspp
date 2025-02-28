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

#endif //GRADIENT_H
