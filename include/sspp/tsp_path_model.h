#pragma once
#include "tsp_types.h"

namespace tsp {

struct PathModel {
    using Fitter = Eigen::SplineFitting<Spline>;
    std::vector<double> u_;
    std::vector<Point>  via_;    // includes start..end; interior slots are where K vias go

    void setupLinear(const Point& a, const Point& b, int total_points){
        u_.resize(total_points);
        via_.resize(total_points);
        for (int i = 0; i < total_points; ++i) {
            const double t = double(i) / (total_points - 1);
            u_[i]   = t;
            via_[i] = (1.0 - t)*a + t*b;     // linear initialization
        }
    }

    Spline initLinear(const Point& a, const Point& b, int total_points){
        setupLinear(a, b, total_points);
        Eigen::MatrixXd V(kDOF, via_.size());
        for (size_t i=0; i<via_.size(); ++i) V.col((Eigen::Index)i) = via_[i];
        return Fitter::Interpolate(
            V, kSplineDegree,
            Eigen::Map<Eigen::VectorXd>(u_.data(), (Eigen::Index)u_.size())
        );
    }

    // Insert K vias into interior indices [1 .. K] (assumes total_points = K+2)
    Spline fromVias(const ViaSet& vias) const {
        Eigen::MatrixXd V(kDOF, via_.size());
        for (size_t i=0; i<via_.size(); ++i) V.col((Eigen::Index)i) = via_[i];
        // enforce: interior count must match
        const size_t K = vias.size();
        for (size_t i=0; i<K; ++i) V.col((Eigen::Index)(1+i)) = vias[i];

        return Fitter::Interpolate(
            V, kSplineDegree,
            Eigen::Map<const Eigen::VectorXd>(u_.data(), (Eigen::Index)u_.size())
        );
    }

    const std::vector<Point>& initial_via_points() const { return via_; }
};

} // namespace tsp
