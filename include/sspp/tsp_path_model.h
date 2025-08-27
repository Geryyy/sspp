#pragma once
#include "tsp_types.h"

namespace tsp {

struct PathModel {
    using Fitter = Eigen::SplineFitting<Spline>;
    std::vector<double> u_;
    std::vector<Point>  via_;

    void setupLinear(const Point& a, const Point& b, int npts){
        u_.resize(npts); via_.resize(npts);
        for (int i=0;i<npts;++i){
            const double t = double(i) / (npts-1);
            u_[i]   = t;
            via_[i] = (1.0 - t)*a + t*b;
        }
    }

    Spline initLinear(const Point& a, const Point& b, int npts){
        setupLinear(a,b,npts);
        Eigen::MatrixXd V(kDOF, via_.size());
        for (size_t i=0;i<via_.size();++i) V.col((Eigen::Index)i) = via_[i];
        return Fitter::Interpolate(V, kSplineDegree,
               Eigen::Map<Eigen::VectorXd>(u_.data(), (Eigen::Index)u_.size()));
    }

    Spline fromVia(const Point& via) const {
        Eigen::MatrixXd V(kDOF, via_.size());
        for (size_t i=0;i<via_.size();++i) V.col((Eigen::Index)i) = via_[i];
        V.col(1) = via; // middle control
        return Fitter::Interpolate(V, kSplineDegree,
               Eigen::Map<const Eigen::VectorXd>(u_.data(), (Eigen::Index)u_.size()));
    }

    const std::vector<Point>& initial_via_points() const { return via_; }
};

} // namespace tsp
