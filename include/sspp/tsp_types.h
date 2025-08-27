#pragma once
#include <Eigen/Core>
#include <unsupported/Eigen/Splines>
#include <optional>
#include <vector>

namespace tsp {
constexpr int kDOF = 4;
constexpr int kSplineDegree = 2;

using Point  = Eigen::Matrix<double, kDOF, 1>;
using Spline = Eigen::Spline<double, kDOF, kSplineDegree>;
using ViaSet = std::vector<Point>;  // K via points, ordered along the path


enum class SolverStatus { Converged, Failed, BelowFloor, MaxIter, Unknown };

template<int DOF>
struct GradientStep {
    Eigen::Matrix<double, DOF, 1> x;
    double f{0.0};
};
using GradientStepType = GradientStep<kDOF>;

struct PathCandidate {
    ViaSet via;
    std::optional<ViaSet> refined;               // (unused in CES-only)
    std::vector<GradientStepType> steps;        // (unused in CES-only)
    SolverStatus status{SolverStatus::Failed};
    // cached metrics
    double L{-1.0};        // arc length
    double C_nf{-1.0};     // collision (no floor)
    double C_wf{-1.0};     // collision + floor penalty
};

inline const char* SolverStatustoString(SolverStatus s) {
    switch(s){
        case SolverStatus::Converged: return "Converged";
        case SolverStatus::Failed: return "Failed";
        case SolverStatus::BelowFloor: return "BelowFloor";
        case SolverStatus::MaxIter: return "MaxIter";
        default: return "Unknown";
    }
}
} // namespace tsp
