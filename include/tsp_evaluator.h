#pragma once
#include "tsp_types.h"
#include <functional>

namespace tsp {

struct Evaluator {
    double z_min{0.0}, floor_margin{0.01}, floor_scale{10.0};

    inline double floorPenalty(const Point& p) const {
        const double tgt = z_min + floor_margin;
        const double deficit = tgt - p[2];
        return (deficit > 0.0) ? floor_scale * deficit * deficit : 0.0;
    }

    // One-pass: returns length, pure collision, and penalized
    template<class World>
    void eval_one_pass(const Spline& s, int cp, World& world,
                       double& L_out, double& Cnf_out, double& Cwf_out) const {
        const double du = 1.0 / cp;
        Point prev = s(0.0);
        double L=0.0, Cnf=0.0, Cwf=0.0;
        for (int i = 1; i <= cp; ++i) {
            const Point p = s(i*du);
            L   += (p - prev).norm();
            const double c = world.pointCost(p, /*use_center=*/true);
            Cnf += c;
            Cwf += c + floorPenalty(p);
            prev = p;
        }
        L_out = L; Cnf_out = Cnf; Cwf_out = Cwf;
    }

    template<class World>
    bool feasible_fast(const Spline& s, int cp, World& world) const {
        const double du = 1.0 / cp;
        for (int i=0;i<=cp;++i){
            if (world.pointCost(s(i*du), /*use_center=*/true) > 0.0) return false;
        }
        return true;
    }
};

} // namespace tsp
