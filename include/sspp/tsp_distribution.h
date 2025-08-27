#pragma once
#include "tsp_types.h"

namespace tsp {

struct Distribution {
    std::vector<Point> mean_vias;    // size K
    std::vector<Point> sigma_vias;   // size K
    double sigma_floor{0.0};
    double var_ema_beta{0.2};
    double mean_lr{0.5};
    double stddev_min{1e-3}, stddev_max{0.5};
    Point lo{ -Point::Ones()*2.0 }, hi{ Point::Ones()*2.0 };
    double z_min{0.0};

    void reset(const ViaSet& mean0, double s0){
        mean_vias = mean0;
        sigma_vias.assign(mean0.size(), Point::Ones()*s0);
        // clamp z & bounds per via point
        for (auto& m : mean_vias) {
            m[2] = std::max(m[2], z_min);
            for (int d=0; d<kDOF; ++d) m[d] = std::clamp(m[d], lo[d], hi[d]);
        }
        for (auto& s : sigma_vias) {
            s = s.cwiseMax(Point::Ones()*stddev_min)
                 .cwiseMin(Point::Ones()*stddev_max)
                 .cwiseMax(Point::Ones()*sigma_floor);
        }
    }

    void adapt(bool success, double inc=1.5, double dec=0.95){
        for (auto& s : sigma_vias) {
            s *= (success ? dec : inc);
            s  = s.cwiseMax(Point::Ones()*stddev_min)
                 .cwiseMin(Point::Ones()*stddev_max)
                 .cwiseMax(Point::Ones()*sigma_floor);
        }
    }

    static double wrap_angle_diff(double a, double b, double min, double max){
        const double range = max - min;
        double d = a - b;
        while (d >  0.5*range) d -= range;
        while (d < -0.5*range) d += range;
        return d;
    }

    template<class EliteGetter>
    void update(const std::vector<PathCandidate>& C,
                const std::vector<size_t>& elite_idx,
                const std::vector<double>& w,
                EliteGetter get_vias)
    {
        const size_t K = mean_vias.size();
        // update each via index independently
        for (size_t i=0; i<K; ++i) {
            // weighted mean
            Point elite_mean = Point::Zero();
            for (size_t j=0; j<elite_idx.size(); ++j) {
                const auto& vs = get_vias(C[elite_idx[j]]);
                elite_mean += w[j] * vs[i];
            }
            Point new_mean = mean_vias[i] + mean_lr * (elite_mean - mean_vias[i]);
            new_mean[2] = std::max(new_mean[2], z_min);
            for (int d=0; d<kDOF; ++d) new_mean[d] = std::clamp(new_mean[d], lo[d], hi[d]);
            mean_vias[i] = new_mean;

            // weighted var (EMA)
            Point var_elite = Point::Zero();
            for (size_t j=0; j<elite_idx.size(); ++j) {
                const auto& vs = get_vias(C[elite_idx[j]]);
                Point diff = vs[i] - mean_vias[i];
                if (lo[3] != hi[3]) diff[3] = wrap_angle_diff(vs[i][3], mean_vias[i][3], lo[3], hi[3]);
                var_elite += w[j] * diff.cwiseProduct(diff);
            }
            const Point prev_var = sigma_vias[i].cwiseProduct(sigma_vias[i]);
            const Point blend    = (1.0 - var_ema_beta)*prev_var + var_ema_beta*var_elite;
            sigma_vias[i] = blend.cwiseSqrt()
                               .cwiseMax(Point::Ones()*stddev_min)
                               .cwiseMin(Point::Ones()*stddev_max)
                               .cwiseMax(Point::Ones()*sigma_floor);
        }
    }
};

} // namespace tsp
