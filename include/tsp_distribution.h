#pragma once
#include "tsp_types.h"

namespace tsp {

    struct Distribution {
        Point mean{Point::Zero()}, sigma{Point::Ones()*0.3};
        double sigma_floor{0.0};
        double var_ema_beta{0.2};
        double mean_lr{0.5};
        double stddev_min{1e-3}, stddev_max{0.5};
        Point lo{ -Point::Ones()*2.0 }, hi{ Point::Ones()*2.0 };
        double z_min{0.0};

        void reset(const Point& m0, double s0){
            mean = m0;
            mean[2] = std::max(mean[2], z_min);
            sigma = Point::Ones()*s0;
            for (int i=0;i<kDOF;++i) mean[i] = std::clamp(mean[i], lo[i], hi[i]);
        }

        void adapt(bool success, double inc=1.5, double dec=0.95){
            sigma *= (success ? dec : inc);
            sigma  = sigma.cwiseMax(Point::Ones()*stddev_min)
                    .cwiseMin(Point::Ones()*stddev_max)
                    .cwiseMax(Point::Ones()*sigma_floor);
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
                    EliteGetter get_via)
        {
            // mean
            Point elite_mean = Point::Zero();
            for (size_t i=0;i<elite_idx.size();++i)
                elite_mean += w[i] * get_via(C[elite_idx[i]]);
            Point new_mean = mean + mean_lr * (elite_mean - mean);
            new_mean[2] = std::max(new_mean[2], z_min);
            for (int i=0;i<kDOF;++i) new_mean[i] = std::clamp(new_mean[i], lo[i], hi[i]);
            mean = new_mean;

            // variance
            Point var_elite = Point::Zero();
            for (size_t i=0;i<elite_idx.size();++i){
                Point rp = get_via(C[elite_idx[i]]);
                Point d  = rp - mean;
                if (lo[3] != hi[3]) d[3] = wrap_angle_diff(rp[3], mean[3], lo[3], hi[3]);
                var_elite += w[i] * d.cwiseProduct(d);
            }
            const Point prev_var = sigma.cwiseProduct(sigma);
            const Point blend    = (1.0 - var_ema_beta) * prev_var + var_ema_beta * var_elite;
            sigma = blend.cwiseSqrt();
            sigma = sigma.cwiseMax(Point::Ones()*stddev_min)
                    .cwiseMin(Point::Ones()*stddev_max)
                    .cwiseMax(Point::Ones()*sigma_floor);
        }
    };

} // namespace tsp
