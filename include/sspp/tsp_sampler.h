#pragma once
#include "tsp_types.h"
#include <random>

namespace tsp {

struct Sampler {
    Point limits_min, limits_max;

    Sampler(const Point& lo, const Point& hi) : limits_min(lo), limits_max(hi) {}

    Point sample(const Point& mean, const Point& sigma) {
        static thread_local std::mt19937 gen(std::random_device{}());
        Point pt;
        // xyz truncated normal
        for (int i = 0; i < 3; ++i) {
            std::normal_distribution<double> dist(mean[i], sigma[i]);
            const double lo = limits_min[i], hi = limits_max[i];
            double s; int tries=0;
            do {
                s = dist(gen);
                if (++tries >= 100) { std::uniform_real_distribution<double> uni(lo,hi); s = uni(gen); break; }
            } while (s < lo || s > hi);
            pt[i] = s;
        }
        // yaw wrap
        if (limits_min[3] != limits_max[3]) {
            std::normal_distribution<double> dist(mean[3], sigma[3]);
            double yaw = dist(gen);
            const double range = limits_max[3] - limits_min[3];
            while (yaw < limits_min[3]) yaw += range;
            while (yaw > limits_max[3]) yaw -= range;
            pt[3] = yaw;
        } else {
            pt[3] = mean[3];
        }
        return pt;
    }

    ViaSet sample_set(const std::vector<Point>& mean_set,
                      const std::vector<Point>& sigma_set,
                      double z_min) {
        const size_t K = mean_set.size();
        ViaSet vias; vias.reserve(K);
        for (size_t i=0; i<K; ++i) {
            Point p = sample(mean_set[i], sigma_set[i]);
            p[2] = std::max(p[2], z_min);
            vias.push_back(std::move(p));
        }
        return vias;
    }
};

} // namespace tsp
