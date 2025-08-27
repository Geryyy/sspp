#pragma once
#include "tsp_types.h"
#include "Collision.h"
#include <memory>
#include <vector>
#include <omp.h>
#include "mujoco/mujoco.h"

namespace tsp {

// thin wrapper that provides per-thread Collision<Point>
struct CollisionWorld {
    std::string body_name;
    mjModel* model{nullptr};
    std::vector<std::shared_ptr<Collision<Point>>> pool;

    CollisionWorld(mjModel* m, std::string body) : body_name(std::move(body)), model(m) {
        ensurePool();
    }

    void ensurePool() {
        const int need = omp_get_max_threads();
        if ((int)pool.size() == need) return;
        pool.clear(); pool.reserve(need);
        for (int i=0;i<need;++i)
            pool.push_back(std::make_shared<Collision<Point>>(body_name, model));
    }

    inline Collision<Point>& env() {
#ifdef _OPENMP
        int tid = omp_get_thread_num();
#else
        int tid = 0;
#endif
        if (tid < 0 || tid >= (int)pool.size()) tid = 0;
        return *pool[tid];
    }

    inline double pointCost(const Point& p, bool use_center=true) {
        return env().collision_point_cost(p, use_center);
    }
};

} // namespace tsp
