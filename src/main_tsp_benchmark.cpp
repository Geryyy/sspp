// main_benchmark.cpp
#include <iostream>
#include <string>
#include <vector>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <chrono>

#include <mujoco/mujoco.h>
#include <Eigen/Core>

#include "sspp/tsp.h"
#include "utility.h"

using Point = tsp::Point;

struct Stats {
    double mean_ms{0.0};
    double std_ms{0.0};
    double min_ms{0.0};
    double max_ms{0.0};
};

static Stats compute_stats(const std::vector<double>& ms) {
    Stats s{};
    if (ms.empty()) return s;
    const double n = static_cast<double>(ms.size());
    s.mean_ms = std::accumulate(ms.begin(), ms.end(), 0.0) / n;
    double sq_sum = std::inner_product(ms.begin(), ms.end(), ms.begin(), 0.0);
    s.std_ms = std::sqrt(std::max(0.0, sq_sum / n - s.mean_ms * s.mean_ms));
    auto [mn, mx] = std::minmax_element(ms.begin(), ms.end());
    s.min_ms = *mn;
    s.max_ms = *mx;
    return s;
}

// Approximate path length from sampled path points
static double compute_path_length(const tsp::TaskSpacePlanner& planner, int num_samples = 50) {
    auto pts = planner.get_path_pts(num_samples);
    if (pts.size() < 2) return 0.0;
    double length = 0.0;
    for (size_t i = 1; i < pts.size(); ++i) {
        length += (pts[i] - pts[i-1]).head<3>().norm(); // only xyz for distance
    }
    return length;
}

int main(int argc, char** argv) {
    if (argc < 3 || argc > 6) {
        std::cerr << "Usage: " << argv[0]
                  << " <model_file.xml> <collision_body_name> [N=50] [start_body=block_green/] [end_body=block_orange/]\n";
        return 1;
    }

    const std::string modelFile = argv[1];
    const std::string collisionBodyName = argv[2];
    const int N = (argc >= 4) ? std::max(1, std::atoi(argv[3])) : 50;
    const std::string start_body_name = (argc >= 5) ? argv[4] : "block_green/";
    const std::string end_body_name   = (argc >= 6) ? argv[5] : "block_orange/";

    // --- Load MuJoCo model ---
    char error_buffer[1024];
    mjModel* m = mj_loadXML(modelFile.c_str(), nullptr, error_buffer, sizeof(error_buffer));
    if (!m) {
        std::cerr << "Failed to load MuJoCo model: " << error_buffer << "\n";
        return 2;
    }
    mjData* d = mj_makeData(m);

    // --- Basic env init ---
    mj_forward(m, d);
    mj_collision(m, d);

    // --- Planner setup ---
    const double stddev_initial         = 0.2;
    const double stddev_min             = 0.0001;
    const double stddev_max             = 0.5;
    const double stddev_increase_factor = 1.5;
    const double stddev_decay_factor    = 0.9;
    const double elite_fraction         = 0.3;
    const int    sample_count           = 15;
    const int    check_points           = 40;
    const int    gd_iterations          = 100;
    const int    init_points            = 3;
    const double collision_weight       = 1.0;
    const double z_min                  = 0.1;
    const bool   use_gradient_descent   = false; // GD disabled (paper focus on CES)

// Advanced knobs
    const double sigma_floor        = 0.005;
    const double var_ema_beta       = 0.2;
    const double mean_lr            = 0.5;
    const double max_step_norm      = 0.1;
    const double floor_margin       = 0.01;
    const double floor_penalty_scale= 10.0;

    Point limit_max, limit_min;
    limit_max << 0.7,  0.7, 0.6,  1.6;
    limit_min << 0.0, -0.7, 0.1, -1.6;

    tsp::TaskSpacePlanner planner(
            m, collisionBodyName,
            stddev_initial, stddev_min, stddev_max,
            stddev_increase_factor, stddev_decay_factor,
            elite_fraction, sample_count, check_points,
            gd_iterations, init_points, collision_weight, z_min,
            limit_min, limit_max, use_gradient_descent,
            sigma_floor, var_ema_beta, mean_lr, max_step_norm,
            floor_margin, floor_penalty_scale
    );


    // --- Start/End points ---
    const Utility::BodyJointInfo coll_body_info = Utility::get_free_body_joint_info(collisionBodyName, m);
    Point start_pt = Utility::get_body_point<Point>(m, d, start_body_name);
    Point end_pt   = Utility::get_body_point<Point>(m, d, end_body_name);
    start_pt[2] += 0.02;
    end_pt[2]   += 0.02;

    // --- Benchmark: cold-start ---
    std::vector<double> times_cold_ms;
    double total_length_cold = 0.0;
    int successes_cold = 0;

    for (int i = 0; i < N; ++i) {
        const auto t0 = std::chrono::high_resolution_clock::now();
        auto succ = planner.plan(start_pt, end_pt, /*iterate_flag=*/false);
        const auto t1 = std::chrono::high_resolution_clock::now();
        const double dt_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        times_cold_ms.push_back(dt_ms);

        if (!succ.empty()) {
            successes_cold++;
            total_length_cold += compute_path_length(planner);
        }
    }

    // --- Benchmark: warm-start ---
    std::vector<double> times_warm_ms;
    double total_length_warm = 0.0;
    int successes_warm = 0;

    for (int i = 0; i < N; ++i) {
        const auto t0 = std::chrono::high_resolution_clock::now();
        auto succ = planner.plan(start_pt, end_pt, /*iterate_flag=*/true);
        const auto t1 = std::chrono::high_resolution_clock::now();
        const double dt_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        times_warm_ms.push_back(dt_ms);

        if (!succ.empty()) {
            successes_warm++;
            total_length_warm += compute_path_length(planner);
        }
    }

    // --- Stats ---
    const Stats cold = compute_stats(times_cold_ms);
    const Stats warm = compute_stats(times_warm_ms);

    std::cout << "\n=== Benchmark Results (N=" << N << ") ===\n";
    std::cout << "[Cold start] iterate=false\n";
    std::cout << "  Successes: " << successes_cold << " / " << N << "\n";
    std::cout << "  Mean time: " << cold.mean_ms << " ms,  Std: " << cold.std_ms
              << " ms,  Min: " << cold.min_ms << " ms,  Max: " << cold.max_ms << " ms\n";
    if (successes_cold > 0) {
        std::cout << "  Avg path length: " << (total_length_cold / successes_cold) << " m\n";
    }

    std::cout << "\n[Warm start] iterate=true\n";
    std::cout << "  Successes: " << successes_warm << " / " << N << "\n";
    std::cout << "  Mean time: " << warm.mean_ms << " ms,  Std: " << warm.std_ms
              << " ms,  Min: " << warm.min_ms << " ms,  Max: " << warm.max_ms << " ms\n";
    if (successes_warm > 0) {
        std::cout << "  Avg path length: " << (total_length_warm / successes_warm) << " m\n";
    }

    // Clean up
    mj_deleteData(d);
    mj_deleteModel(m);
    return 0;
}
