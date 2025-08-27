// main_bench.cpp — Benchmark for modular C++ TaskSpacePlanner
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include <mujoco/mujoco.h>
#include <Eigen/Core>

#include <sspp/tsp.h>      // your modular planner adapter
#include "utility.h"       // get_free_body_joint_info, get_body_point

using Point = tsp::Point;
using Clock = std::chrono::high_resolution_clock;

struct Stats {
    double mean_ms{0.0}, std_ms{0.0}, min_ms{0.0}, max_ms{0.0};
};
static Stats compute_stats(const std::vector<double>& ms) {
    Stats s{};
    if (ms.empty()) return s;
    const double n = static_cast<double>(ms.size());
    s.mean_ms = std::accumulate(ms.begin(), ms.end(), 0.0) / n;
    const double sq = std::inner_product(ms.begin(), ms.end(), ms.begin(), 0.0);
    s.std_ms  = std::sqrt(std::max(0.0, sq / n - s.mean_ms * s.mean_ms));
    const auto [mn, mx] = std::minmax_element(ms.begin(), ms.end());
    s.min_ms = *mn; s.max_ms = *mx;
    return s;
}

// XYZ path length from planner's current best spline
static double compute_path_length(const tsp::TaskSpacePlanner& planner, int samples = 60) {
    const auto pts = planner.get_path_pts(samples);
    if (pts.size() < 2) return 0.0;
    double L = 0.0;
    for (size_t i = 1; i < pts.size(); ++i) L += (pts[i] - pts[i - 1]).head<3>().norm();
    return L;
}

// Run a single "converged" solve: first call iterate=false, then iterate=true up to max_iter-1
// Returns (time_ms, success, path_length_if_success_else_0)
static std::tuple<double, bool, double>
run_converged(tsp::TaskSpacePlanner& planner,
              const Point& start_pt, const Point& end_pt,
              int max_iter)
{
    const auto t0 = Clock::now();

    // First iteration (fresh or continuing depends on whether caller reset the planner)
    auto succ = planner.plan(start_pt, end_pt, /*iterate=*/false);
    bool any_success = !succ.empty();

    // Subsequent refinement iterations
    for (int k = 1; k < max_iter; ++k) {
        succ = planner.plan(start_pt, end_pt, /*iterate=*/true);
        any_success = any_success || !succ.empty();
    }

    const auto t1 = Clock::now();
    const double dt_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double L = any_success ? compute_path_length(planner) : 0.0;
    return {dt_ms, any_success, L};
}

// "Anytime" solve: iterate repeatedly until budget_ms elapsed (chunk = one planner iteration)
// Returns (time_ms_used, success_seen, best_path_length_if_success_else_0)
static std::tuple<double, bool, double>
run_anytime(tsp::TaskSpacePlanner& planner,
            const Point& start_pt, const Point& end_pt,
            double budget_ms)
{
    const auto t0 = Clock::now();
    bool any_success = false;
    double best_L = std::numeric_limits<double>::infinity();

    // First iteration as non-iterative, then keep iterating until budget
    auto succ = planner.plan(start_pt, end_pt, /*iterate=*/false);
    if (!succ.empty()) { any_success = true; best_L = std::min(best_L, compute_path_length(planner)); }

    for (;;) {
        const auto now = Clock::now();
        const double elapsed = std::chrono::duration<double, std::milli>(now - t0).count();
        if (elapsed >= budget_ms) break;

        succ = planner.plan(start_pt, end_pt, /*iterate=*/true);
        if (!succ.empty()) {
            any_success = true;
            best_L = std::min(best_L, compute_path_length(planner));
        }
    }

    const auto t1 = Clock::now();
    const double used_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return {used_ms, any_success, any_success ? best_L : 0.0};
}

// Simple comma-separated list parser (e.g., "20,50,100")
static std::vector<int> parse_budgets(const std::string& s) {
    std::vector<int> out;
    size_t start = 0;
    while (start < s.size()) {
        size_t comma = s.find(',', start);
        std::string token = (comma == std::string::npos) ? s.substr(start) : s.substr(start, comma - start);
        if (!token.empty()) out.push_back(std::max(1, std::atoi(token.c_str())));
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    return out;
}

int main(int argc, char** argv) {
#ifdef _OPENMP
    std::cout << "[OMP] OpenMP enabled\n"
              << "[OMP] Max threads    : " << omp_get_max_threads() << "\n"
              << "[OMP] Num processors : " << omp_get_num_procs()   << "\n"
              << "[OMP] Dynamic threads: " << omp_get_dynamic()     << "\n"
              << "[OMP] Nested parallel: " << omp_get_nested()      << "\n";
#else
    std::cout << "[OMP] OpenMP not enabled (compiled without -fopenmp)\n";
#endif
    // Usage:
    //   main_bench <model.xml> <collision_body> [N=50] [start_body=block_green/] [end_body=block_orange/] [num_vias=1]
    //              [--max_iter=60] [--budgets_ms=20,50,100]
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <model.xml> <collision_body> [N=50] [start_body=block_green/] [end_body=block_orange/] [num_vias=1]\n"
                  << "        [--max_iter=60] [--budgets_ms=]\n";
        return 1;
    }

    const std::string modelFile       = argv[1];
    const std::string collisionBody   = argv[2];
    int    N                          = (argc >= 4 && argv[3][0] != '-') ? std::max(1, std::atoi(argv[3])) : 50;
    std::string start_body_name       = (argc >= 5 && argv[4][0] != '-') ? argv[4] : "block_green/";
    std::string end_body_name         = (argc >= 6 && argv[5][0] != '-') ? argv[5] : "block_orange/";
    int    num_vias                   = (argc >= 7 && argv[6][0] != '-') ? std::max(0, std::atoi(argv[6])) : 1;

    int max_iter = 1;
    std::vector<int> budgets_ms; // empty => skip anytime

    // Parse optional flags
    for (int i = 7; i < argc; ++i) {
        std::string a = argv[i];
        if (a.rfind("--max_iter=", 0) == 0) {
            max_iter = std::max(1, std::atoi(a.substr(11).c_str()));
        } else if (a.rfind("--budgets_ms=", 0) == 0) {
            budgets_ms = parse_budgets(a.substr(13));
        }
    }

    // --- Load MuJoCo model ---
    char error_buffer[1024];
    mjModel* m = mj_loadXML(modelFile.c_str(), nullptr, error_buffer, sizeof(error_buffer));
    if (!m) { std::cerr << "Failed to load MuJoCo model: " << error_buffer << "\n"; return 2; }
    mjData* d = mj_makeData(m);
    mj_forward(m, d);
    mj_collision(m, d);

    // --- Planner setup (adapter API) ---
    const double stddev_initial         = 0.2;
    const double stddev_min             = 0.0001;
    const double stddev_max             = 0.5;
    const double stddev_increase_factor = 1.5;
    const double stddev_decay_factor    = 0.9;
    const double elite_fraction         = 0.3;
    const int    sample_count           = 15;
    const int    check_points           = 40;
    const int    gd_iterations          = 0;                 // GD off for CES paper
    const int    total_points           = num_vias + 2;      // start + K vias + end
    const double collision_weight       = 1.0;
    const double z_min                  = 0.1;
    const bool   use_gradient_descent   = false;

    const double sigma_floor         = 0.005;
    const double var_ema_beta        = 0.2;
    const double mean_lr             = 0.5;
    const double max_step_norm       = 0.1;
    const double floor_margin        = 0.01;
    const double floor_penalty_scale = 10.0;

    Point limit_max, limit_min;
    limit_max << 0.7,  0.7, 0.6,  1.6;
    limit_min << 0.0, -0.7, 0.1, -1.6;

    auto make_planner = [&]() {
        return tsp::TaskSpacePlanner(
            m, collisionBody,
            stddev_initial, stddev_min, stddev_max,
            stddev_increase_factor, stddev_decay_factor,
            elite_fraction, sample_count, check_points,
            gd_iterations, total_points, collision_weight, z_min,
            limit_min, limit_max, use_gradient_descent,
            sigma_floor, var_ema_beta, mean_lr, max_step_norm,
            floor_margin, floor_penalty_scale
        );
    };

    // --- Start/End points ---
    const Utility::BodyJointInfo coll_body_info = Utility::get_free_body_joint_info(collisionBody, m);
    Point q0 = Utility::get_body_point<Point>(m, d, start_body_name);
    Point qT = Utility::get_body_point<Point>(m, d, end_body_name);
    q0[2] += 0.02; qT[2] += 0.02;

    // =========================
    // Converged Benchmark
    // =========================

    // Cold start: new planner per run
    std::vector<double> times_cold; times_cold.reserve(N);
    int succ_cold = 0; double sumL_cold = 0.0;
    for (int i = 0; i < N; ++i) {
        auto planner = make_planner();             // fresh
        auto [ms, ok, L] = run_converged(planner, q0, qT, max_iter);
        times_cold.push_back(ms);
        if (ok) { ++succ_cold; sumL_cold += L; }
    }

    // Warm start: reuse one planner (persistent mean/sigma/forwarded best)
    std::vector<double> times_warm; times_warm.reserve(N);
    int succ_warm = 0; double sumL_warm = 0.0;
    {
        auto planner = make_planner();
        for (int i = 0; i < N; ++i) {
            auto [ms, ok, L] = run_converged(planner, q0, qT, max_iter);
            times_warm.push_back(ms);
            if (ok) { ++succ_warm; sumL_warm += L; }
        }
    }

    const Stats S_c = compute_stats(times_cold);
    const Stats S_w = compute_stats(times_warm);

    std::cout << "\n=== Converged Benchmark (max_iter=" << max_iter << ", N=" << N << ") ===\n";
    std::cout << "[Cold]  Successes: " << succ_cold << "/" << N
              << "  Time mean=" << S_c.mean_ms << " ms  std=" << S_c.std_ms
              << " ms  min=" << S_c.min_ms << " ms  max=" << S_c.max_ms << " ms\n";
    if (succ_cold) std::cout << "        Avg path length: " << (sumL_cold / succ_cold) << " m\n";
    std::cout << "[Warm]  Successes: " << succ_warm << "/" << N
              << "  Time mean=" << S_w.mean_ms << " ms  std=" << S_w.std_ms
              << " ms  min=" << S_w.min_ms << " ms  max=" << S_w.max_ms << " ms\n";
    if (succ_warm) std::cout << "        Avg path length: " << (sumL_warm / succ_warm) << " m\n";

    // =========================
    // Anytime Benchmark (optional budgets)
    // =========================
    if (!budgets_ms.empty()) {
        std::cout << "\n=== Anytime Benchmark (budgets: ";
        for (size_t i=0;i<budgets_ms.size();++i) {
            std::cout << budgets_ms[i] << (i+1<budgets_ms.size() ? "," : "");
        }
        std::cout << " ms, N=" << N << ") ===\n";

        for (int budget : budgets_ms) {
            // Cold: fresh per run
            std::vector<double> t_c; t_c.reserve(N);
            int sc=0; double Lc=0.0;
            for (int i=0;i<N;++i) {
                auto planner = make_planner();
                auto [ms, ok, L] = run_anytime(planner, q0, qT, double(budget));
                t_c.push_back(ms);
                if (ok) { ++sc; Lc += L; }
            }
            // Warm: reuse one instance
            std::vector<double> t_w; t_w.reserve(N);
            int sw=0; double Lw=0.0;
            {
                auto planner = make_planner();
                for (int i=0;i<N;++i) {
                    auto [ms, ok, L] = run_anytime(planner, q0, qT, double(budget));
                    t_w.push_back(ms);
                    if (ok) { ++sw; Lw += L; }
                }
            }

            const Stats C = compute_stats(t_c);
            const Stats W = compute_stats(t_w);

            std::cout << "[Budget " << budget << " ms]  Cold: "
                      << sc << "/" << N << "  mean=" << C.mean_ms << " ms  std=" << C.std_ms
                      << " ms  min=" << C.min_ms << " ms  max=" << C.max_ms << " ms";
            if (sc) std::cout << "  avgL=" << (Lc / sc) << " m";
            std::cout << "\n";

            std::cout << "                 Warm: "
                      << sw << "/" << N << "  mean=" << W.mean_ms << " ms  std=" << W.std_ms
                      << " ms  min=" << W.min_ms << " ms  max=" << W.max_ms << " ms";
            if (sw) std::cout << "  avgL=" << (Lw / sw) << " m";
            std::cout << "\n";
        }
    }

    // Cleanup
    mj_deleteData(d);
    mj_deleteModel(m);
    return 0;
}
