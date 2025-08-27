#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include <mujoco/mujoco.h>
#include <Eigen/Core>
#include <GLFW/glfw3.h>
#include <omp.h>

//#include "tsp.h"
#include "sspp/tsp.h"
#include "utility.h"
#include "visu.h"
#include "ui.h"
#include "Timer.h"

using namespace visu;
using namespace Utility;
using Point = tsp::Point;

// MuJoCo globals
mjModel* m = nullptr;
mjData*  d = nullptr;
mjvCamera cam; mjvOption opt; mjvScene scn; mjrContext con;

constexpr float SPHERE_SIZE = 0.01f;
constexpr float PATH_WIDTH  = 0.2f;

// Runtime state for visualization
std::vector<tsp::PathCandidate> path_candidates, failed_candidates;
std::vector<Point> via_pts, sampled_via_pts;

bool first_plan_call = true;

// ---------- Helpers ----------

static void report_planning_statistics(const std::vector<double>& ms,
                                       const std::vector<tsp::PathCandidate>& ok,
                                       const std::vector<tsp::PathCandidate>& fail,
                                       const tsp::TaskSpacePlanner& planner,
                                       const std::string& tag) {
    std::cout << "-- " << tag << " --\n";
    if (!ms.empty()) {
        const double mean = std::accumulate(ms.begin(), ms.end(), 0.0) / ms.size();
        const double sq   = std::inner_product(ms.begin(), ms.end(), ms.begin(), 0.0);
        const double sd   = std::sqrt(std::max(0.0, sq / ms.size() - mean * mean));
        const auto [mn, mx] = std::minmax_element(ms.begin(), ms.end());
        std::cout << "Mean planning time: " << mean << " ms\n"
                  << "Std. deviation   : " << sd   << " ms\n"
                  << "Min / Max        : " << *mn  << " / " << *mx << " ms\n";
    } else {
        std::cout << "No planning attempts recorded.\n";
    }
    std::cout << "Successful candidates: " << ok.size()   << "\n"
              << "Failed candidates    : " << fail.size() << "\n";

    const Point mu  = planner.get_current_mean();
    const Point sig = planner.get_current_stddev();
    std::cout << "Mean  : [" << mu.transpose()  << "]\n"
              << "Stddev: [" << sig.transpose() << "]\n";
}

static std::vector<tsp::PathCandidate>
run_planning(tsp::TaskSpacePlanner& planner,
             const std::function<std::vector<tsp::PathCandidate>()>& fn,
             Timer& tmr,
             std::vector<double>& ms_hist) {
    planner.reset();
    tmr.tic();
    auto out = fn();
    const auto us = tmr.toc();
    ms_hist.push_back(double(us) / 1e6);
    return out;
}

static void execute_planning_cycle(tsp::TaskSpacePlanner& planner,
                                   const Point& start_pt,
                                   const Point& end_pt,
                                   const Utility::BodyJointInfo& coll_body_info,
                                   Timer& exec_timer,
                                   std::vector<double>& ms_hist,
                                   const std::string& report_name = "Planning attempt") {
    const bool iterate_flag = !first_plan_call;

    path_candidates = run_planning(planner, [&]{
        return planner.plan(start_pt, end_pt, iterate_flag);
    }, exec_timer, ms_hist);

    first_plan_call = false;

    failed_candidates = planner.get_failed_path_candidates();
    report_planning_statistics(ms_hist, path_candidates, failed_candidates, planner, report_name);

    auto ctrl_pts = planner.get_ctrl_pts();
    std::cout << "Spline control points:\n" << ctrl_pts << std::endl;

    sampled_via_pts = planner.get_sampled_via_pts();

    // get_via_pts() returns a const& in the planner; copy into local vector for visu
    {
        const auto& init_via = planner.get_via_pts();
        via_pts.assign(init_via.begin(), init_via.end());
    }

    // For testing: set end point on the body in MuJoCo
    Utility::mj_set_point(end_pt, coll_body_info, d);
}

// ---------- Main ----------

int main(int argc, char** argv) {
    // OpenMP environment dump
#ifdef _OPENMP
    std::cout << "[OMP] OpenMP enabled\n"
              << "[OMP] Max threads    : " << omp_get_max_threads() << "\n"
              << "[OMP] Num processors : " << omp_get_num_procs()   << "\n"
              << "[OMP] Dynamic threads: " << omp_get_dynamic()     << "\n"
              << "[OMP] Nested parallel: " << omp_get_nested()      << "\n";
#else
    std::cout << "[OMP] OpenMP not enabled (compiled without -fopenmp)\n";
#endif

    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <model_file> <collision_body_name>\n";
        return 1;
    }
    const std::string modelFileArg        = argv[1];
    const std::string collisionBodyName   = argv[2];

    Timer exec_timer;
    std::vector<double> duration_ms;

    // MuJoCo init
    std::cout << "MuJoCo version: " << mj_version() << "\n"
              << "Eigen version : " << EIGEN_WORLD_VERSION << "."
              << EIGEN_MAJOR_VERSION << "."
              << EIGEN_MINOR_VERSION << "\n";

    char errbuf[1000];
    m = mj_loadXML(modelFileArg.c_str(), nullptr, errbuf, sizeof(errbuf));
    if (!m) throw std::runtime_error(std::string("Failed to load MuJoCo model: ") + errbuf);
    d = mj_makeData(m);

    Utility::print_body_info(m);
    std::cout << "Taskspace Planner\nDoFs: " << m->nq << "\n";
    print_menue();

    mj_forward(m, d);
    mj_collision(m, d);

    // Planner configuration
    const std::string coll_body_name = collisionBodyName;
    const auto coll_body_info = get_free_body_joint_info(coll_body_name, m);

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
    const double sigma_floor        = 0.001;
    const double var_ema_beta       = 0.2;
    const double mean_lr            = 0.5;
    const double max_step_norm      = 0.1;
    const double floor_margin       = 0.01;
    const double floor_penalty_scale= 10.0;

    Point limit_max, limit_min;
    limit_max << 0.7,  0.7, 0.6,  1.6;
    limit_min << 0.0, -0.7, 0.1, -1.6;

    tsp::TaskSpacePlanner path_planner(
            m, coll_body_name,
            stddev_initial, stddev_min, stddev_max,
            stddev_increase_factor, stddev_decay_factor,
            elite_fraction, sample_count, check_points,
            gd_iterations, init_points, collision_weight, z_min,
            limit_min, limit_max, use_gradient_descent,
            sigma_floor, var_ema_beta, mean_lr, max_step_norm,
            floor_margin, floor_penalty_scale
    );

    // Start/end points
    const std::string start_body_name = "block_green/";
    const std::string end_body_name   = "block_orange/";
    Point end_pt   = Utility::get_body_point<Point>(m, d, end_body_name);
    Point start_pt = Utility::get_body_point<Point>(m, d, start_body_name);
    end_pt[2]   += 0.02;
    start_pt[2] += 0.02;

    // Initial plan
    execute_planning_cycle(path_planner, start_pt, end_pt, coll_body_info, exec_timer, duration_ms, "Initial planning (new path)");

    // GLFW init
    if (!glfwInit()) mju_error("Could not initialize GLFW");
    GLFWwindow* window = glfwCreateWindow(1200, 900, "Get to the Choppa!", nullptr, nullptr);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    // Visualization setup
    mjv_defaultCamera(&cam); mjv_defaultOption(&opt); mjv_defaultScene(&scn); mjr_defaultContext(&con);
    mjv_makeScene(m, &scn, 4000);
    mjr_makeContext(m, &con, mjFONTSCALE_150);

    // Callbacks
    glfwSetKeyCallback(window, keyboard_cb);
    glfwSetCursorPosCallback(window, mouse_move_cb);
    glfwSetMouseButtonCallback(window, mouse_button_cb);
    glfwSetScrollCallback(window, scroll_cb);

    const int pts_cnt = 10;

    // Main loop
    while (!glfwWindowShouldClose(window)) {
        mj_forward(m, d);
        d->time += 1e-2;

        mjrRect viewport{0, 0, 0, 0};
        glfwGetFramebufferSize(window, &viewport.width, &viewport.height);

        mjv_updateScene(m, d, &opt, nullptr, &cam, mjCAT_ALL, &scn);

        if (via_pts.size() >= 3) {
            draw_sphere(&scn, convert_point(via_pts[0]), SPHERE_SIZE, start_color);
            draw_sphere(&scn, convert_point(via_pts[2]), SPHERE_SIZE, end_color);
        }

        if (vis_best_path) {
            auto pts = path_planner.get_path_pts(pts_cnt);
            draw_path(&scn, convert_point_vector(pts), PATH_WIDTH, path_color);
        }

        visualize_via_pts(vis_sampled_via_pts, sampled_via_pts, path_planner, scn, pts_cnt,
                          sampled_via_path_color, sampled_via_color);

        visualize_candidates(vis_succ_candidates, vis_grad_descent, path_candidates, path_planner, scn, pts_cnt,
                             path_color, via_color, graddesc_color, graddesc_via_color);
        visualize_candidates(vis_failed_candidates, vis_grad_descent, failed_candidates, path_planner, scn, pts_cnt,
                             failed_path_color, failed_via_color, failed_graddesc_color, failed_graddesc_via_color);

        if (vis_animate_block) {
            const double u = std::fmod(d->time / 10.0, 1.0);
            const Point  p = path_planner.evaluate(u);
            Utility::mj_set_point(p, coll_body_info, d);
        }

        if (flag_plan_path) {
            std::cout << "debug out: iterative planning refinement\n";
            execute_planning_cycle(path_planner, start_pt, end_pt, coll_body_info, exec_timer, duration_ms, "Iterative refinement");
            flag_plan_path = false;
        }

        mjr_render(viewport, &scn, &con);
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Cleanup
    mjv_freeScene(&scn);
    mjr_freeContext(&con);
    mj_deleteData(d);
    mj_deleteModel(m);
    return 0;
}
