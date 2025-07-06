#include <iostream>
#include <string>
#include <vector>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <thread>
#include <cstdio>
#include <cstring>

#include <mujoco/mujoco.h>
#include <Eigen/Core>
#include <GLFW/glfw3.h>
#include <omp.h>

#include "Timer.h"
#include "tsp.h"
#include "utility.h"
#include "visu.h"
#include "ui.h"

using namespace visu;
using namespace Utility;
using Point = tsp::Point;

// Path to the XML file for the MuJoCo model
// "/home/geraldebmer/repos/robocrane/mujoco_env_builder/presets/robocrane/robocrane.xml" "block_cyan"
const std::string modelFile = "/home/geraldebmer/repos/robocrane/sspp/mjcf/stacking.xml";

// MuJoCo data structures
mjModel* m = nullptr;
mjData* d = nullptr;
mjvCamera cam;
mjvOption opt;
mjvScene scn;
mjrContext con;

std::vector<tsp::PathCandidate> path_candidates;
std::vector<tsp::PathCandidate> failed_candidates;

// Function to print planning statistics
void report_planning_statistics(const std::vector<double>& duration_vec,
                                const std::vector<tsp::PathCandidate>& successful_candidates,
                                const std::vector<tsp::PathCandidate>& failed_candidates,
                                const tsp::TaskSpacePlanner& path_planner,
                                const std::string& attempt_name) {
    std::cout << "-- " << attempt_name << " --" << std::endl;
    if (!duration_vec.empty()) {
        double mean = std::accumulate(duration_vec.begin(), duration_vec.end(), 0.0) / duration_vec.size();
        double sq_sum = std::inner_product(duration_vec.begin(), duration_vec.end(), duration_vec.begin(), 0.0);
        double stdev = std::sqrt(sq_sum / duration_vec.size() - mean * mean);
        auto [min_it, max_it] = std::minmax_element(duration_vec.begin(), duration_vec.end());
        double min_val = (min_it != duration_vec.end()) ? *min_it : 0.0;
        double max_val = (max_it != duration_vec.end()) ? *max_it : 0.0;

        std::cout << "Mean planning time: " << mean << " ms" << std::endl;
        std::cout << "Standard Deviation: " << stdev << " ms" << std::endl;
        std::cout << "Min planning time: " << min_val << " ms" << std::endl;
        std::cout << "Max planning time: " << max_val << " ms" << std::endl;
    } else {
        std::cout << "No planning attempts recorded for " << attempt_name << "." << std::endl;
    }
    std::cout << "Number of successful path candidates: " << successful_candidates.size() << std::endl;
    std::cout << "Number of failed path candidates: " << failed_candidates.size() << std::endl;
//    std::cout << "Control Points of Spline: " << path_planner.get_ctrl_pts() << std::endl;
//    std::cout << "Shape of Control Points: " << path_planner.get_ctrl_pts().cols() << std::endl;
//    std::cout << "Knot Vector of Spline: " << path_planner.get_knot_vector() << std::endl;
}

// Function to run a planning attempt
std::vector<tsp::PathCandidate> run_planning(tsp::TaskSpacePlanner& path_planner,
                                             const std::function<std::vector<tsp::PathCandidate>(void)>& plan_function,
                                             Timer& exec_timer,
                                             std::vector<double>& duration_vec) {
    path_planner.reset();
    exec_timer.tic();
    std::vector<tsp::PathCandidate> candidates = plan_function();
    auto duration = exec_timer.toc();
    duration_vec.push_back(static_cast<double>(duration / 1e6));
    return candidates;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <model_file> <collision_body_name>" << std::endl;
        return 1;
    }
    const std::string modelFileArg = argv[1];
    const std::string collisionBodyNameArg = argv[2];

    Timer exec_timer;
    std::vector<double> duration_vec_first;
    std::vector<double> duration_vec_second;
    constexpr int N = 1;

    // Initialize MuJoCo
    std::cout << "MuJoCo version: " << mj_version() << std::endl;
    std::cout << "Eigen version: " << EIGEN_WORLD_VERSION << "."
              << EIGEN_MAJOR_VERSION << "."
              << EIGEN_MINOR_VERSION << std::endl;

    char error_buffer_[1000];
    m = mj_loadXML(modelFileArg.c_str(), nullptr, error_buffer_, sizeof(error_buffer_));
    if (!m) {
        throw std::runtime_error("Failed to load MuJoCo model from XML: " + std::string(error_buffer_));
    }
    d = mj_makeData(m);

   Utility::print_body_info(m);
//    return 0;

    std::cout << "Taskspace Planner" << std::endl;
    std::cout << "DoFs: " << m->nq << std::endl;

    print_menue();

    mj_forward(m, d);
    mj_collision(m, d);

    // Setup Task Space Planner
    std::string coll_body_name = collisionBodyNameArg;
    auto coll_body_info = get_free_body_joint_info(coll_body_name, m);
    tsp::TaskSpacePlanner path_planner(m, coll_body_name);

    Point end_derivative;
    end_derivative << 0, 0, -1, 0;
    Point limits;
    limits << 1, 1, 1, 1.5708;
    double sigma = 0.1;
    int sample_cnt = 20;
    int check_cnt = 20;
    int gd_iterations = 10;
    int ctrl_cnt = 3;

    Point end_pt = Utility::get_body_point<Point>(m, d, coll_body_name);
    end_pt[2] += 0.01;
    Point start_pos;
    start_pos << -0.5, 0.1, 0.1, 1.5708;

    // First Planning Attempt
    for (int i = 0; i < N; ++i) {
        path_candidates = run_planning(path_planner, [&]() {
            return path_planner.plan_with_end_derivatives(start_pos, end_pt, end_derivative,
                                                          sigma, limits, sample_cnt, check_cnt, gd_iterations, ctrl_cnt);
        }, exec_timer, duration_vec_first);
    }
    failed_candidates = path_planner.get_failed_path_candidates();
    auto sampled_via_pts = path_planner.get_sampled_via_pts();
    report_planning_statistics(duration_vec_first, path_candidates, failed_candidates, path_planner, "First planning attempt with start, end and endderivatives");

    // Second Planning Attempt with Via Points Initialization
    path_planner.reset();
    auto via_pts_init = path_planner.get_via_pts();
    for (int i = 0; i < N; ++i) {
        path_candidates = run_planning(path_planner, [&]() {
            return path_planner.plan_with_via_pts(via_pts_init, sigma, limits,
                                                  sample_cnt, check_cnt, gd_iterations, ctrl_cnt);
        }, exec_timer, duration_vec_second);
    }
    failed_candidates = path_planner.get_failed_path_candidates();
    report_planning_statistics(duration_vec_second, path_candidates, failed_candidates, path_planner, "Second planning attempt with via_pts initialization");

    // TEST purpose: Set end point in MuJoCo
    Utility::mj_set_point(end_pt, coll_body_info, d);
    auto via_pts = path_planner.get_via_pts();

    // Initialize GLFW
    if (!glfwInit()) {
        mju_error("Could not initialize GLFW");
    }

    // Create window and OpenGL context
    GLFWwindow* window = glfwCreateWindow(1200, 900, "Get to the Choppa!", nullptr, nullptr);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    // Initialize visualization data structures
    mjv_defaultCamera(&cam);
    mjv_defaultOption(&opt);
    mjv_defaultScene(&scn);
    mjr_defaultContext(&con);

    // Create scene and context
    mjv_makeScene(m, &scn, 4000);
    mjr_makeContext(m, &con, mjFONTSCALE_150);

    // Install GLFW callbacks
    glfwSetKeyCallback(window, keyboard_cb);
    glfwSetCursorPosCallback(window, mouse_move_cb);
    glfwSetMouseButtonCallback(window, mouse_button_cb);
    glfwSetScrollCallback(window, scroll_cb);

    const int pts_cnt = 10;

    // Main loop
    while (!glfwWindowShouldClose(window)) {
        mj_forward(m, d);
        d->time += 1e-2;

        // Get framebuffer viewport
        mjrRect viewport = {0, 0, 0, 0};
        glfwGetFramebufferSize(window, &viewport.width, &viewport.height);

        // Update scene and render
        mjv_updateScene(m, d, &opt, nullptr, &cam, mjCAT_ALL, &scn);

        draw_sphere(&scn, convert_point(via_pts[0]), 0.03, start_color);
        draw_sphere(&scn, convert_point(via_pts[2]), 0.03, end_color);

        if (vis_best_path) {
            auto pts = path_planner.get_path_pts(pts_cnt);
            draw_path(&scn, convert_point_vector(pts), 0.2, path_color);
        }

        // Visualize sampled via points
        visualize_via_pts(vis_sampled_via_pts, sampled_via_pts, path_planner, scn, pts_cnt, sampled_via_path_color, sampled_via_color);

        // Draw path candidates
        visualize_candidates(vis_succ_candidates, vis_grad_descent,
                             path_candidates, path_planner, scn, pts_cnt, path_color, via_color, graddesc_color, graddesc_via_color);
        // Draw failed path candidates
        visualize_candidates(vis_failed_candidates, vis_grad_descent,
                             failed_candidates, path_planner, scn, pts_cnt, failed_path_color, failed_via_color,
                             failed_graddesc_color, failed_graddesc_via_color);

        // Animate block
        if (vis_animate_block) {
            double u = std::fmod(d->time / 10.0, 1.0);
            Point pt = path_planner.evaluate(u);
            Utility::mj_set_point(pt, coll_body_info, d);
        }

        mjr_render(viewport, &scn, &con);

        // Swap OpenGL buffers and poll events
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Free visualization storage
    mjv_freeScene(&scn);
    mjr_freeContext(&con);

    // Free MuJoCo model and data
    mj_deleteData(d);
    mj_deleteModel(m);

    return 0;
}