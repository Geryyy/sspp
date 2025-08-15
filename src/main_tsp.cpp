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
std::vector<tsp::Point> via_pts, sampled_via_pts;

// Planning state tracking
bool first_plan_call = true;

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

    // Print current distribution state
    Point current_mean = path_planner.get_current_mean();
    Point current_stddev = path_planner.get_current_stddev();
    std::cout << "Current mean: [" << current_mean.transpose() << "]" << std::endl;
    std::cout << "Current stddev: [" << current_stddev.transpose() << "]" << std::endl;
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

void execute_planning_cycle(tsp::TaskSpacePlanner& path_planner,
                            const Point& start_pt,
                            const Point& end_pt,
                            const Utility::BodyJointInfo& coll_body_info,
                            Timer& exec_timer,
                            std::vector<double>& duration_vec,
                            const std::string& report_name = "Planning attempt") {

    // Use iterative flag for subsequent calls
    bool iterate_flag = !first_plan_call;

    path_candidates = run_planning(path_planner, [&]() {
        return path_planner.plan(start_pt, end_pt, iterate_flag);
    }, exec_timer, duration_vec);

    if (first_plan_call) {
        first_plan_call = false;
    }

    failed_candidates = path_planner.get_failed_path_candidates();
    report_planning_statistics(duration_vec, path_candidates, failed_candidates, path_planner, report_name);

    auto ctrl_pts = path_planner.get_ctrl_pts();
    std::cout << "Spline control points:\n" << ctrl_pts << std::endl;

    sampled_via_pts = path_planner.get_sampled_via_pts();
    via_pts = path_planner.get_via_pts();

    // TEST purpose: Set end point in MuJoCo
    Utility::mj_set_point(end_pt, coll_body_info, d);
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

    std::cout << "Taskspace Planner" << std::endl;
    std::cout << "DoFs: " << m->nq << std::endl;

    print_menue();

    mj_forward(m, d);
    mj_collision(m, d);

    // Setup Task Space Planner with configuration parameters
    std::string coll_body_name = collisionBodyNameArg;
    auto coll_body_info = get_free_body_joint_info(coll_body_name, m);

    // Configure planner parameters
    double stddev_initial = 0.3;    // Initial sampling spread
    double stddev_min = 0.001;       // Minimum stddev (convergence limit)
    double stddev_max = 0.5;        // Maximum stddev (exploration limit)
    double stddev_increase_factor = 1.5;  // Increase factor when no success
    double stddev_decay_factor = 0.99;    // Decay factor when successful
    double elite_fraction = 0.3;    // Top 30% candidates used for distribution update
    int sample_count = 20;          // Number of via point candidates per iteration
    int check_points = 50;          // Points checked along spline for collision
    int gd_iterations = 10;         // Gradient descent iterations
    int init_points = 3;            // Number of initial via points
    double collision_weight = 1.0;  // Weight for collision cost in path evaluation
    double z_min = 0.1;             // Minimum z-coordinate (ground level)

    Point limit_max, limit_min;
    limit_max << 0.7,0.7,0.6, 1.6; // Max limits for x, y, z, yaw
    limit_min << 0,-0.7,0.1, -1.6;

    tsp::TaskSpacePlanner path_planner(m, coll_body_name,
                                       stddev_initial, stddev_min, stddev_max,
                                       stddev_increase_factor, stddev_decay_factor,
                                       elite_fraction, sample_count, check_points,
                                       gd_iterations, init_points, collision_weight, z_min,
                                       limit_min, limit_max, true);

    std::string start_body_name = "block_green/";
    std::string end_body_name = "block_orange/";
    Point end_pt = Utility::get_body_point<Point>(m, d, end_body_name);
    // Point end_pt = Utility::get_body_point<Point>(m, d, coll_body_name);
    // end_pt[2] += 0.8;
    Point start_pt;
    // start_pt << -0.5, 0.7, 0.8, 1.5708;
    start_pt = Utility::get_body_point<Point>(m, d, start_body_name);

    end_pt[2] += 0.02;
    start_pt[2] += 0.02;

    // Initial Planning Attempt
    execute_planning_cycle(path_planner, start_pt, end_pt,
                           coll_body_info, exec_timer, duration_vec_second,
                           "Initial planning (new path)");

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

        if (flag_plan_path){
            std::cout << "debug out: iterative planning refinement" << std::endl;

            execute_planning_cycle(path_planner, start_pt, end_pt,
                                   coll_body_info, exec_timer, duration_vec_second,
                                   "Iterative refinement");

            flag_plan_path = false;
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