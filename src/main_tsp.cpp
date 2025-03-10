#include <iostream>
#include <string>
#include <mujoco/mujoco.h>
#include <Eigen/Core>
#include "Timer.h"
#include "tsp.h"
#include "utility.h"
#include <cstdio>
#include <cstring>
#include <thread>
#include <GLFW/glfw3.h>
#include <mujoco/mujoco.h>
#include "visu.h"
#include "ui.h"
#include <omp.h>

using namespace visu;

// Path to the XML file for the MuJoCo model
// const std::string modelFile = "/home/geraldebmer/repos/robocrane/sspp/mjcf/planner.xml";
 // const std::string modelFile = "/home/geraldebmer/repos/robocrane/sspp/mjcf/stacking.xml";
// const std::string modelFile = "/home/gebmer/repos/sspp/mjcf/planner.xml";
const std::string modelFile = "/home/gebmer/repos/sspp/mjcf/stacking.xml";

// MuJoCo data structures
mjModel* m = NULL;                  // MuJoCo model
mjData* d = NULL;                   // MuJoCo data
mjvCamera cam;                      // abstract camera
mjvOption opt;                      // visualization options
mjvScene scn;                       // abstract scene
mjrContext con;                     // custom GPU context


std::vector<tsp::PathCandidate> path_candidates;
std::vector<tsp::PathCandidate> failed_candidates;


int main(int argc, char** argv) {
    Timer exec_timer;
    // Print MuJoCo version
    std::cout << "MuJoCo version: " << mj_version() << std::endl;
    // Print Eigen version
    std::cout << "Eigen version: " << EIGEN_WORLD_VERSION << "."
              << EIGEN_MAJOR_VERSION << "."
              << EIGEN_MINOR_VERSION << std::endl;
    char error_buffer_[1000]; // Buffer for storing MuJoCo error messages
    m = mj_loadXML(modelFile.c_str(), nullptr, error_buffer_, sizeof(error_buffer_));
    if (!m)
    {
        throw std::runtime_error("Failed to load MuJoCo model from XML: " + std::string(error_buffer_));
    }

    d = mj_makeData(m);

    // Get the maximum number of threads available
    int max_threads = omp_get_max_threads();

    // Get the number of processors available
    int num_procs = omp_get_num_procs();

    // Get the current number of threads in use
    int num_threads;
#pragma omp parallel
    {
#pragma omp single
        num_threads = omp_get_num_threads();
    }

    std::cout << "Maximum threads available: " << max_threads << std::endl;
    std::cout << "Number of processors available: " << num_procs << std::endl;
    std::cout << "Current number of threads in use: " << num_threads << std::endl;


    std::cout << "Taskspace Planner" << std::endl;
    std::cout << "DoFs: " << m->nq << std::endl;
    // std::cout << "Jacobian sparse: " << mj_isSparse(m) << std::endl;
    // std::cout << "jacobian dimension: " << d->nJ << std::endl;

    print_menue();

    mj_forward(m, d);
    mj_collision(m,d);

    auto block1_pos = get_body_position(m, d, "block1");
    auto block2_pos = get_body_position(m, d, "block2");
    // std::cout << "block1_pos: " << block1_pos.transpose() << std::endl;
    // std::cout << "block2_pos: " << block2_pos.transpose() << std::endl;
    auto body2_id = get_body_id(m, "block2");
    auto block2_yaw = get_body_yaw(body2_id, d);
    std::cout << "block2 yaw: " << block2_yaw << std::endl;
    // static sampling path planner

    using TSP = tsp::TaskSpacePlanner;
    TSP path_planner(m);
    using Point = tsp::Point;
    tsp::Spline init_spline;
    Point end_derivative;
    end_derivative << 0,0,-1, 0;

    init_spline = path_planner.initializePath(Point::Zero(), Point::Ones(), end_derivative, 3);

    Point limits;
    limits << 1,1,1, 1.5708;
    double sigma = 0.1;
    int sample_cnt = 20;
    int check_cnt = 20;
    int gd_iterations = 10;
    int ctrl_cnt = 3; // THIS MUST BE CONSTANT: start, via, end!!
    Point end_pos;
    end_pos.block<3,1>(0,0) = block2_pos;
    end_pos[3] = block2_yaw;
    end_pos[2] += 0.01;
    Point start_pos;
    start_pos << 0.5,0.5,0.5, 1.5708;
    // start_pos = block1_pos;
//    start_pos[2] += 0.5;
//    start_pos << 1,1,1;


    std::vector<double> duration_vec;
    constexpr int N = 100;
    for(int i = 0; i < N; i++) {
        path_planner.reset();
        exec_timer.tic();

        path_candidates = path_planner.plan(start_pos,
            end_pos, end_derivative, sigma, limits, sample_cnt, check_cnt, gd_iterations, ctrl_cnt);

        auto duration = exec_timer.toc();
        duration_vec.push_back(static_cast<double>(duration/1e6));
    }

    // Compute mean
    double sum = std::accumulate(duration_vec.begin(), duration_vec.end(), 0.0);
    double mean = sum / duration_vec.size();

    // Compute standard deviation
    double sq_sum = std::inner_product(duration_vec.begin(), duration_vec.end(), duration_vec.begin(), 0.0);
    double stdev = std::sqrt(sq_sum / duration_vec.size() - mean * mean);

    // Compute min and max
    auto min_it = std::min_element(duration_vec.begin(), duration_vec.end());
    auto max_it = std::max_element(duration_vec.begin(), duration_vec.end());
    int64_t min_val = (min_it != duration_vec.end()) ? *min_it : 0;
    int64_t max_val = (max_it != duration_vec.end()) ? *max_it : 0;

    // Print results
    std::cout << "Mean: " << mean << " ms" << std::endl;
    std::cout << "Standard Deviation: " << stdev << " ms" << std::endl;
    std::cout << "Min: " << min_val << " ms" << std::endl;
    std::cout << "Max: " << max_val << " ms" << std::endl;


    // std::cerr << "duration [ms]: " << duration/1e6 << std::endl;

    failed_candidates = path_planner.get_failed_path_candidates();
    auto sampled_via_pts = path_planner.get_sampled_via_pts();

    std::cout << "nr of succesful path candidates: " << path_candidates.size() << std::endl;
//    for(int i = 0; i < path_candidates.size(); i++) {
//        std::cout << "candidate " << i << " gd steps: " << path_candidates[i].gradient_steps.size() << std::endl;
//    }
    std::cout << "nr of failed path candidates: " << path_planner.get_failed_path_candidates().size() << std::endl;

    return 0;
    // TEST purpose
    for(int i = 0; i <3; i++) {
        d->qpos[i] = end_pos[i];
    }

    auto via_pts = path_planner.get_via_pts();



    // init GLFW
    if (!glfwInit()) {
        mju_error("Could not initialize GLFW");
    }

    // create window, make OpenGL context current, request v-sync
    GLFWwindow* window = glfwCreateWindow(1200, 900, "Get to the Choppa!", NULL, NULL);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    // initialize visualization data structures
    mjv_defaultCamera(&cam);
    mjv_defaultOption(&opt);
    mjv_defaultScene(&scn);
    mjr_defaultContext(&con);

    // create scene and context
    mjv_makeScene(m, &scn, 4000);
    mjr_makeContext(m, &con, mjFONTSCALE_150);

    // install GLFW mouse and keyboard callbacks
    glfwSetKeyCallback(window, keyboard);
    glfwSetCursorPosCallback(window, mouse_move);
    glfwSetMouseButtonCallback(window, mouse_button);
    glfwSetScrollCallback(window, scroll);

    const int pts_cnt = 10;

    // run main loop, target real-time simulation and 60 fps rendering
    while (!glfwWindowShouldClose(window)) {
        mjtNum simstart = d->time;
        // while (d->time - simstart < 1.0/60.0) {
        //     mj_step(m, d);
        // }
        mj_forward(m, d);
        // std::this_thread::sleep_for(std::chrono::milliseconds(1));
        d->time += 1e-2;


        // get framebuffer viewport
        mjrRect viewport = {0, 0, 0, 0};
        glfwGetFramebufferSize(window, &viewport.width, &viewport.height);

        // update scene and render
        mjv_updateScene(m, d, &opt, NULL, &cam, mjCAT_ALL, &scn);

        draw_sphere(&scn,convert_point(via_pts[0]),0.03, start_color);
        draw_sphere(&scn,convert_point(via_pts[2]),0.03, end_color);

        if(vis_best_path) {
            auto pts = path_planner.get_path_pts(pts_cnt);
            draw_path(&scn, convert_point_vector(pts), 0.2, path_color);
            // draw_sphere(&scn, via_pt, 0.03, via_color);
        }

        // vis_sampled_via_pts
        visualize_via_pts(vis_sampled_via_pts, sampled_via_pts, path_planner, scn, pts_cnt, sampled_via_path_color, sampled_via_color);

        // draw path candidates
        visualize_candidates(vis_succ_candidates, vis_grad_descent,
            path_candidates, path_planner, scn, pts_cnt, path_color, via_color, graddesc_color, graddesc_via_color);
        // draw failed path candidates
        visualize_candidates(vis_failed_candidates, vis_grad_descent,
            failed_candidates, path_planner, scn, pts_cnt, failed_path_color, failed_via_color,
            failed_graddesc_color, failed_graddesc_via_color);

        /* animate block */
        if(vis_animate_block) {
            double u = std::fmod(d->time/10., 1.0);
            Point qpos = path_planner.evaluate(u);
            for (int j = 0; j < 3; ++j)
            {
                d->qpos[j] = qpos[j];
            }
            auto quat = yaw_to_quat(qpos[3], d);
            d->qpos[3] = quat.w();
            d->qpos[4] = quat.x();
            d->qpos[5] = quat.y();
            d->qpos[6] = quat.z();

            // set_body_yaw(body2_id, qpos[3], d);
            // std::cout << "eval(u): " << qpos[3] << std::endl;
            // std::cout << "eval(u): " << quat << std::endl;
        }

        mjr_render(viewport, &scn, &con);

        // swap OpenGL buffers (blocking call due to v-sync)
        glfwSwapBuffers(window);

        // process pending GUI events, call GLFW callbacks
        glfwPollEvents();
    }

    //free visualization storage
    mjv_freeScene(&scn);
    mjr_freeContext(&con);

    // free MuJoCo model and data
    mj_deleteData(d);
    mj_deleteModel(m);
    return 0;

}