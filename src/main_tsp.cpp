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

// Path to the XML file for the MuJoCo model
// const std::string modelFile = "/home/geraldebmer/repos/robocrane/sspp/mjcf/planner.xml";
// const std::string modelFile = "/home/gebmer/repos/sspp/mjcf/planner.xml";
const std::string modelFile = "/home/gebmer/repos/sspp/mjcf/stacking.xml";

// MuJoCo data structures
mjModel* m = NULL;                  // MuJoCo model
mjData* d = NULL;                   // MuJoCo data
mjvCamera cam;                      // abstract camera
mjvOption opt;                      // visualization options
mjvScene scn;                       // abstract scene
mjrContext con;                     // custom GPU context


Eigen::Vector3d get_body_position(mjModel* m, mjData* d, std::string name){
    auto block_id = mj_name2id(m, mjtObj::mjOBJ_BODY, name.c_str());
//    std::cout << block_name << " id: " << block_id << std::endl;
    Eigen::Vector3d body_pos;
    body_pos << d->xpos[block_id*3], d->xpos[block_id*3+1], d->xpos[block_id*3+2];
    return body_pos;
}


std::vector<tsp::PathCandidate> path_candidates;
std::vector<tsp::PathCandidate> failed_candidates;


int main(int argc, char** argv) {
    Timer exec_timer;
    std::cout << "Mujoco Collission Checker" << std::endl;
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

    // static sampling path planner

    using TSP = tsp::TaskSpacePlanner;
    TSP path_planner(m);
    using Point = tsp::Point;
    tsp::Spline init_spline;
    Point end_derivative;
    end_derivative << 0,0,-1;

    init_spline = path_planner.initializePath(Point::Zero(), Point::Ones(), end_derivative, 3);

    Point limits;
    limits << 1,1,1;
    double sigma = 0.1;
    int sample_cnt = 30;
    int check_cnt = 10;
    int gd_iterations = 10;
    int ctrl_cnt = 3; // THIS MUST BE CONSTANT: start, via, end!!
    Point end_pos = block2_pos;
    end_pos[2] += 0.01;
    Point start_pos;
    start_pos << 0.5,0.5,0.5;
    // start_pos = block1_pos;
//    start_pos[2] += 0.5;
//    start_pos << 1,1,1;

    exec_timer.tic();

    path_candidates = path_planner.plan(start_pos,
        end_pos, end_derivative, sigma, limits, sample_cnt, check_cnt, gd_iterations, ctrl_cnt);

    auto duration = exec_timer.toc();
    std::cerr << "duration [ms]: " << duration/1e6 << std::endl;

    failed_candidates = path_planner.get_failed_path_candidates();

    std::cout << "nr of succesful path candidates: " << path_candidates.size() << std::endl;
    for(int i = 0; i < path_candidates.size(); i++) {
        std::cout << "candidate " << i << " gd steps: " << path_candidates[i].gradient_steps.size() << std::endl;
    }
    std::cout << "nr of failed path candidates: " << path_planner.get_failed_path_candidates().size() << std::endl;

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
    mjv_makeScene(m, &scn, 2000);
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

        draw_sphere(&scn,via_pts[0],0.03, start_color);
        draw_sphere(&scn,via_pts[2],0.03, end_color);

        if(vis_best_path) {
            auto pts = path_planner.get_path_pts(pts_cnt);
            draw_path(&scn, pts, 0.2, path_color);
            // draw_sphere(&scn, via_pt, 0.03, via_color);
        }

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