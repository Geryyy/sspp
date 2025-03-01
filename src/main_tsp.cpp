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


// mouse interaction
bool button_left = false;
bool button_middle = false;
bool button_right =  false;
double lastx = 0;
double lasty = 0;


bool vis_best_path = true;
bool vis_succ_candidates = false;
bool vis_failed_candidates = false;
bool vis_grad_descent = false;
bool vis_animate_block = false;

std::vector<tsp::PathCandidate> path_candidates;
std::vector<tsp::PathCandidate> failed_candidates;

void print_candidates_statistics(const std::vector<tsp::PathCandidate>& candidates, const std::string& label) {
    std::cout << "number of " << label << " candidates: " << candidates.size() << std::endl;
    for(const auto& candidate : candidates) {
        std::cout << "candidate.gds_steps: " << candidate.gradient_steps.size() << " status: " << SolverStatustoString(candidate.status) << std::endl;
    }
}

void print_menue() {
    std::cout << "backspace\treset" << std::endl;
    std::cout << "Q\tvis_best_path" << std::endl;
    std::cout << "W\tvis_succ_candidates" << std::endl;
    std::cout << "E\tvis_failed_candidates" << std::endl;
    std::cout << "R\tvis_grad_descent" << std::endl;
    std::cout << "A\tanimate block" << std::endl;
}

// keyboard callback
void keyboard(GLFWwindow* window, int key, int scancode, int act, int mods) {
    // backspace: reset simulation
    if (act==GLFW_PRESS && key==GLFW_KEY_BACKSPACE) {
        mj_resetData(m, d);
        mj_forward(m, d);
        std::cout << "reset pressed" << std::endl;
    }

    // visualize best path
    if (act==GLFW_PRESS && key==GLFW_KEY_Q) {
        vis_best_path = !vis_best_path;
        std::cout << "vis_best_path: " << vis_best_path << std::endl;
    }

    // visualize succesul path candidates
    if (act==GLFW_PRESS && key==GLFW_KEY_W) {
        vis_succ_candidates = !vis_succ_candidates;
        std::cout << "vis_succ_candidates: " << vis_succ_candidates << std::endl;
        print_candidates_statistics(path_candidates, "succesful");
    }

    // visualize failed path candidates
    if (act==GLFW_PRESS && key==GLFW_KEY_E) {
        vis_failed_candidates = !vis_failed_candidates;
        std::cout << "vis_failed_candidates: " << vis_failed_candidates << std::endl;
        print_candidates_statistics(failed_candidates, "failed");
    }

    // visualize gradient descent steps
    if (act==GLFW_PRESS && key==GLFW_KEY_R) {
        vis_grad_descent = !vis_grad_descent;
        std::cout << "vis_grad_descent: " << vis_grad_descent << std::endl;
    }

    // visualize animation of block along path
    if (act==GLFW_PRESS && key==GLFW_KEY_A) {
        vis_animate_block = !vis_animate_block;
        std::cout << "vis_animate_block: " << vis_animate_block << std::endl;
    }

}


// mouse button callback
void mouse_button(GLFWwindow* window, int button, int act, int mods) {
    // update button state
    button_left = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT)==GLFW_PRESS);
    button_middle = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE)==GLFW_PRESS);
    button_right = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT)==GLFW_PRESS);

    // update mouse position
    glfwGetCursorPos(window, &lastx, &lasty);
}


// mouse move callback
void mouse_move(GLFWwindow* window, double xpos, double ypos) {
    // no buttons down: nothing to do
    if (!button_left && !button_middle && !button_right) {
        return;
    }

    // compute mouse displacement, save
    double dx = xpos - lastx;
    double dy = ypos - lasty;
    lastx = xpos;
    lasty = ypos;

    // get current window size
    int width, height;
    glfwGetWindowSize(window, &width, &height);

    // get shift key state
    bool mod_shift = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT)==GLFW_PRESS ||
                      glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT)==GLFW_PRESS);

    // determine action based on mouse button
    mjtMouse action;
    if (button_right) {
        action = mod_shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
    } else if (button_left) {
        action = mod_shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
    } else {
        action = mjMOUSE_ZOOM;
    }

    // move camera
    mjv_moveCamera(m, action, dx/height, dy/height, &scn, &cam);
}


// scroll callback
void scroll(GLFWwindow* window, double xoffset, double yoffset) {
    // emulate vertical mouse motion = 5% of window height
    mjv_moveCamera(m, mjMOUSE_ZOOM, 0, -0.05*yoffset, &scn, &cam);
}



void draw_sphere(mjvScene* scn, Eigen::Vector3d pos, float size=0.1, float* rgba = nullptr) {
    Eigen::Vector3d line_size = Eigen::Vector3d::Ones() * size;
    float line_rgba[] = {1,0,1,1};

    if(!rgba) {
        rgba = line_rgba;
    }

    if(scn->ngeom >= scn->maxgeom-1)
        throw std::runtime_error("ngeom > maxgeom!");

    scn->ngeom++;
    mjvGeom *geom_ptr = &scn->geoms[scn->ngeom - 1];
    mjv_initGeom(geom_ptr, mjGEOM_SPHERE, line_size.data(), pos.data(), NULL, rgba);
}


void draw_arrow(mjvScene* scn, Eigen::Vector3d pos, Eigen::Vector3d gradient, float size=0.1, float* rgba = nullptr) {
    if (gradient.norm() < 1e-6) return; // Avoid division by zero for zero gradient

    Eigen::Vector3d line_size;
    line_size << 0.1,0.1,1.0;
    line_size *= size;
    Eigen::Matrix3d rotmat;

    // Normalize gradient to get direction
    Eigen::Vector3d dir = gradient.normalized();
    Eigen::Vector3d z_axis(0, 0, 1); // Default arrow orientation in MuJoCo

    // Compute quaternion rotation from z-axis to gradient direction
    Eigen::Quaterniond q = Eigen::Quaterniond::FromTwoVectors(z_axis, dir);

    // Convert quaternion to rotation matrix
    rotmat = q.toRotationMatrix();

    float line_rgba[] = {1, 0.5, 0.5, 1};
    if (!rgba) {
        rgba = line_rgba;
    }

    if(scn->ngeom >= scn->maxgeom-1)
        throw std::runtime_error("ngeom > maxgeom!");

    scn->ngeom++;
    mjvGeom *geom_ptr = &scn->geoms[scn->ngeom - 1];
    mjv_initGeom(geom_ptr, mjGEOM_ARROW, line_size.data(), pos.data(), rotmat.data(), rgba);
}


void draw_path(mjvScene* scn, std::vector<Eigen::Vector3d> pts, float width=0.5, float* rgba = nullptr) {
    Eigen::Vector3d line_size = Eigen::Vector3d::Ones()*width;
    float line_rgba[] = {1,1,0,1};
    if (!rgba) {
        rgba = line_rgba;
    }

    for(int i = 1; i < pts.size(); i++){
        Eigen::Vector3d start_pos = pts[i-1];
        Eigen::Vector3d end_pos = pts[i];

        if(scn->ngeom >= scn->maxgeom-1)
            throw std::runtime_error("ngeom > maxgeom!");

        scn->ngeom++;
        mjvGeom *geom_ptr = &scn->geoms[scn->ngeom - 1];
        mjv_initGeom(geom_ptr, mjGEOM_SPHERE, line_size.data(), start_pos.data(), NULL, rgba);
        mjv_connector(geom_ptr, mjGEOM_LINE, width, start_pos.data(), end_pos.data());
    }

}



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
    exec_timer.tic();
    init_spline = path_planner.initializePath(Point::Zero(), Point::Ones(), end_derivative, 3);
    auto duration = exec_timer.toc();
    std::cout << "duration [us]: " << duration/1e3 << std::endl;
    Point limits;
    limits << 1,1,1;
    double sigma = 0.1;
    int sample_cnt = 3;
    int check_cnt = 100;
    int gd_iterations = 20;
    int ctrl_cnt = 3; // THIS MUST BE CONSTANT: start, via, end!!
    Point end_pos = block2_pos;
    end_pos[2] += 0.01;
    Point start_pos;
    start_pos << 0.5,0.5,0.5;
    // start_pos = block1_pos;
//    start_pos[2] += 0.5;
//    start_pos << 1,1,1;
    path_candidates = path_planner.plan(start_pos,
        end_pos, end_derivative, sigma, limits, sample_cnt, check_cnt, gd_iterations, ctrl_cnt);

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
    float via_color[] = {0,1,1,1};
    float start_color[] = {0,1,0,1};
    float end_color[] = {1,0,0,1};
    float path_color[] = {0,1,1,1};
    float graddesc_color[] = {0,0.7,0.5,0.5};
    float graddesc_via_color[] = {0,1,1,0.5};

    float failed_via_color[] = {1.0, 0.5, 0.5, 0.5};       // More red, more transparent
    float failed_path_color[] = {1.0, 0.5, 0.5, 0.5};      // Same as via for consistency
    float failed_graddesc_color[] = {0.8, 0.3, 0.3, 0.3};  // Darker red, more transparent
    float failed_graddesc_via_color[] = {1.0, 0.4, 0.4, 0.3}; // Slightly more red than graddesc


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

        // draw successful path candidates
        if(vis_succ_candidates) {
            for(const auto& candidate : path_candidates) {
                Point via_pt = candidate.via_point;
                tsp::Spline spline = path_planner.path_from_via_pt(via_pt);
                auto pts = path_planner.get_path_pts(spline, pts_cnt);
                draw_path(&scn, pts, 0.2, path_color);
                draw_sphere(&scn, via_pt, 0.03, via_color);

                if(vis_grad_descent){
                    for(const auto& step : candidate.gradient_steps) {
                        Point via_pt = step.via_point;
                        Point coll_grad = step.gradient;
                        tsp::Spline spline = path_planner.path_from_via_pt(via_pt);
                        auto pts = path_planner.get_path_pts(spline, pts_cnt);
                        draw_path(&scn, pts, 0.1, graddesc_color);
                        draw_sphere(&scn, via_pt, 0.03, graddesc_via_color);
                        draw_arrow(&scn, via_pt, coll_grad, 0.2);
                    }
                }
            }
        }

        // draw failed path candidates
        // draw path candidates
        if(vis_failed_candidates) {
            for(const auto& candidate : failed_candidates) {
                Point via_pt = candidate.via_point;
                tsp::Spline spline = path_planner.path_from_via_pt(via_pt);
                auto pts = path_planner.get_path_pts(spline, pts_cnt);
                draw_path(&scn, pts, 0.2, failed_path_color);
                draw_sphere(&scn, via_pt, 0.03, failed_via_color);

                if(vis_grad_descent){
                    for(const auto& step : candidate.gradient_steps) {
                        Point via_pt = step.via_point;
                        Point coll_grad = step.gradient;
                        tsp::Spline spline = path_planner.path_from_via_pt(via_pt);
                        auto pts = path_planner.get_path_pts(spline, pts_cnt);
                        draw_path(&scn, pts, 0.1, failed_graddesc_color);
                        draw_sphere(&scn, via_pt, 0.03, failed_graddesc_via_color);
                        draw_arrow(&scn, via_pt, coll_grad, 0.2);
                    }
                }
            }
        }

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