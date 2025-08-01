//
// Created by gebmer on 02.03.25.
//

#include"tsp.h"
#include <GLFW/glfw3.h>
#include "mujoco/mujoco.h"

extern mjModel* m;
extern mjData* d;
extern mjvCamera cam;                      // abstract camera
extern mjvOption opt;                      // visualization options
extern mjvScene scn;                       // abstract scene
extern mjrContext con;                     // custom GPU context

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
bool vis_sampled_via_pts = false;
bool flag_plan_path = false;

extern std::vector<tsp::PathCandidate> path_candidates;
extern std::vector<tsp::PathCandidate> failed_candidates;

void print_candidates_statistics(const std::vector<tsp::PathCandidate>& candidates, const std::string& label) {
    std::cout << "number of " << label << " candidates: " << candidates.size() << std::endl;
    for(const auto& candidate : candidates) {
        std::cout << "candidate.gds_steps: " << candidate.gradient_steps.size()
            << " status: " << SolverStatustoString(candidate.status)
            << std::endl;
    }
}

void print_menue() {
    std::cout << "backspace\treset" << std::endl;
    std::cout << "Q\tvis_best_path" << std::endl;
    std::cout << "W\tvis_succ_candidates" << std::endl;
    std::cout << "E\tvis_failed_candidates" << std::endl;
    std::cout << "R\tvis_grad_descent" << std::endl;
    std::cout << "A\tanimate block" << std::endl;
    std::cout << "T\tvis_sampled_via_points" << std::endl;
    std::cout << "P\tflag_plan_path" << std::endl;
}

// keyboard callback
void keyboard(GLFWwindow* window, int key, int scancode, int act, int mods, 
    mjModel* m, mjData* d, 
    bool& vis_best_path, bool& vis_succ_candidates,
    bool& vis_failed_candidates, bool& vis_grad_descent, bool& vis_animate_block,
    bool& vis_sampled_via_pts, bool& flag_plan_path,
    std::vector<tsp::PathCandidate>& path_candidates,
    std::vector<tsp::PathCandidate>& failed_candidates) {
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

    // visualize sampled via points
    if (act==GLFW_PRESS && key==GLFW_KEY_T) {
        vis_sampled_via_pts = !vis_sampled_via_pts;
        std::cout << "vis_sampled_via_pts: " << vis_sampled_via_pts << std::endl;
    }

    // visualize sampled via points
    if (act==GLFW_PRESS && key==GLFW_KEY_P) {
        flag_plan_path = true;
        std::cout << "plan new path" << std::endl;
    }

}


// mouse button callback
void mouse_button(GLFWwindow* window, int button, int act, int mods,
    double& lastx, double& lasty, bool& button_left, bool& button_middle, bool& button_right) {
    // update button state
    button_left = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT)==GLFW_PRESS);
    button_middle = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE)==GLFW_PRESS);
    button_right = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT)==GLFW_PRESS);

    // update mouse position
    glfwGetCursorPos(window, &lastx, &lasty);
}


// mouse move callback
void mouse_move(GLFWwindow* window, double xpos, double ypos,
    mjModel* m, mjvScene& scn, mjvCamera& cam,
    double& lastx, double& lasty,
    bool& button_left, bool& button_middle, bool& button_right) {

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
void scroll(GLFWwindow* window, double xoffset, double yoffset, 
    mjModel* m, mjvScene& scn, mjvCamera& cam) {
    // emulate vertical mouse motion = 5% of window height
    mjv_moveCamera(m, mjMOUSE_ZOOM, 0, -0.05*yoffset, &scn, &cam);
}


// -----
// GLFW callbacks for keyboard and mouse events
// -----
//

void keyboard_cb(GLFWwindow* window, int key, int scancode, int act, int mods){
    keyboard(window, key, scancode, act, mods, m, d,
        vis_best_path, vis_succ_candidates,
        vis_failed_candidates, vis_grad_descent, vis_animate_block,
        vis_sampled_via_pts, flag_plan_path,
        path_candidates, failed_candidates);
}

void mouse_button_cb(GLFWwindow* window, int button, int act, int mods){
    mouse_button(window, button, act, mods, lastx, lasty, button_left, button_middle, button_right);
}

void mouse_move_cb(GLFWwindow* window, double xpos, double ypos){
    mouse_move(window, xpos, ypos, m, scn, cam, lastx, lasty, button_left, button_middle, button_right);
}

void scroll_cb(GLFWwindow* window, double xoffset, double yoffset){
    scroll(window, xoffset, yoffset, m, scn, cam);
}