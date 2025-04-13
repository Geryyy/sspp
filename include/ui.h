//
// Created by gebmer on 02.03.25.
//

#ifndef UI_H
#define UI_H

#include"tsp.h"
#include <GLFW/glfw3.h>
#include "mujoco/mujoco.h"


// mouse interaction
extern bool button_left;
extern bool button_middle;
extern bool button_right;
extern double lastx;
extern double lasty;

extern bool vis_best_path;
extern bool vis_succ_candidates;
extern bool vis_failed_candidates;
extern bool vis_grad_descent;
extern bool vis_animate_block;
extern bool vis_sampled_via_pts;


void print_candidates_statistics(const std::vector<tsp::PathCandidate>& candidates, const std::string& label);


void print_menue();

// keyboard callback
void keyboard(GLFWwindow* window, int key, int scancode, int act, int mods, 
    mjModel* m, mjData* d, 
    bool& vis_best_path, bool& vis_succ_candidates,
    bool& vis_failed_candidates, bool& vis_grad_descent, bool& vis_animate_block,
    bool& vis_sampled_via_pts,
    std::vector<tsp::PathCandidate>& path_candidates,
    std::vector<tsp::PathCandidate>& failed_candidates);


// mouse button callback
void mouse_button(GLFWwindow* window, int button, int act, int mods,
    double& lastx, double& lasty, bool& button_left, bool& button_middle, bool& button_right);


// mouse move callback
void mouse_move(GLFWwindow* window, double xpos, double ypos,
    mjModel* m, mjvScene& scn, mjvCamera& cam,
    double& lastx, double& lasty,
    bool& button_left, bool& button_middle, bool& button_right);

// scroll callback
void scroll(GLFWwindow* window, double xoffset, double yoffset, 
    mjModel* m, mjvScene& scn, mjvCamera& cam);

// GLFW callbacks for keyboard and mouse events
void keyboard_cb(GLFWwindow* window, int key, int scancode, int act, int mods);

void mouse_button_cb(GLFWwindow* window, int button, int act, int mods);

void mouse_move_cb(GLFWwindow* window, double xpos, double ypos);

void scroll_cb(GLFWwindow* window, double xoffset, double yoffset);


#endif //UI_H
