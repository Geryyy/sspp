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


void print_candidates_statistics(const std::vector<tsp::PathCandidate>& candidates, const std::string& label);


void print_menue();

// keyboard callback
void keyboard(GLFWwindow* window, int key, int scancode, int act, int mods);


// mouse button callback
void mouse_button(GLFWwindow* window, int button, int act, int mods);


// mouse move callback
void mouse_move(GLFWwindow* window, double xpos, double ypos);

// scroll callback
void scroll(GLFWwindow* window, double xoffset, double yoffset);



#endif //UI_H
