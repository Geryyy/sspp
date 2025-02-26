#include <iostream>
#include <string>
#include <mujoco/mujoco.h>
#include <Eigen/Core>
#include "Timer.h"
#include "tsp.h"
#include "utility.h"
#include <cstdio>
#include <cstring>

#include <GLFW/glfw3.h>
#include <mujoco/mujoco.h>

// Path to the XML file for the MuJoCo model
const std::string modelFile = "/home/geraldebmer/repos/robocrane/sspp/mjcf/planner.xml";

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


// keyboard callback
void keyboard(GLFWwindow* window, int key, int scancode, int act, int mods) {
    // backspace: reset simulation
    if (act==GLFW_PRESS && key==GLFW_KEY_BACKSPACE) {
        mj_resetData(m, d);
        mj_forward(m, d);
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



void draw_line(mjvScene* scn, Eigen::Vector3d start_pos, Eigen::Vector3d end_pos, int line_width=5) {
    Eigen::Vector3d line_size;
    Eigen::Vector3d line_pos;
    Eigen::Matrix3d line_rotmat;
    Eigen::Vector4f line_rgba;
    line_size << 10, 10., 10.;
    line_pos << 0.5, 0.2, 0.2;
    line_rotmat << 1, 0, 0, 0, 1, 0, 0, 0, 1;
    line_rgba << 1, 1, 0, 1;

    scn->ngeom++;
    mjvGeom *geom_ptr = &scn->geoms[scn->ngeom - 1];
    mjv_initGeom(geom_ptr, mjGEOM_SPHERE, line_size.data(), line_pos.data(), NULL, line_rgba.data());
    mjv_connector(geom_ptr, mjGEOM_LINE, line_width, start_pos.data(), end_pos.data());
}


void draw_path(mjvScene* scn, std::vector<Eigen::Vector3d> pts, int line_width=5) {
    Eigen::Vector3d line_size;
    Eigen::Vector3d line_pos;
//    Eigen::Matrix3d line_rotmat;
    Eigen::Vector4f line_rgba;
    line_size << 10, 10., 10.;
    line_pos << 0.5, 0.2, 0.2;
//    line_rotmat << 1, 0, 0, 0, 1, 0, 0, 0, 1;
    line_rgba << 1, 1, 0, 1;

    for(int i = 1; i < pts.size(); i++){
        Eigen::Vector3d start_pos = pts[i-1];
        Eigen::Vector3d end_pos = pts[i];

        if(scn->ngeom >= scn->maxgeom-1)
            break;

        scn->ngeom++;
        mjvGeom *geom_ptr = &scn->geoms[scn->ngeom - 1];
        mjv_initGeom(geom_ptr, mjGEOM_SPHERE, line_size.data(), start_pos.data(), NULL, line_rgba.data());
        mjv_connector(geom_ptr, mjGEOM_LINE, line_width, start_pos.data(), end_pos.data());
//        std::cout << "start_pos("<<i<<"): " << start_pos.transpose() << std::endl;
//        std::cout << "end_pos("<<i<<"): " << end_pos.transpose() << std::endl;
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

    m = mj_loadXML(modelFile.c_str(), NULL, NULL, 0);
    d = mj_makeData(m);

    std::cout << "DoFs: " << m->nq << std::endl;


    mj_forward(m, d);
    mj_collision(m,d);
//
//    // get the number of contacts
//    std::cout << "Number of contacts: " << d->ncon << std::endl;
//
//    // iterate over all contacts
//    for(int i=0; i<d->ncon; i++) {
//        std::cout << "Contact " << i << ": " << d->contact[i].dist << std::endl;
//    }

    auto block1_pos = get_body_position(m, d, "block1");
    auto block2_pos = get_body_position(m, d, "block2");
    std::cout << "block1_pos: " << block1_pos.transpose() << std::endl;
    std::cout << "block2_pos: " << block2_pos.transpose() << std::endl;

    // static sampling path planner
    std::cout << "Taskspace Planner" << std::endl;
    using TSP = tsp::TaskSpacePlanner;
    TSP path_planner(m, d);
    using Point = TSP::Point;
    TSP::Spline init_spline;
    Point end_derivative;
    end_derivative << 0,0,-1;
    auto err_code = path_planner.initializePath(Point::Zero(), Point::Ones(), end_derivative, init_spline, 10);
    std::cout << "Error code: " << err_code << std::endl;
    Point limits;
    limits << 1,1,1;
    double sigma = 0.2;
    int sample_cnt = 10;
    int check_cnt = 10;
    int ctrl_cnt = 5;
    Point end_pos = block2_pos;
    end_pos[2] += 0.3;
    Point start_pos;
    start_pos = block1_pos;
//    start_pos[2] += 0.1;
//    start_pos << 1,1,1;
    auto ret = path_planner.plan(start_pos, end_pos, end_derivative, sigma, limits, sample_cnt, check_cnt, ctrl_cnt);
    std::cout << "ret: " << ret << std::endl;
//    return 0;




    // init GLFW
    if (!glfwInit()) {
        mju_error("Could not initialize GLFW");
    }

    // create window, make OpenGL context current, request v-sync
    GLFWwindow* window = glfwCreateWindow(1200, 900, "Demo", NULL, NULL);
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

    const int pts_cnt = 50;
    std::vector<Point> pts;
    for(int i = 0; i < pts_cnt; i++){
        double u = static_cast<double>(i)/pts_cnt;
        auto pt = path_planner.evaluate(u);
        pts.push_back(pt);
//        std::cout << "pt("<<u<<") " << pt.transpose() << std::endl;
    }

    // run main loop, target real-time simulation and 60 fps rendering
    while (!glfwWindowShouldClose(window)) {
        // advance interactive simulation for 1/60 sec
        //  Assuming MuJoCo can simulate faster than real-time, which it usually can,
        //  this loop will finish on time for the next frame to be rendered at 60 fps.
        //  Otherwise add a cpu timer and exit this loop when it is time to render.
        mjtNum simstart = d->time;
        while (d->time - simstart < 1.0/60.0) {
            mj_step(m, d);
//            d->qpos[0] = start_pos[0];
//            d->qpos[1] = start_pos[1];
//            d->qpos[2] = start_pos[2];
//
//            d->qpos[7+0] = end_pos[0];
//            d->qpos[7+1] = end_pos[1];
//            d->qpos[7+2] = end_pos[2];
        }

        // get framebuffer viewport
        mjrRect viewport = {0, 0, 0, 0};
        glfwGetFramebufferSize(window, &viewport.width, &viewport.height);

        // update scene and render
        mjv_updateScene(m, d, &opt, NULL, &cam, mjCAT_ALL, &scn);

//      draw_line(&scn, block1_pos, block2_pos);
        draw_path(&scn, pts);
        mjr_render(viewport, &scn, &con);
//        return 0;


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










    // exec_timer.tic();
    // auto success = path_planner.plan(Point::Zero(), Point::Ones(), 0.5, Point::Ones());
    // std::cout << "Planning time: " << static_cast<double>(exec_timer.toc())*1e-3 << " us" << std::endl;
    // std::cout << "Path found: " << success << std::endl;

    constexpr int num_points = 10;
    Eigen::VectorXd param(num_points);
    Eigen::MatrixXd data(3,num_points);

     for(int i = 0; i < num_points; i++) {
         double u = static_cast<double>(i)/9.;
        Point p = path_planner.evaluate(init_spline, u);
      std::cout << "Point " << i << ": " << p.transpose() << std::endl;
        param[i] = u;
        data.block<3,1>(0,i) = p;
     }

    exportToCSV("tsp.csv", param, data);

    return 0;
}