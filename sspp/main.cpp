#include <iostream>
#include <string>
#include <mujoco/mujoco.h>
#include <Eigen/Core>
#include "Timer.h"
#include "sspp.h"

// Path to the XML file for the MuJoCo model
const std::string modelFile = "/home/geraldebmer/repos/robocrane/mujoco_env_builder/presets/robocrane/robocrane.xml";



int main(int argc, char** argv) {
    Timer exec_timer;

    std::cout << "Mujoco Collission Checker" << std::endl;

    // Print MuJoCo version
    std::cout << "MuJoCo version: " << mj_version() << std::endl;

    // Print Eigen version
    std::cout << "Eigen version: " << EIGEN_WORLD_VERSION << "."
              << EIGEN_MAJOR_VERSION << "."
              << EIGEN_MINOR_VERSION << std::endl;

    mjModel* m = mj_loadXML(modelFile.c_str(), NULL, NULL, 0);
    mjData* d = mj_makeData(m);

    std::cout << "DoFs: " << m->nq << std::endl;

    exec_timer.tic();
    for(int i = 0; i < 100000; i++) {
        mj_step(m, d);
//        mj_collision(m, d);
    }
    std::cout << "Execution time: " << static_cast<double>(exec_timer.toc())*1e-3/100000. << " us" << std::endl;

    mj_forward(m, d);
    mj_collision(m,d);

    // get the number of contacts
    std::cout << "Number of contacts: " << d->ncon << std::endl;

    // iterate over all contacts
    for(int i=0; i<d->ncon; i++) {
        std::cout << "Contact " << i << ": " << d->contact[i].dist << std::endl;
    }

    constexpr int dof = 3;
    using SSPP = sspp::SSPP<dof>;
    SSPP path_planner;
    using Point = SSPP::Point;
    auto err_code = path_planner.initialize(Point::Zero(), Point::Ones(), 10);
    std::cout << "Error code: " << err_code << std::endl;

//    for(int i = 0; i < 10; i++) {
//        Point p = path_planner.evaluate(static_cast<double>(i)/9.);
//        std::cout << "Point " << i << ": " << p.transpose() << std::endl;
//    }
//
//    exec_timer.tic();
//    auto sampled_spline = path_planner.sample(0.1, Point::Ones());
//    std::cout << "Sampling time: " << static_cast<double>(exec_timer.toc())*1e-3 << " us" << std::endl;
//
//    exec_timer.tic();
//    auto collision_detected = path_planner.check_collision(sampled_spline, m, d);
//    std::cout << "Collision checking time: " << static_cast<double>(exec_timer.toc())*1e-3 << " us" << std::endl;
//    std::cout << "Collision detected: " << collision_detected << std::endl;

    exec_timer.tic();
    auto success = path_planner.plan(Point::Zero(), Point::Ones(), 0.1, Point::Ones(), m, d);
    std::cout << "Planning time: " << static_cast<double>(exec_timer.toc())*1e-3 << " us" << std::endl;
    std::cout << "Path found: " << success << std::endl;

    return 0;
}