#include <iostream>
#include <string>
#include <mujoco/mujoco.h>
#include <Eigen/Core>
#include "Timer.h"
#include "sspp.h"
#include "failed_dreams/dspp.h"

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

//    exec_timer.tic();
//    for(int i = 0; i < 100000; i++) {
//        mj_step(m, d);
////        mj_collision(m, d);
//    }
//    std::cout << "Execution time: " << static_cast<double>(exec_timer.toc())*1e-3/100000. << " us" << std::endl;

    mj_forward(m, d);
    mj_collision(m,d);

//    // get the number of contacts
//    std::cout << "Number of contacts: " << d->ncon << std::endl;
//
//    // iterate over all contacts
//    for(int i=0; i<d->ncon; i++) {
//        std::cout << "Contact " << i << ": " << d->contact[i].dist << std::endl;
//    }

    // static sampling path planner
    std::cout << "Static Sampling Path Planner" << std::endl;
    constexpr int dof = 3;
    using SPP = sspp::SamplingPathPlanner<dof>;
    SPP path_planner(m, d);
    using Point = SPP::Point;
    SPP::Spline init_spline;
    auto err_code = path_planner.initializePath(Point::Zero(), Point::Ones(), init_spline, 10);
    std::cout << "Error code: " << err_code << std::endl;

    exec_timer.tic();
    auto success = path_planner.plan(Point::Zero(), Point::Ones(), 0.5, Point::Ones());
    std::cout << "Planning time: " << static_cast<double>(exec_timer.toc())*1e-3 << " us" << std::endl;
    std::cout << "Path found: " << success << std::endl;

    for(int i = 0; i < 10; i++) {
        Point p = path_planner.evaluate(static_cast<double>(i)/9.);
        std::cout << "Point " << i << ": " << p.transpose() << std::endl;
    }

    // dynamic sampling path planner
//    std::cout << "Dynamic Sampling Path Planner" << std::endl;
//    using DSP = dspp::DynamicSamplingPathPlanner;
//    using Point = DSP::Point;
//    DSP dsp(m, d, dof);
//    DSP::Spline dsp_init_spline;
//    auto err_code = dsp.initializePath(Point::Zero(dof), Point::Ones(dof), dsp_init_spline, 10);
//    std::cout << "Error code: " << err_code << std::endl;
//
//    exec_timer.tic();
//    auto success = dsp.plan(Point::Zero(dof), Point::Ones(dof), 0.5, Point::Ones(dof));
//    std::cout << "Planning time: " << static_cast<double>(exec_timer.toc())*1e-3 << " us" << std::endl;
//    std::cout << "Path found: " << success << std::endl;
//
//    for(int i = 0; i < 10; i++) {
//        Point p = dsp.evaluate(static_cast<double>(i)/9.);
//        std::cout << "Point " << i << ": " << p.transpose() << std::endl;
//    }

    return 0;
}