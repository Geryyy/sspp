//
// Created by geraldebmer on 15.11.24.
//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include "sspp.h"

#include <pybind11/pybind11.h>

namespace py = pybind11;

template <int kDOF>
void bind_SamplingPathPlanner(py::module &m, const std::string &class_name, const std::string &splineclass_name) {
    using SamplingPathPlanner = sspp::SamplingPathPlanner<kDOF>;
    using Point = typename SamplingPathPlanner::Point;
    using Spline = typename SamplingPathPlanner::Spline;

    py::class_<Spline>(m, splineclass_name.c_str())
    .def(py::init<>())  
    .def("ctrls", &Spline::ctrls, py::return_value_policy::reference_internal);

    py::class_<SamplingPathPlanner>(m, class_name.c_str())
        .def(py::init([](const std::string &xml_string)
                      { return new SamplingPathPlanner(xml_string); }),
             py::arg("xml_string"))
        .def("initializePath", &SamplingPathPlanner::initializePath,
             py::arg("start"), py::arg("end"), py::arg("init_spline"), py::arg("num_points") = 10)
        .def("evaluate", py::overload_cast<double>(&SamplingPathPlanner::evaluate, py::const_),
             py::arg("u"))
        .def("evaluate", py::overload_cast<const Spline &, double>(&SamplingPathPlanner::evaluate, py::const_),
             py::arg("spline"), py::arg("u"))
        .def("sampleWithNoise", &SamplingPathPlanner::sampleWithNoise,
             py::arg("init_spline"), py::arg("sigma"), py::arg("limits"), py::arg("generator"))
        .def("checkCollision", &SamplingPathPlanner::checkCollision,
             py::arg("spline"), py::arg("num_samples"), py::arg("data"))
        .def("computeArcLength", &SamplingPathPlanner::computeArcLength,
             py::arg("spline"), py::arg("check_points"))
        .def("findBestPath", &SamplingPathPlanner::findBestPath,
             py::arg("successful_paths"), py::arg("best_spline"), py::arg("check_points") = 10)
        .def("get_ctrl_pts", &SamplingPathPlanner::get_ctrl_pts, py::return_value_policy::reference_internal)
        .def("plan", [](SamplingPathPlanner &self, 
               const Point &start, const Point &end, double sigma, const Point &limits, int sample_count, int check_points, int init_points)
             {
          std::vector<Spline> result;
          bool success = self.plan(start, end, sigma, limits, result, sample_count, check_points, init_points);
          return py::make_tuple(success, result); }, 
          // return py::make_tuple(success, success); }, 
          py::arg("start"), py::arg("end"), py::arg("sigma"), py::arg("limits"), py::arg("sample_count") = 50, py::arg("check_points") = 50, py::arg("init_points") = 10);
     //    .def("plan", py::overload_cast<const Point &, const Point &, double, const Point &, int, int, int>(&SamplingPathPlanner::plan), py::arg("start"), py::arg("end"), py::arg("sigma"), py::arg("limits"), py::arg("sample_count") = 50, py::arg("check_points") = 50, py::arg("init_points") = 10);
}

// This must be placed outside of any function or template
PYBIND11_MODULE(_sspp, m) {
    m.doc() = "Pybind11 bindings for SamplingPathPlanner";

    bind_SamplingPathPlanner<3>(m, "SamplingPathPlanner3", "Spline3");
    bind_SamplingPathPlanner<6>(m, "SamplingPathPlanner6", "Spline6");
    bind_SamplingPathPlanner<7>(m, "SamplingPathPlanner7", "Spline7");

    m.def("create_model_capsule", [](uint64_t model) {
        return py::capsule(reinterpret_cast<void*>(model), "mjModel");
    });
    m.def("create_data_capsule", [](uint64_t data) {
        return py::capsule(reinterpret_cast<void*>(data), "mjData");
    });
}