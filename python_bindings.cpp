//
// Created by geraldebmer on 15.11.24.
//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include "sspp.h"

namespace py = pybind11;

template<int kDOF>
void bind_SamplingPathPlanner(py::module& m, const std::string& class_name) {
    using SamplingPathPlanner = sspp::SamplingPathPlanner<kDOF>;
    using Point = typename SamplingPathPlanner::Point;
    using Spline = typename SamplingPathPlanner::Spline;

    py::class_<SamplingPathPlanner>(m, class_name.c_str())
            .def(py::init<mjModel*, mjData*>(), py::arg("model"), py::arg("data"))
            .def("initializePath", &SamplingPathPlanner::initializePath,
                 py::arg("start"), py::arg("end"), py::arg("init_spline"), py::arg("num_points") = 10)
            .def("evaluate", py::overload_cast<double>(&SamplingPathPlanner::evaluate, py::const_),
                 py::arg("u"))
            .def("evaluate", py::overload_cast<const Spline&, double>(&SamplingPathPlanner::evaluate, py::const_),
                 py::arg("spline"), py::arg("u"))
            .def("sampleWithNoise", &SamplingPathPlanner::sampleWithNoise,
                 py::arg("init_spline"), py::arg("sigma"), py::arg("limits"), py::arg("generator"))
            .def("checkCollision", &SamplingPathPlanner::checkCollision,
                 py::arg("spline"), py::arg("num_samples"), py::arg("data"))
            .def("computeArcLength", &SamplingPathPlanner::computeArcLength,
                 py::arg("spline"), py::arg("check_points"))
            .def("findBestPath", &SamplingPathPlanner::findBestPath,
                 py::arg("successful_paths"), py::arg("best_spline"), py::arg("check_points") = 10)
            .def("plan", py::overload_cast<const Point&, const Point&, double, const Point&, std::vector<Spline>&, int, int, int>(
                         &SamplingPathPlanner::plan),
                 py::arg("start"), py::arg("end"), py::arg("sigma"), py::arg("limits"),
                 py::arg("ret_success_paths"), py::arg("sample_count") = 50,
                 py::arg("check_points") = 50, py::arg("init_points") = 10)
            .def("plan", py::overload_cast<const Point&, const Point&, double, const Point&, int, int, int>(
                         &SamplingPathPlanner::plan),
                 py::arg("start"), py::arg("end"), py::arg("sigma"), py::arg("limits"),
                 py::arg("sample_count") = 50, py::arg("check_points") = 50, py::arg("init_points") = 10);
}

PYBIND11_MODULE(_sspp, m) {
    m.doc() = "Pybind11 bindings for SamplingPathPlanner";

    bind_SamplingPathPlanner<3>(m, "SamplingPathPlanner3");
    bind_SamplingPathPlanner<6>(m, "SamplingPathPlanner6");
}
