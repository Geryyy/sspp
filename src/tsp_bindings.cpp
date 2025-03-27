#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include "tsp.h"

namespace py = pybind11;
using namespace tsp;

PYBIND11_MODULE(_tsp, m) {
    m.doc() = "Pybind11 bindings for TaskSpacePlanner";

    py::class_<Spline>(m, "Spline")
        .def(py::init<>())
        .def("ctrls", &Spline::ctrls, py::return_value_policy::reference_internal);

    py::class_<TaskSpacePlanner>(m, "TaskSpacePlanner")
        .def(py::init([](const std::string &xml_string) {
            return new TaskSpacePlanner(xml_string);
        }), py::arg("xml_string"))
        .def("initializePath", &TaskSpacePlanner::initializePath,
             py::arg("start"), py::arg("end"), py::arg("end_derivative"), py::arg("num_points") = 3)
        .def("evaluate", py::overload_cast<double>(&TaskSpacePlanner::evaluate, py::const_),
             py::arg("u"))
     //    .def("evaluate", py::overload_cast<double, const Spline &>(&TaskSpacePlanner::evaluate),
     //         py::arg("u"), py::arg("spline"))
        .def("get_via_pts", &TaskSpacePlanner::get_via_pts, py::return_value_policy::reference_internal)
        .def("get_ctrl_pts", &TaskSpacePlanner::get_ctrl_pts, py::return_value_policy::reference_internal)
        .def("check_collision", &TaskSpacePlanner::check_collision,
             py::arg("spline"), py::arg("num_samples"))
        .def("computeArcLength", &TaskSpacePlanner::computeArcLength,
             py::arg("spline"), py::arg("check_points"))
        .def("findBestPath", &TaskSpacePlanner::findBestPath,
             py::arg("candidates"), py::arg("best_spline"), py::arg("check_points") = 10)
        .def("plan", &TaskSpacePlanner::plan,
             py::arg("start"), py::arg("end"), py::arg("end_derivative"), py::arg("sigma"),
             py::arg("limits"), py::arg("sample_count") = 50, py::arg("check_points") = 50,
             py::arg("gd_iterations") = 10, py::arg("init_points") = 3);
}