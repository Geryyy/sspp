#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include "tsp.h"

namespace py = pybind11;
using namespace tsp;

PYBIND11_MODULE(_tsp, m) {
    m.doc() = "Pybind11 bindings for TaskSpacePlanner with iterative planning";

    // Bind Point type for easier access
    py::class_<Point>(m, "Point")
            .def(py::init<>())
            .def("__getitem__", [](const Point &p, int i) {
                if (i >= 4 || i < 0) throw py::index_error();
                return p[i];
            })
            .def("__setitem__", [](Point &p, int i, double value) {
                if (i >= 4 || i < 0) throw py::index_error();
                p[i] = value;
            })
            .def("__len__", [](const Point &) { return 4; })
            .def("__repr__", [](const Point &p) {
                return "Point([" + std::to_string(p[0]) + ", " + std::to_string(p[1]) +
                       ", " + std::to_string(p[2]) + ", " + std::to_string(p[3]) + "])";
            })
            .def("norm", &Point::norm)
            .def("transpose", [](const Point &p) {
                return p.transpose().eval();
            }, "Get transpose of the point");

    // Bind Spline class
    py::class_<Spline>(m, "Spline")
            .def(py::init<>())
            .def("ctrls", &Spline::ctrls, py::return_value_policy::reference_internal)
            .def("knots", &Spline::knots, py::return_value_policy::reference_internal)
            .def("__call__", [](const Spline &s, double u) { return s(u); },
                 "Evaluate spline at parameter u");

    // Bind SolverStatus enum
    py::enum_<SolverStatus>(m, "SolverStatus")
            .value("Converged", SolverStatus::Converged)
            .value("MaxIterationsExceeded", SolverStatus::MaxIterationsExceeded)
            .value("BelowFloor", SolverStatus::BelowFloor)
            .value("Invalid", SolverStatus::Invalid)
            .value("Failed", SolverStatus::Failed)
            .export_values();

    // Add string conversion for SolverStatus
    m.def("solver_status_to_string", &SolverStatustoString,
          "Convert SolverStatus enum to string representation");

    // Bind GradientStep struct
    py::class_<GradientStepType>(m, "GradientStep")
            .def(py::init<Point, Point, double>(),
                 py::arg("via_point"), py::arg("gradient"), py::arg("cost"))
            .def_readwrite("via_point", &GradientStepType::via_point)
            .def_readwrite("gradient", &GradientStepType::gradient)
            .def_readwrite("cost", &GradientStepType::cost)
            .def("__repr__", [](const GradientStepType &gs) {
                return "GradientStep(via_point=" + std::to_string(gs.via_point[0]) + "," +
                       std::to_string(gs.via_point[1]) + "," + std::to_string(gs.via_point[2]) +
                       ", cost=" + std::to_string(gs.cost) + ")";
            });

    // Bind PathCandidate struct
    py::class_<PathCandidate>(m, "PathCandidate")
            .def(py::init<Point, std::vector<GradientStepType>, SolverStatus>(),
                 py::arg("via_point"), py::arg("gradient_steps"), py::arg("status"))
            .def_readwrite("via_point", &PathCandidate::via_point)
            .def_readwrite("gradient_steps", &PathCandidate::gradient_steps)
            .def_readwrite("status", &PathCandidate::status)
            .def("__repr__", [](const PathCandidate &pc) {
                return "PathCandidate(status=" + SolverStatustoString(pc.status) +
                       ", gradient_steps=" + std::to_string(pc.gradient_steps.size()) + ")";
            });

    // Bind Gradient class (for advanced users)
    py::class_<Gradient<4>>(m, "Gradient")
            .def(py::init<typename Gradient<4>::CostFunction, const Point&, const Point&>(),
                 py::arg("cost_function"), py::arg("delta"), py::arg("point"))
            .def("compute", &Gradient<4>::compute,
                 "Compute gradient using finite differences");

    // Bind GradientDescent class
    py::class_<GradientDescentType>(m, "GradientDescent")
            .def(py::init<double, int, typename GradientDescentType::CostFunction, const Point&>(),
                 py::arg("step_size"), py::arg("iterations"), py::arg("cost_function"), py::arg("delta"))
            .def("optimize", &GradientDescentType::optimize, py::arg("via_candidate"),
                 "Optimize via point using gradient descent with Barzilai-Borwein step size")
            .def("get_result", &GradientDescentType::get_result,
                 "Get optimization result")
            .def("get_gradient_descent_steps", &GradientDescentType::get_gradient_descent_steps,
                 "Get all gradient descent steps taken during optimization");

    // Main TaskSpacePlanner class with updated constructor parameters
    py::class_<TaskSpacePlanner>(m, "TaskSpacePlanner")
            .def(py::init([](const std::string &xml_string,
                             const std::string &body_name,
                             double stddev_initial,
                             double stddev_min,
                             double stddev_max,
                             double stddev_increase_factor,
                             double stddev_decay_factor,
                             double elite_fraction,
                             int sample_count,
                             int check_points,
                             int gd_iterations,
                             int init_points,
                             double collision_weight,
                             double z_min,
                             const Point &limits_min,
                             const Point &limits_max,
                             bool enable_gradient_descent) {
                     return new TaskSpacePlanner(xml_string, body_name,
                                                 stddev_initial, stddev_min, stddev_max,
                                                 stddev_increase_factor, stddev_decay_factor,
                                                 elite_fraction, sample_count, check_points,
                                                 gd_iterations, init_points, collision_weight,
                                                 z_min, limits_min, limits_max, enable_gradient_descent);
                 }),
                 py::arg("xml_string"),
                 py::arg("body_name"),
                 py::arg("stddev_initial") = 0.3,
                 py::arg("stddev_min") = 0.01,
                 py::arg("stddev_max") = 2.0,
                 py::arg("stddev_increase_factor") = 1.5,
                 py::arg("stddev_decay_factor") = 0.95,
                 py::arg("elite_fraction") = 0.3,
                 py::arg("sample_count") = 50,
                 py::arg("check_points") = 50,
                 py::arg("gd_iterations") = 10,
                 py::arg("init_points") = 3,
                 py::arg("collision_weight") = 1.0,
                 py::arg("z_min") = 0.0,
                 py::arg("limits_min") = -Point::Ones() * 2.0,
                 py::arg("limits_max") = Point::Ones() * 2.0,
                 py::arg("enable_gradient_descent") = true,
                 R"pbdoc(
        Initialize TaskSpacePlanner with evolutionary path planning.

        Parameters:
        - xml_string: MuJoCo XML model as string
        - body_name: Name of collision body to track
        - stddev_initial: Initial sampling standard deviation
        - stddev_min/max: Bounds for adaptive stddev
        - stddev_increase/decay_factor: Adaptation rates
        - elite_fraction: Fraction of best candidates for distribution update
        - sample_count: Number of via point candidates per iteration
        - check_points: Collision checking resolution along splines
        - gd_iterations: Gradient descent iterations (0 to disable)
        - init_points: Number of initial spline control points
        - collision_weight: Weight of collision cost vs arc length
        - z_min: Minimum z-coordinate (ground level)
        - limits_min: Minimum sampling bounds for each dimension [x,y,z,rot]
        - limits_max: Maximum sampling bounds for each dimension [x,y,z,rot]
        - enable_gradient_descent: Enable/disable gradient refinement
        )pbdoc")

                    // Simplified plan method - main interface
            .def("plan", &TaskSpacePlanner::plan,
                 py::arg("start"), py::arg("end"), py::arg("iterate_flag") = false,
                 R"pbdoc(
             Plan a path from start to end using evolutionary strategy.

             Parameters:
             - start: Starting point [x,y,z,rotation]
             - end: Target point [x,y,z,rotation]
             - iterate_flag: False for new path, True for iterative refinement

             Returns:
             - List of successful PathCandidate objects
             )pbdoc")

                    // Core evaluation and path methods
            .def("evaluate", py::overload_cast<double>(&TaskSpacePlanner::evaluate, py::const_),
                 py::arg("u"), "Evaluate current best spline at parameter u ∈ [0,1]")
            .def("get_path_pts", py::overload_cast<int>(&TaskSpacePlanner::get_path_pts, py::const_),
                 py::arg("pts_cnt") = 10,
                 "Get discrete points along the current best path")

                    // State and results access
            .def("get_succesful_path_candidates", &TaskSpacePlanner::get_succesful_path_candidates,
                 "Get all successful path candidates from last planning iteration")
            .def("get_failed_path_candidates", &TaskSpacePlanner::get_failed_path_candidates,
                 "Get all failed path candidates from last planning iteration")
            .def("get_sampled_via_pts", &TaskSpacePlanner::get_sampled_via_pts,
                 "Get all sampled via points from last planning iteration")

                    // Distribution state monitoring (for convergence analysis)
            .def("get_current_mean", &TaskSpacePlanner::get_current_mean,
                 "Get current sampling distribution mean")
            .def("get_current_stddev", &TaskSpacePlanner::get_current_stddev,
                 "Get current sampling distribution standard deviation")
            .def("get_limits_min", &TaskSpacePlanner::get_limits_min,
                 "Get minimum sampling limits for each dimension")
            .def("get_limits_max", &TaskSpacePlanner::get_limits_max,
                 "Get maximum sampling limits for each dimension")

                    // Spline inspection methods
            .def("get_via_pts", &TaskSpacePlanner::get_via_pts,
                 py::return_value_policy::reference_internal,
                 "Get current via points used for spline construction")
            .def("get_ctrl_pts", &TaskSpacePlanner::get_ctrl_pts,
                 py::return_value_policy::reference_internal,
                 "Get control points of current best spline")
            .def("get_knot_vector", &TaskSpacePlanner::get_knot_vector,
                 py::return_value_policy::reference_internal,
                 "Get knot vector of current best spline")

                    // Utility methods
            .def("reset", &TaskSpacePlanner::reset,
                 "Reset internal state (clear candidates and statistics)")
            .def("initializePath", &TaskSpacePlanner::initializePath,
                 py::arg("start"), py::arg("end"), py::arg("num_points") = 3,
                 "Initialize linear path between start and end points")
            .def("path_from_via_pt", &TaskSpacePlanner::path_from_via_pt,
                 py::arg("via_pt"), "Create spline path through given via point")

                    // Static utility methods
            .def_static("evaluate_spline",
                        [](double u, const Spline& spline) { return TaskSpacePlanner::evaluate(u, spline); },
                        py::arg("u"), py::arg("spline"),
                        "Static method to evaluate any spline at parameter u")
            .def_static("get_spline_path_pts",
                        [](const Spline& spline, int pts_cnt) { return TaskSpacePlanner::get_path_pts(spline, pts_cnt); },
                        py::arg("spline"), py::arg("pts_cnt") = 10,
                        "Static method to get discrete points along any spline");

    // Utility functions for creating Points from Python
    m.def("create_point", [](double x, double y, double z, double w = 0.0) {
              Point p;
              p << x, y, z, w;
              return p;
          }, py::arg("x"), py::arg("y"), py::arg("z"), py::arg("w") = 0.0,
          "Create a Point from individual coordinates");

    m.def("create_point_from_list", [](const std::vector<double>& coords) {
        if (coords.size() < 3 || coords.size() > 4) {
            throw std::runtime_error("Point must have 3 or 4 coordinates");
        }
        Point p;
        p << coords[0], coords[1], coords[2], (coords.size() == 4 ? coords[3] : 0.0);
        return p;
    }, py::arg("coords"), "Create a Point from a list of 3 or 4 coordinates");

    m.def("point_ones", [](double value = 1.0) {
        return Point::Ones() * value;
    }, py::arg("value") = 1.0, "Create a Point with all components set to value");

    m.def("point_zero", []() {
        return Point::Zero();
    }, "Create a Point with all components set to zero");

    // Mathematical operations on Points
    m.def("point_norm", [](const Point& p) {
        return p.norm();
    }, py::arg("point"), "Calculate Euclidean norm of point");

    m.def("point_distance", [](const Point& a, const Point& b) {
        return (a - b).norm();
    }, py::arg("point_a"), py::arg("point_b"), "Calculate distance between two points");

    // Constants and version info
    m.attr("DOF") = py::int_(kDOF);
    m.attr("SPLINE_DEGREE") = py::int_(kSplineDegree);
    m.attr("__version__") = "1.0.0";
}