//
// Created by geraldebmer on 11.11.24.
//

#ifndef SSPP_SSPP_H
#define SSPP_SSPP_H

#include <vector>
#include <Eigen/Core>
#include <unsupported/Eigen/Splines>
#include <iostream>
#include <random>

namespace sspp {

    template<int DOF>
    class SSPP {
    public:
        using Point = Eigen::Matrix<double, DOF, 1>;
    private:
        static const int spline_deg = 3;
        typedef Eigen::Spline<double, DOF, spline_deg> Spline_t;
        typedef Eigen::SplineFitting<Spline_t> SplineFitting_t;

        Spline_t init_spline;

    public:
        SSPP() = default;
        ~SSPP() = default;

        int initialize(Point start, Point end, int num_pts){
            Eigen::VectorXd u_knots(num_pts, 1);
            Eigen::MatrixXd via_points(DOF, num_pts);

            for (int i = 0; i < num_pts; ++i) {
                double t = static_cast<double>(i) / (num_pts - 1);
                Point point = (1 - t) * start + t * end;  // Linear interpolation
                via_points.col(i) = point;
                u_knots(i) = t;
            }

            init_spline = SplineFitting_t::Interpolate(via_points, spline_deg, u_knots);
            return 0;
        }

        Point evaluate(double u) {
            return init_spline(u);
        }

        Point evaluate(Spline_t spline, double u) {
            return spline(u);
        }

        Spline_t sample(double sigma, Point limits) {
            auto ctrl_pts = init_spline.ctrls();
            int p = spline_deg;

            std::default_random_engine generator;
            std::normal_distribution<double> distribution(0.0,sigma);

            // Add noise to the control points
            for (int i = 0; i < ctrl_pts.rows(); ++i) {
                for (int j = p; j < ctrl_pts.cols() - p; ++j) {
                    double noise = distribution(generator);
                    ctrl_pts(i, j) += noise * limits(i);
                }
            }

            Spline_t sampled_spline(init_spline.knots(), ctrl_pts);
//            std::cout << "ctrl pts rows: " << ctrl_pts.rows() << " cols: " << ctrl_pts.cols() << std::endl;
            return sampled_spline;
        }


//        bool check_collision(Point p) {
//            return true;
//        }




    };

} // sspp

#endif //SSPP_SSPP_H
