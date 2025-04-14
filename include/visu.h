//
// Created by gebmer on 02.03.25.
//

#ifndef VISU_H
#define VISU_H

#include "tsp.h"
namespace visu
{
    using Point3 = Eigen::Vector3d;

    inline float via_color[] = {0, 1, 1, 1};
    inline float start_color[] = {0, 1, 0, 1};
    inline float end_color[] = {1, 0, 0, 1};
    inline float path_color[] = {0, 1, 1, 1};
    inline float graddesc_color[] = {0, 0.7, 0.5, 0.5};
    inline float graddesc_via_color[] = {0, 1, 1, 0.5};

    inline float sampled_via_color[] = {0.5, 1, 0.5, 0.5};
    inline float sampled_via_path_color[] = {0.5, 1, 0.5, 0.5};

    inline float failed_via_color[] = {1.0, 0.5, 0.5, 0.5};          // More red, more transparent
    inline float failed_path_color[] = {1.0, 0.5, 0.5, 0.5};         // Same as via for consistency
    inline float failed_graddesc_color[] = {0.8, 0.3, 0.3, 0.3};     // Darker red, more transparent
    inline float failed_graddesc_via_color[] = {1.0, 0.4, 0.4, 0.3}; // Slightly more red than graddesc

    Point3 convert_point(tsp::Point pt);

    std::vector<Point3> convert_point_vector(std::vector<tsp::Point> pts);

    void draw_sphere(mjvScene *scn, Point3 pos, float size = 0.1, float *rgba = nullptr);

    void draw_arrow(mjvScene *scn, Point3 pos, Point3 gradient, float size = 0.1, float *rgba = nullptr);

    void draw_path(mjvScene *scn, std::vector<Point3> pts, float width = 0.5, float *rgba = nullptr);

    inline void visualize_candidates(bool vis_candidates, bool vis_grad_desc,
                              const std::vector<tsp::PathCandidate> &candidates,
                              tsp::TaskSpacePlanner &planner, mjvScene &scn, int pts_cnt,
                              float *path_color, float *via_color, float *graddesc_color, float *graddesc_via_color)
    {
        if (vis_candidates)
        {
            for (const auto &candidate : candidates)
            {
                tsp::Point via_pt = (candidate.via_point);
                tsp::Spline spline = planner.path_from_via_pt(via_pt);
                auto pts = planner.get_path_pts(spline, pts_cnt);
                auto pts3 = visu::convert_point_vector(pts);
                visu::draw_path(&scn, pts3, 0.2, path_color);
                visu::draw_sphere(&scn, convert_point(via_pt), 0.03, via_color);

                if (vis_grad_desc)
                {
                    for (const auto &step : candidate.gradient_steps)
                    {
                        tsp::Point via_pt = step.via_point;
                        Point3 coll_grad = visu::convert_point(step.gradient);
                        tsp::Spline spline = planner.path_from_via_pt(via_pt);
                        Point3 via_pt3 = visu::convert_point(via_pt);
                        auto pts = planner.get_path_pts(spline, pts_cnt);
                        auto pts3 = visu::convert_point_vector(pts);
                        visu::draw_path(&scn, pts3, 0.1, graddesc_color);
                        visu::draw_sphere(&scn, via_pt3, 0.03, graddesc_via_color);
                        visu::draw_arrow(&scn, via_pt3, coll_grad, 0.2);
                    }
                }
            }
        }
    }

    inline void visualize_via_pts(bool vis_via_pts, const std::vector<tsp::Point> &via_pts,
                                  tsp::TaskSpacePlanner &planner, mjvScene &scn, int pts_cnt,
                                  float *path_color, float *via_color)
    {
        if (vis_via_pts)
        {
            for (const auto &pt : via_pts)
            {
                const tsp::Point &via_pt = pt;
                tsp::Spline spline = planner.path_from_via_pt(via_pt);
                auto pts = planner.get_path_pts(spline, pts_cnt);
                visu::draw_path(&scn, visu::convert_point_vector(pts), 0.2, path_color);
                visu::draw_sphere(&scn, visu::convert_point(via_pt), 0.03, via_color);
            }
        }
    }

} // namespace visu
#endif // VISU_H
