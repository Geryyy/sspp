//
// Created by gebmer on 02.03.25.
//
#include <Eigen/Dense>
#include <mujoco/mujoco.h>
#include "visu.h"
#include "tsp.h"

namespace visu {
    void draw_sphere(mjvScene *scn, Point3 pos, float size, float *rgba) {
        Point3 line_size = Point3::Ones() * size;
        float line_rgba[] = {1, 0, 1, 1};

        if (!rgba) {
            rgba = line_rgba;
        }

        if (scn->ngeom >= scn->maxgeom - 1)
            throw std::runtime_error("ngeom > maxgeom!");

        scn->ngeom++;
        mjvGeom *geom_ptr = &scn->geoms[scn->ngeom - 1];
        mjv_initGeom(geom_ptr, mjGEOM_SPHERE, line_size.data(), pos.data(), NULL, rgba);
    }


    void draw_arrow(mjvScene *scn, Point3 pos, Point3 gradient, float size, float *rgba) {
        if (gradient.norm() < 1e-6) return; // Avoid division by zero for zero gradient

        Point3 line_size;
        line_size << 0.1, 0.1, 1.0;
        line_size *= size;
        Eigen::Matrix3d rotmat;

        // Normalize gradient to get direction
        Point3 dir = gradient.normalized();
        Point3 z_axis(0, 0, 1); // Default arrow orientation in MuJoCo

        // Compute quaternion rotation from z-axis to gradient direction
        Eigen::Quaterniond q = Eigen::Quaterniond::FromTwoVectors(z_axis, dir);

        // Convert quaternion to rotation matrix
        rotmat = q.toRotationMatrix();

        float line_rgba[] = {1, 0.5, 0.5, 1};
        if (!rgba) {
            rgba = line_rgba;
        }

        if (scn->ngeom >= scn->maxgeom - 1)
            throw std::runtime_error("ngeom > maxgeom!");

        scn->ngeom++;
        mjvGeom *geom_ptr = &scn->geoms[scn->ngeom - 1];
        mjv_initGeom(geom_ptr, mjGEOM_ARROW, line_size.data(), pos.data(), rotmat.data(), rgba);
    }


    void draw_path(mjvScene *scn, std::vector<Point3> pts, float width, float *rgba) {
        Point3 line_size = Point3::Ones() * width;
        float line_rgba[] = {1, 1, 0, 1};
        if (!rgba) {
            rgba = line_rgba;
        }

        for (int i = 1; i < pts.size(); i++) {
            Point3 start_pos = pts[i - 1];
            Point3 end_pos = pts[i];

            if (scn->ngeom >= scn->maxgeom - 1)
                throw std::runtime_error("ngeom > maxgeom!");

            scn->ngeom++;
            mjvGeom *geom_ptr = &scn->geoms[scn->ngeom - 1];
            mjv_initGeom(geom_ptr, mjGEOM_SPHERE, line_size.data(), start_pos.data(), NULL, rgba);
            mjv_connector(geom_ptr, mjGEOM_LINE, width, start_pos.data(), end_pos.data());
        }

    }

    Point3 convert_point(tsp::Point pt){
        Point3 pt3 = pt.block<3,1>(0,0);
        return pt3;
    }

    std::vector<Point3> convert_point_vector(std::vector<tsp::Point> pts){
        std::vector<Point3> pts3;
        for(auto pt : pts){
            pts3.push_back(pt.block<3,1>(0,0));
        }
        return pts3;
    }

// TODO: add color info!
    void visualize_candidates(bool vis_candidates, bool vis_grad_desc,
                              const std::vector<tsp::PathCandidate> &candidates,
                              tsp::TaskSpacePlanner &planner, mjvScene &scn, int pts_cnt,
                              float *path_color, float *via_color, float *graddesc_color, float *graddesc_via_color) {
        if (vis_candidates) {
            for (const auto &candidate: candidates) {
                tsp::Point via_pt = (candidate.via_point);
                tsp::Spline spline = planner.path_from_via_pt(via_pt);
                auto pts = planner.get_path_pts(spline, pts_cnt);
                auto pts3 = convert_point_vector(pts);
                draw_path(&scn, pts3, 0.2, path_color);
                draw_sphere(&scn, convert_point(via_pt), 0.03, via_color);

                if (vis_grad_desc) {
                    for (const auto &step: candidate.gradient_steps) {
                        tsp::Point via_pt = step.via_point;
                        Point3 coll_grad = convert_point(step.gradient);
                        tsp::Spline spline = planner.path_from_via_pt(via_pt);
                        Point3 via_pt3 = convert_point(via_pt);
                        auto pts = planner.get_path_pts(spline, pts_cnt);
                        auto pts3 = convert_point_vector(pts);
                        draw_path(&scn, pts3, 0.1, graddesc_color);
                        draw_sphere(&scn, via_pt3, 0.03, graddesc_via_color);
                        draw_arrow(&scn, via_pt3, coll_grad, 0.2);
                    }
                }
            }
        }
    }

    void visualize_via_pts(bool vis_via_pts, const std::vector<tsp::Point> &via_pts,
                           tsp::TaskSpacePlanner &planner, mjvScene &scn, int pts_cnt,
                           float *path_color, float *via_color) {
        if (vis_via_pts) {
            for (const auto &pt: via_pts) {
                const tsp::Point &via_pt = pt;
                tsp::Spline spline = planner.path_from_via_pt(via_pt);
                auto pts = planner.get_path_pts(spline, pts_cnt);
                draw_path(&scn, convert_point_vector(pts), 0.2, path_color);
                draw_sphere(&scn, convert_point(via_pt), 0.03, via_color);
            }
        }
    }

} // namespace visu