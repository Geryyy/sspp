//
// Created by gebmer on 02.03.25.
//
#include <Eigen/Dense>
#include <mujoco/mujoco.h>
#include "visu.h"
#include "tsp.h"


void draw_sphere(mjvScene* scn, Eigen::Vector3d pos, float size, float* rgba) {
    Eigen::Vector3d line_size = Eigen::Vector3d::Ones() * size;
    float line_rgba[] = {1,0,1,1};

    if(!rgba) {
        rgba = line_rgba;
    }

    if(scn->ngeom >= scn->maxgeom-1)
        throw std::runtime_error("ngeom > maxgeom!");

    scn->ngeom++;
    mjvGeom *geom_ptr = &scn->geoms[scn->ngeom - 1];
    mjv_initGeom(geom_ptr, mjGEOM_SPHERE, line_size.data(), pos.data(), NULL, rgba);
}


void draw_arrow(mjvScene* scn, Eigen::Vector3d pos, Eigen::Vector3d gradient, float size, float* rgba) {
    if (gradient.norm() < 1e-6) return; // Avoid division by zero for zero gradient

    Eigen::Vector3d line_size;
    line_size << 0.1,0.1,1.0;
    line_size *= size;
    Eigen::Matrix3d rotmat;

    // Normalize gradient to get direction
    Eigen::Vector3d dir = gradient.normalized();
    Eigen::Vector3d z_axis(0, 0, 1); // Default arrow orientation in MuJoCo

    // Compute quaternion rotation from z-axis to gradient direction
    Eigen::Quaterniond q = Eigen::Quaterniond::FromTwoVectors(z_axis, dir);

    // Convert quaternion to rotation matrix
    rotmat = q.toRotationMatrix();

    float line_rgba[] = {1, 0.5, 0.5, 1};
    if (!rgba) {
        rgba = line_rgba;
    }

    if(scn->ngeom >= scn->maxgeom-1)
        throw std::runtime_error("ngeom > maxgeom!");

    scn->ngeom++;
    mjvGeom *geom_ptr = &scn->geoms[scn->ngeom - 1];
    mjv_initGeom(geom_ptr, mjGEOM_ARROW, line_size.data(), pos.data(), rotmat.data(), rgba);
}


void draw_path(mjvScene* scn, std::vector<Eigen::Vector3d> pts, float width, float* rgba) {
    Eigen::Vector3d line_size = Eigen::Vector3d::Ones()*width;
    float line_rgba[] = {1,1,0,1};
    if (!rgba) {
        rgba = line_rgba;
    }

    for(int i = 1; i < pts.size(); i++){
        Eigen::Vector3d start_pos = pts[i-1];
        Eigen::Vector3d end_pos = pts[i];

        if(scn->ngeom >= scn->maxgeom-1)
            throw std::runtime_error("ngeom > maxgeom!");

        scn->ngeom++;
        mjvGeom *geom_ptr = &scn->geoms[scn->ngeom - 1];
        mjv_initGeom(geom_ptr, mjGEOM_SPHERE, line_size.data(), start_pos.data(), NULL, rgba);
        mjv_connector(geom_ptr, mjGEOM_LINE, width, start_pos.data(), end_pos.data());
    }

}


// TODO: add color info!
void visualize_candidates(bool vis_candidates, bool vis_grad_desc,
    const std::vector<tsp::PathCandidate>& candidates,
    tsp::TaskSpacePlanner &planner, mjvScene &scn, int pts_cnt,
    float* path_color, float* via_color, float* graddesc_color, float* graddesc_via_color) {
    if(vis_candidates) {
        for(const auto& candidate : candidates) {
            tsp::Point via_pt = candidate.via_point;
            tsp::Spline spline = planner.path_from_via_pt(via_pt);
            auto pts = planner.get_path_pts(spline, pts_cnt);
            draw_path(&scn, pts, 0.2, path_color);
            draw_sphere(&scn, via_pt, 0.03, via_color);

            if(vis_grad_desc){
                for(const auto& step : candidate.gradient_steps) {
                    tsp::Point via_pt = step.via_point;
                    tsp::Point coll_grad = step.gradient;
                    tsp::Spline spline = planner.path_from_via_pt(via_pt);
                    auto pts = planner.get_path_pts(spline, pts_cnt);
                    draw_path(&scn, pts, 0.1, graddesc_color);
                    draw_sphere(&scn, via_pt, 0.03, graddesc_via_color);
                    draw_arrow(&scn, via_pt, coll_grad, 0.2);
                }
            }
        }
    }
}