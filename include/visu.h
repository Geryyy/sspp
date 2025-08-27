//
// Created by gebmer on 02.03.25.
//

#ifndef VISU_H
#define VISU_H

#include <vector>
#include <Eigen/Core>
#include <unsupported/Eigen/Splines>
#include <mujoco/mujoco.h>

#include "sspp/tsp.h"  // your modular adapter (TaskSpacePlanner, PathCandidate, etc.)

namespace visu {

    using Point3 = Eigen::Vector3d;

// --- colors (RGBA) -----------------------------------------------------------
    inline float via_color[]                = {0.0f, 1.0f, 1.0f, 1.0f};
    inline float start_color[]              = {0.0f, 1.0f, 0.0f, 1.0f};
    inline float end_color[]                = {1.0f, 0.0f, 0.0f, 1.0f};
    inline float path_color[]               = {0.0f, 1.0f, 1.0f, 1.0f};
    inline float graddesc_color[]           = {0.0f, 0.7f, 0.5f, 0.5f};
    inline float graddesc_via_color[]       = {0.0f, 1.0f, 1.0f, 0.5f};

    inline float sampled_via_color[]        = {0.5f, 1.0f, 0.5f, 0.5f};
    inline float sampled_via_path_color[]   = {0.5f, 1.0f, 0.5f, 0.5f};

    inline float failed_via_color[]         = {1.0f, 0.5f, 0.5f, 0.5f};
    inline float failed_path_color[]        = {1.0f, 0.5f, 0.5f, 0.5f};
    inline float failed_graddesc_color[]    = {0.8f, 0.3f, 0.3f, 0.3f};
    inline float failed_graddesc_via_color[]= {1.0f, 0.4f, 0.4f, 0.3f};

// --- small helpers -----------------------------------------------------------
    inline Point3 convert_point(const tsp::Point& pt) {
        return pt.template head<3>();
    }

    inline std::vector<Point3> convert_point_vector(const std::vector<tsp::Point>& pts) {
        std::vector<Point3> out;
        out.reserve(pts.size());
        for (const auto& p : pts) out.emplace_back(convert_point(p));
        return out;
    }

// sample N points from any spline using the adapter's static evaluate(u, s)
    inline std::vector<tsp::Point> sample_spline(const tsp::Spline& s, int N) {
        std::vector<tsp::Point> pts;
        if (N < 2) return pts;
        pts.reserve(N);
        for (int i = 0; i < N; ++i) {
            const double u = (N == 1) ? 0.0 : double(i) / double(N - 1);
            pts.push_back(tsp::TaskSpacePlanner::evaluate(u, s));
        }
        return pts;
    }

// --- drawing primitives (implemented in visu.cpp) ----------------------------
    void draw_sphere(mjvScene* scn, Point3 pos, float size = 0.1f, float* rgba = nullptr);
    void draw_arrow (mjvScene* scn, Point3 pos, Point3 gradient, float size = 0.1f, float* rgba = nullptr);
    void draw_path  (mjvScene* scn, std::vector<Point3> pts, float width = 0.5f, float* rgba = nullptr);

// --- visualize path candidates ----------------------------------------------
    inline void visualize_candidates(bool vis_candidates,
                                     bool vis_grad_desc,
                                     const std::vector<tsp::PathCandidate>& candidates,
                                     tsp::TaskSpacePlanner& planner,
                                     mjvScene& scn,
                                     int pts_cnt,
                                     float* path_rgba,
                                     float* via_rgba,
                                     float* graddesc_path_rgba,
                                     float* graddesc_via_rgba)
    {
        if (!vis_candidates) return;

        for (const auto& cand : candidates) {
            // prefer refined via-point if present
            const tsp::Point via_pt = cand.refined ? *cand.refined : cand.via;

            // build spline for this via point and draw
            const tsp::Spline spline = planner.spline_from_via(via_pt);
            const auto pts  = sample_spline(spline, pts_cnt);
            draw_path(&scn, convert_point_vector(pts), 0.2f, path_rgba);
            draw_sphere(&scn, convert_point(via_pt), 0.03f, via_rgba);

            if (vis_grad_desc && !cand.steps.empty()) {
                // NOTE: in the modular types, GradientStep has fields {x, f}
                // We visualize the intermediate via points as spheres/paths; no arrow (no gradient vector stored).
                for (const auto& step : cand.steps) {
                    const tsp::Point step_via = step.x;
                    const tsp::Spline step_spline = planner.spline_from_via(step_via);
                    const auto step_pts = sample_spline(step_spline, pts_cnt);
                    draw_path(&scn, convert_point_vector(step_pts), 0.1f, graddesc_path_rgba);
                    draw_sphere(&scn, convert_point(step_via), 0.03f, graddesc_via_rgba);
                    // If you later add a gradient vector to GradientStep, call draw_arrow(...) here.
                }
            }
        }
    }

// --- visualize a list of via points (e.g., sampled set) ----------------------
    inline void visualize_via_pts(bool vis_via_pts,
                                  const std::vector<tsp::Point>& via_pts,
                                  tsp::TaskSpacePlanner& planner,
                                  mjvScene& scn,
                                  int pts_cnt,
                                  float* path_rgba,
                                  float* via_rgba)
    {
        if (!vis_via_pts) return;

        for (const auto& via_pt : via_pts) {
            const tsp::Spline spline = planner.spline_from_via(via_pt);
            const auto pts = sample_spline(spline, pts_cnt);
            draw_path(&scn, convert_point_vector(pts), 0.2f, path_rgba);
            draw_sphere(&scn, convert_point(via_pt), 0.03f, via_rgba);
        }
    }

} // namespace visu

#endif // VISU_H
