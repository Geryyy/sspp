//
// Created by gebmer on 02.03.25.
//

#ifndef VISU_H
#define VISU_H

#include "tsp.h"

inline float via_color[] = {0,1,1,1};
inline float start_color[] = {0,1,0,1};
inline float end_color[] = {1,0,0,1};
inline float path_color[] = {0,1,1,1};
inline float graddesc_color[] = {0,0.7,0.5,0.5};
inline float graddesc_via_color[] = {0,1,1,0.5};

inline float failed_via_color[] = {1.0, 0.5, 0.5, 0.5};       // More red, more transparent
inline float failed_path_color[] = {1.0, 0.5, 0.5, 0.5};      // Same as via for consistency
inline float failed_graddesc_color[] = {0.8, 0.3, 0.3, 0.3};  // Darker red, more transparent
inline float failed_graddesc_via_color[] = {1.0, 0.4, 0.4, 0.3}; // Slightly more red than graddesc

void draw_sphere(mjvScene* scn, Eigen::Vector3d pos, float size=0.1, float* rgba = nullptr);

void draw_arrow(mjvScene* scn, Eigen::Vector3d pos, Eigen::Vector3d gradient, float size=0.1, float* rgba = nullptr);

void draw_path(mjvScene* scn, std::vector<Eigen::Vector3d> pts, float width=0.5, float* rgba = nullptr);

void visualize_candidates(bool vis_candidates, bool vis_grad_desc,
    const std::vector<tsp::PathCandidate>& candidates, tsp::TaskSpacePlanner &planner, mjvScene &scn, int pts_cnt);

#endif //VISU_H
