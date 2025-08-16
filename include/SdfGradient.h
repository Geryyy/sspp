#pragma once

#include "mujoco/mujoco.h"
#include "utility.h"
#include <Eigen/Core>
#include <limits>
#include <unsupported/Eigen/Splines>
#include <vector>

namespace tsp {

template <int kDOF> class SdfGradient {
  using Point = Eigen::Matrix<double, kDOF, 1>;
  using Spline = Eigen::Spline<double, kDOF, kSplineDegree>;

  mjModel *model_;
  mjData *data_;
  Utility::BodyJointInfo joint_info_;
  std::vector<int> body_geom_ids_;
  std::vector<int> env_geom_ids_;
  int samples_;

public:
  SdfGradient(mjModel *model, const std::string &body_name,
              const std::vector<int> &env_geom_ids, int samples = 10)
      : model_(model), data_(mj_makeData(model)),
        joint_info_(Utility::get_free_body_joint_info(body_name, model)),
        env_geom_ids_(env_geom_ids), samples_(samples) {
    body_geom_ids_.reserve(model_->ngeom);
    for (int g = 0; g < model_->ngeom; ++g) {
      if (model_->geom_bodyid[g] == joint_info_.body_id) {
        body_geom_ids_.push_back(g);
      }
    }
  }

  ~SdfGradient() { mj_deleteData(data_); }

  void set_samples(int samples) { samples_ = samples; }

  Eigen::Matrix<double, kDOF, Eigen::Dynamic> compute(const Spline &spline) {
    const int n_ctrl = spline.ctrls().cols();
    Eigen::Matrix<double, kDOF, Eigen::Dynamic> grad =
        Eigen::Matrix<double, kDOF, Eigen::Dynamic>::Zero(kDOF, n_ctrl);

    for (int i = 0; i <= samples_; ++i) {
      double u = static_cast<double>(i) / samples_;
      Point pos = spline(u);
      auto [dist, normal] = query_sdf(pos);
      (void)dist; // distance can be used for cost weighting if desired
      Eigen::RowVectorXd basis = spline.basisFunctionDerivatives(u, 0);
      for (int j = 0; j < basis.size(); ++j) {
        grad.col(j) += normal * basis[j];
      }
    }
    return grad;
  }

private:
  std::pair<double, Point> query_sdf(const Point &pt) {
    Utility::mj_set_point(pt, joint_info_, data_);
    mj_forward(model_, data_);

    double min_dist = std::numeric_limits<double>::infinity();
    Point min_normal = Point::Zero();

    for (int gid_a : body_geom_ids_) {
      for (int gid_b : env_geom_ids_) {
        mjtNum p1[3];
        mjtNum p2[3];
        mjtNum nrm[3];
        mjtNum dist;
        mj_geomDistance(model_, data_, p1, p2, nrm, &dist, gid_a, gid_b);
        if (dist < min_dist) {
          min_dist = dist;
          min_normal << nrm[0], nrm[1], nrm[2], 0.0;
        }
      }
    }
    return {min_dist, min_normal};
  }
};

} // namespace tsp
