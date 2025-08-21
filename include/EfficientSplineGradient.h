#pragma once

#include "mujoco/mujoco.h"
#include "utility.h"
#include <Eigen/Core>
#include <unsupported/Eigen/Splines>
#include <limits>
#include <vector>

namespace tsp {

/// Efficient collision gradient evaluation along a spline path.
/// Similar to \c Gradient but operates directly on the spline control points.
/// Example:
/// \code
/// EfficientSplineGradient<kDOF> grad(model, "body");
/// Eigen::MatrixXd g = grad.compute(spline, collision_geoms);
/// \endcode
///
template <int kDOF>
class EfficientSplineGradient {
  using Point = Eigen::Matrix<double, kDOF, 1>;
  using Spline = Eigen::Spline<double, kDOF, kSplineDegree>;

  mjModel *model_;
  mjData *data_;
  Utility::BodyJointInfo joint_info_;
  std::vector<int> body_geom_ids_;
  int samples_;

public:
  EfficientSplineGradient(mjModel *model, const std::string &body_name,
                          int samples = 10)
      : model_(model), data_(mj_makeData(model)),
        joint_info_(Utility::get_free_body_joint_info(body_name, model)),
        samples_(samples) {
    body_geom_ids_.reserve(model_->ngeom);
    for (int g = 0; g < model_->ngeom; ++g) {
      if (model_->geom_bodyid[g] == joint_info_.body_id) {
        body_geom_ids_.push_back(g);
      }
    }
  }

  ~EfficientSplineGradient() { mj_deleteData(data_); }

  void set_samples(int samples) { samples_ = samples; }

  Eigen::Matrix<double, kDOF, Eigen::Dynamic>
  compute(const Spline &spline, const std::vector<int> &collision_geom_ids) {
    const int n_ctrl = spline.ctrls().cols();
    Eigen::Matrix<double, kDOF, Eigen::Dynamic> grad =
        Eigen::Matrix<double, kDOF, Eigen::Dynamic>::Zero(kDOF, n_ctrl);

    for (int i = 0; i < samples_; ++i) {
      double u = (samples_ == 1) ? 0.0 : static_cast<double>(i) / (samples_ - 1);
      Point pos = spline(u);
      Utility::mj_set_point(pos, joint_info_, data_);
      mj_forward(model_, data_);

      double min_dist = std::numeric_limits<double>::infinity();
      Point min_normal = Point::Zero();

      for (int gid_a : body_geom_ids_) {
        for (int gid_b : collision_geom_ids) {
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

      Eigen::RowVectorXd basis = spline.basisFunctionDerivatives(u, 0);
      for (int j = 0; j < basis.size(); ++j) {
        grad.col(j) += min_normal * basis[j];
      }
    }

    grad /= static_cast<double>(samples_);
    return grad;
  }
};

} // namespace tsp

