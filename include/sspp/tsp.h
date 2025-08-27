#pragma once
#include "sspp/tsp_planner.h"

namespace tsp {

// This adapter keeps your existing API (methods & signatures) intact.
    class TaskSpacePlanner {
    public:
        using Spline = tsp::Spline;
        using GradientStepType = tsp::GradientStepType;

        TaskSpacePlanner(mjModel* model, std::string body_name,
                         double stddev_initial = 0.3,
                         double stddev_min = 0.01,
                         double stddev_max = 0.5,
                         double stddev_increase_factor = 1.5,
                         double stddev_decay_factor = 0.95,
                         double elite_fraction = 0.3,
                         int sample_count = 50,
                         int check_points = 50,
                         int gd_iterations = 0,           // ignored (CES only)
                         int init_points = 3,
                         double collision_weight = 1.0,
                         double z_min = 0.0,
                         Point limits_min = -Point::Ones()*2.0,
                         Point limits_max =  Point::Ones()*2.0,
                         bool enable_gradient_descent = false, // ignored
                         double sigma_floor = 0.0,
                         double var_ema_beta = 0.2,
                         double mean_lr = 0.5,
                         double max_step_norm = 0.1,          // ignored
                         double floor_margin = 0.01,
                         double floor_penalty_scale = 10.0)
        {
            PlannerConfig cfg;
            cfg.samples        = sample_count;
            cfg.checks         = check_points;
            cfg.total_points    = init_points;
            cfg.w_collision    = collision_weight;
            cfg.elite_fraction = elite_fraction;
            cfg.inc            = stddev_increase_factor;
            cfg.dec            = stddev_decay_factor;
            cfg.sigma_floor    = sigma_floor;
            cfg.var_beta       = var_ema_beta;
            cfg.mean_lr        = mean_lr;
            cfg.stddev_min     = stddev_min;
            cfg.stddev_max     = stddev_max;
            cfg.z_min          = z_min;
            cfg.floor_margin   = floor_margin;
            cfg.floor_scale    = floor_penalty_scale;
            cfg.verbose        = false;

            planner_ = std::make_unique<Planner>(model, body_name, cfg, limits_min, limits_max, stddev_initial);
            limits_min_ = limits_min; limits_max_ = limits_max;
        }

        // keep previous behavior
        std::vector<PathCandidate> plan(const Point& start, const Point& end, bool iterate_flag=false) {
            return planner_->plan(start, end, iterate_flag);
        }

        // getters expected by your UI/bench
        std::vector<PathCandidate> get_succesful_path_candidates() { return planner_->successes(); }
        std::vector<PathCandidate> get_failed_path_candidates()    { return planner_->failures();  }
        const std::vector<tsp::ViaSet>& get_sampled_via_sets() const { return planner_->sampled_sets(); }
        
        std::vector<tsp::Point> get_sampled_via_pts() const {
            std::vector<tsp::Point> out;
            for (const auto& S : planner_->sampled_sets()) {
                out.push_back(S.empty() ? tsp::Point::Zero() : S.front());
            }
            return out;
        }

        const std::vector<Point>& get_via_pts() const              { return planner_->initial_via(); }

        Point get_current_mean() const   { return planner_->mean();  }
        Point get_current_stddev() const { return planner_->sigma(); }

        Point get_limits_min() const { return limits_min_; }
        Point get_limits_max() const { return limits_max_; }

        static Point evaluate(double u, const Spline& s) { return s(u); }
        Point evaluate(double u) const { return planner_->spline()(u); }

        std::vector<Point> get_path_pts(int n=10) const { return planner_->get_path_pts(n); }

        typename Spline::ControlPointVectorType get_ctrl_pts() const { return planner_->spline().ctrls(); }
        typename Spline::KnotVectorType         get_knot_vector() const { return planner_->spline().knots(); }

        // verbosity passthrough
        void set_verbose(bool on){ /* planner is quiet by default; add flag if needed */ (void)on; }

        tsp::Spline spline_from_via(const tsp::Point& via) const {
            return planner_->spline_from_via(via);
        }

        tsp::Spline spline_from_vias(const ViaSet& vias) const {
            return planner_->spline_from_vias(vias);
        }

// Back-compat: old code calls reset() before plan(); no-op is fine here
        void reset() {}


    private:
        std::unique_ptr<Planner> planner_;
        Point limits_min_, limits_max_;
    };

} // namespace tsp
