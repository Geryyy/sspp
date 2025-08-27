#pragma once
#include "tsp_types.h"
#include "tsp_sampler.h"
#include "tsp_evaluator.h"
#include "tsp_elites.h"
#include "tsp_distribution.h"
#include "tsp_collision_world.h"
#include "tsp_path_model.h"
#include <algorithm>
#include <numeric>

namespace tsp {

struct PlannerConfig {
    // sampling / checks
    int samples{50}, checks{40}, total_points{3};
    int num_vias() const { return std::max(0, total_points - 2); }
    // costs
    double w_collision{1.0};
    // CES
    double elite_fraction{0.3};
    double inc{1.5}, dec{0.95};
    // distribution limits
    double sigma_floor{0.0}, var_beta{0.2}, mean_lr{0.5};
    double stddev_min{1e-3}, stddev_max{0.5};
    // floor
    double z_min{0.0}, floor_margin{0.01}, floor_scale{10.0};
    bool verbose{false};
};

    class Planner {
    public:
        Planner(mjModel* model,
                const std::string& body_name,
                const PlannerConfig& cfg,
                const Point& limits_min,
                const Point& limits_max,
                double z_min)
                : world_(model, body_name),
                  sampler_(limits_min, limits_max),
                  cfg_(cfg) {
            // Wire limits & knobs into distribution
            dist_.lo           = limits_min;
            dist_.hi           = limits_max;
            dist_.z_min        = z_min;
            dist_.sigma_floor  = cfg_.sigma_floor;
            dist_.var_ema_beta = cfg_.var_beta;
            dist_.mean_lr      = cfg_.mean_lr;
            dist_.stddev_min   = cfg_.stddev_min;
            dist_.stddev_max   = cfg_.stddev_max;
        }

        // Reset distribution and path to a fresh linear initialization
        void reset(const Point& start, const Point& end) {
            world_.ensurePool();
            last_best_.reset();

            path_ = path_model_.initLinear(start, end, cfg_.total_points);
            const int K = cfg_.num_vias();

            ViaSet mean0; mean0.reserve(K);
            const auto& init_via = path_model_.initial_via_points(); // size = total_points
            for (int i = 0; i < K; ++i) {
                Point v = init_via[1 + i];
                v[2] = std::max(v[2], cfg_.z_min);
                mean0.push_back(v);
            }
            dist_.reset(mean0, sigma0_);
        }

        // Main CES step
        std::vector<PathCandidate> plan(const Point& start, const Point& end, bool iterate) {
            world_.ensurePool();
            if (!iterate) reset(start, end);
            else          path_ = path_model_.initLinear(start, end, cfg_.total_points);

            // ---- Seed candidate via-sets ----
            std::vector<ViaSet> seeds;
            seeds.reserve(cfg_.samples + 2);

            // mean set
            ViaSet mean_set = dist_.mean_vias;
            for (auto& v : mean_set) v[2] = std::max(v[2], cfg_.z_min);
            seeds.push_back(mean_set);

            // forwarded best
            if (iterate && last_best_) seeds.push_back(*last_best_);

            // random samples
            for (int i = 0; i < cfg_.samples; ++i)
                seeds.push_back(sampler_.sample_set(dist_.mean_vias, dist_.sigma_vias, cfg_.z_min));

            sampled_sets_ = seeds;

            // ---- Evaluate candidates (OMP) ----
            successes_.clear(); failures_.clear();
#pragma omp parallel
            {
                std::vector<PathCandidate> succ_local, fail_local;
#pragma omp for schedule(static) nowait
                for (int i = 0; i < (int)seeds.size(); ++i) {
                    const ViaSet& vs = seeds[i];
                    const Spline s = path_model_.fromVias(vs);

                    double L = 0.0, Cnf = 0.0, Cwf = 0.0;
                    evaluator_.eval_one_pass(s, cfg_.checks, world_, L, Cnf, Cwf);

                    PathCandidate c;
                    c.via = vs; c.L = L; c.C_nf = Cnf; c.C_wf = Cwf;
                    c.status = (Cnf == 0.0) ? SolverStatus::Converged : SolverStatus::Failed;

                    (c.status == SolverStatus::Converged ? succ_local : fail_local).emplace_back(std::move(c));
                }
#pragma omp critical
                {
                    successes_.insert(successes_.end(), succ_local.begin(), succ_local.end());
                    failures_.insert(failures_.end(),  fail_local.begin(),  fail_local.end());
                }
            }

            // ---- Update distribution, pick best, forward it ----
            if (!successes_.empty()) {
                auto cost = [&](const PathCandidate& c){ return c.L + cfg_.w_collision * c.C_wf; };

                const auto elites_idx = elites_.select(successes_, cfg_.elite_fraction, cost);
                const auto w          = elites_.weights((int)elites_idx.size());

                dist_.update(successes_, elites_idx, w,
                             [](const PathCandidate& c){ return c.refined ? *c.refined : c.via; });

                auto best_it = std::min_element(successes_.begin(), successes_.end(),
                                                [&](const PathCandidate& a, const PathCandidate& b){
                                                    return cost(a) < cost(b);
                                                });
                if (best_it != successes_.end()) {
                    last_best_ = best_it->refined ? *best_it->refined : best_it->via;
                    path_ = path_model_.fromVias(*last_best_);
                }
                dist_.adapt(true, cfg_.inc, cfg_.dec);
            } else {
                dist_.adapt(false, cfg_.inc, cfg_.dec);
            }

            return successes_;
        }

        // --- Helpers / accessors ---
        Spline spline_from_vias(const ViaSet& vias) const { return path_model_.fromVias(vias); }
        Spline spline_from_via(const Point& via) const { return path_model_.fromVias(ViaSet{via}); }

        const std::vector<PathCandidate>& successes() const { return successes_; }
        const std::vector<PathCandidate>& failures()  const { return failures_;  }
        const std::vector<ViaSet>&        sampled_sets() const { return sampled_sets_; }
        const std::vector<Point>&         initial_via()  const { return path_model_.initial_via_points(); }

        // Back-compat: expose mean/sigma of the FIRST via (if any)
        const Point& mean() const  { static const Point Z = Point::Zero(); return dist_.mean_vias.empty()  ? Z : dist_.mean_vias.front(); }
        const Point& sigma() const { static const Point Z = Point::Zero(); return dist_.sigma_vias.empty() ? Z : dist_.sigma_vias.front(); }
        const Spline& spline() const { return path_; }

        std::vector<Point> get_path_pts(int N = 10) const {
            std::vector<Point> pts; pts.reserve(N);
            for (int i = 0; i < N; ++i) pts.push_back(path_(double(i)/(N-1)));
            return pts;
        }

    private:
        CollisionWorld  world_;
        Sampler         sampler_;
        Evaluator       evaluator_;
        EliteSelector   elites_;
        Distribution    dist_;
        PlannerConfig   cfg_;
        PathModel       path_model_;
        Spline          path_;

        double sigma0_{0.3};  // initial sigma for all vias
        std::vector<PathCandidate> successes_, failures_;
        std::optional<ViaSet>      last_best_;
        std::vector<ViaSet>        sampled_sets_;
    };


} // namespace tsp
