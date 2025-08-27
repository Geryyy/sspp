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
    int samples{50}, checks{40}, init_points{3};
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
    Planner(mjModel* m, const std::string& body_name,
            const PlannerConfig& cfg,
            const Point& lo, const Point& hi,
            double sigma0)
        : world_(m, body_name),
          sampler_(lo, hi),
          cfg_(cfg)
    {
        dist_.lo = lo; dist_.hi = hi;
        dist_.z_min = cfg_.z_min;
        dist_.sigma_floor = cfg_.sigma_floor;
        dist_.var_ema_beta = cfg_.var_beta;
        dist_.mean_lr = cfg_.mean_lr;
        dist_.stddev_min = cfg_.stddev_min;
        dist_.stddev_max = cfg_.stddev_max;
        evaluator_.z_min = cfg_.z_min;
        evaluator_.floor_margin = cfg_.floor_margin;
        evaluator_.floor_scale = cfg_.floor_scale;
        sigma0_ = sigma0;
    }

    void reset(const Point& start, const Point& end) {
        world_.ensurePool();
        last_best_.reset();
        Point m0 = 0.5*(start + end);
        m0[2] = std::max(m0[2], cfg_.z_min);
        dist_.reset(m0, sigma0_);
        path_ = path_model_.initLinear(start, end, cfg_.init_points);
    }

    tsp::Spline spline_from_via(const tsp::Point& via) const {
        return path_model_.fromVia(via);
    }

    // iterate=false → new episode
    std::vector<PathCandidate> plan(const Point& start, const Point& end, bool iterate){
        world_.ensurePool();
        if (!iterate) reset(start, end);
        else          path_ = path_model_.initLinear(start, end, cfg_.init_points); // refresh linear base

        // build candidates
        std::vector<Point> seeds;
        seeds.reserve(cfg_.samples + 2);
        // mean
        Point mean = dist_.mean;
        mean[2] = std::max(mean[2], cfg_.z_min);
        seeds.push_back(mean);
        // forward best
        if (iterate && last_best_) seeds.push_back(*last_best_);
        // samples
        for (int i=0;i<cfg_.samples;++i) {
            Point c = sampler_.sample(dist_.mean, dist_.sigma);
            c[2] = std::max(c[2], cfg_.z_min);
            seeds.push_back(c);
        }
        sampled_ = seeds;

        // evaluate (OpenMP)
        successes_.clear(); failures_.clear();
#pragma omp parallel
        {
            std::vector<PathCandidate> succ_local, fail_local;
            succ_local.reserve(cfg_.samples/omp_get_max_threads()+4);
            fail_local.reserve(cfg_.samples/omp_get_max_threads()+4);

#pragma omp for schedule(static) nowait
            for (int i=0; i<(int)seeds.size(); ++i) {
                const Point& v = seeds[i];
                const Spline s = path_model_.fromVia(v);

                double L=0.0, Cnf=0.0, Cwf=0.0;
                evaluator_.eval_one_pass(s, cfg_.checks, world_, L, Cnf, Cwf);

                if (Cnf == 0.0) {
                    PathCandidate c{v};
                    c.status = SolverStatus::Converged;
                    c.L = L; c.C_nf = Cnf; c.C_wf = Cwf;
                    succ_local.emplace_back(std::move(c));
                } else {
                    PathCandidate c{v};
                    c.status = SolverStatus::Failed;
                    fail_local.emplace_back(std::move(c));
                }
            }

#pragma omp critical
            {
                successes_.insert(successes_.end(), succ_local.begin(), succ_local.end());
                failures_.insert(failures_.end(),  fail_local.begin(),  fail_local.end());
            }
        }

        // update & select
        if (!successes_.empty()){
            auto cost = [&](const PathCandidate& c){ return c.L + cfg_.w_collision * c.C_wf; };
            const auto elites_idx = elites_.select(successes_, cfg_.elite_fraction, cost);
            const auto w          = elites_.weights((int)elites_idx.size());

            dist_.update(successes_, elites_idx, w,
                         [](const PathCandidate& c){ return c.refined ? *c.refined : c.via; });

            // pick best
            auto best_it = std::min_element(successes_.begin(), successes_.end(),
                [&](const PathCandidate& a, const PathCandidate& b){ return cost(a) < cost(b); });
            if (best_it != successes_.end()) {
                last_best_ = best_it->refined ? *best_it->refined : best_it->via;
                path_ = path_model_.fromVia(*last_best_);
            }
            dist_.adapt(true, cfg_.inc, cfg_.dec);
        } else {
            dist_.adapt(false, cfg_.inc, cfg_.dec);
        }
        return successes_;
    }

    // Accessors for UI/bench
    const std::vector<PathCandidate>& successes() const { return successes_; }
    const std::vector<PathCandidate>& failures()  const { return failures_;  }
    const std::vector<Point>& sampled() const { return sampled_; }
    const std::vector<Point>& initial_via() const { return path_model_.initial_via_points(); }
    const Point& mean() const { return dist_.mean; }
    const Point& sigma() const { return dist_.sigma; }
    const Spline& spline() const { return path_; }

    std::vector<Point> get_path_pts(int N=10) const {
        std::vector<Point> pts; pts.reserve(N);
        for (int i=0;i<N;++i) pts.push_back(path_(double(i)/(N-1)));
        return pts;
    }

private:
    CollisionWorld world_;
    Sampler   sampler_;
    Evaluator evaluator_;
    EliteSelector elites_;
    Distribution  dist_;
    PlannerConfig cfg_;
    PathModel     path_model_;
    Spline        path_;
    std::optional<Point> last_best_;
    double sigma0_{0.3};

    std::vector<PathCandidate> successes_, failures_;
    std::vector<Point> sampled_;
};

} // namespace tsp
