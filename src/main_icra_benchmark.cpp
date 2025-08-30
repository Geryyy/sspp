// main_bench.cpp — compact benchmark for modular C++ TaskSpacePlanner
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <string>
#include <tuple>
#include <vector>

#include <mujoco/mujoco.h>
#include <Eigen/Core>
#include <sspp/tsp.h>
#include "utility.h"

using Point = tsp::Point;
using Clock = std::chrono::high_resolution_clock;

// ---------- small utils ----------
struct Stats { double mean_ms{0}, std_ms{0}, min_ms{0}, max_ms{0}; };
static Stats stats(const std::vector<double>& ms){
    Stats s{}; if(ms.empty()) return s;
    const double n = double(ms.size());
    s.mean_ms = std::accumulate(ms.begin(), ms.end(), 0.0)/n;
    const double sq = std::inner_product(ms.begin(), ms.end(), ms.begin(), 0.0);
    s.std_ms  = std::sqrt(std::max(0.0, sq/n - s.mean_ms*s.mean_ms));
    const auto [mn,mx] = std::minmax_element(ms.begin(), ms.end());
    s.min_ms = *mn; s.max_ms = *mx; return s;
}
static double path_len_xyz(const tsp::TaskSpacePlanner& p, int samples=60){
    const auto pts = p.get_path_pts(samples); if(pts.size()<2) return 0.0;
    double L=0; for(size_t i=1;i<pts.size();++i) L += (pts[i]-pts[i-1]).head<3>().norm(); return L;
}

// ---------- runners ----------
static std::tuple<double,bool,double>
run_converged(tsp::TaskSpacePlanner& plan, const Point& q0, const Point& qT, int max_iter){
    const auto t0 = Clock::now();
    bool ok = !plan.plan(q0,qT,false).empty();
    for(int k=1;k<max_iter;++k) ok = ok || !plan.plan(q0,qT,true).empty();
    const double ms = std::chrono::duration<double,std::milli>(Clock::now()-t0).count();
    return {ms, ok, ok? path_len_xyz(plan): 0.0};
}


template<class Runner, class Maker, class... Args>
static std::tuple<Stats,int,double> trials(int N, bool warm, Runner run, Maker make, Args&&... args){
    std::vector<double> times; times.reserve(N);
    int succ=0; double sumL=0.0;
    if(warm){
        auto planner = make();
        for(int i=0;i<N;++i){ auto [ms,ok,L] = run(planner, std::forward<Args>(args)...);
            times.push_back(ms); if(ok){ ++succ; sumL += L; } }
    }else{
        for(int i=0;i<N;++i){ auto planner = make();
            auto [ms,ok,L] = run(planner, std::forward<Args>(args)...);
            times.push_back(ms); if(ok){ ++succ; sumL += L; } }
    }
    return {stats(times), succ, succ? (sumL/succ): 0.0};
}

// ---------- main ----------
int main(int argc, char** argv){
    // usage
    if(argc < 3){
        std::cerr << "Usage: " << argv[0]
                  << " <model.xml> <collision_body>"
                  << " [N=50] [start_body=block_green/] [end_body=block_orange/] [num_vias=1]\n";
        return 1;
    }
    const std::string model_xml     = argv[1];
    const std::string coll_body     = argv[2];
    int         N                   = (argc>=4 && argv[3][0]!='-')? std::max(1,std::atoi(argv[3])): 50;
    std::string start_body          = (argc>=5 && argv[4][0]!='-')? argv[4] : "block_green/";
    std::string end_body            = (argc>=6 && argv[5][0]!='-')? argv[5] : "block_orange/";
    int         num_vias            = (argc>=7 && argv[6][0]!='-')? std::max(0,std::atoi(argv[6])): 1;

    // mujoco
    char err[1024];
    mjModel* m = mj_loadXML(model_xml.c_str(), nullptr, err, sizeof(err));
    if(!m){ std::cerr << "MuJoCo load error: " << err << "\n"; return 2; }
    mjData* d = mj_makeData(m);
    mj_forward(m,d); mj_collision(m,d);

    // planner setup (matching your modular defaults)
    const double stddev_initial=0.2, stddev_min=1e-4, stddev_max=0.5;
    const double inc=1.5, dec=0.9, elite=0.3;
    const int samples=15, checks=40, gd_iters=0;
    const int total_points = num_vias + 2;
    const double w_col=1.0, z_min=0.1;
    const bool use_gd=false;
    const double sigma_floor=0.005, var_ema_beta=0.2, mean_lr=0.5, max_step_norm=0.1;
    const double floor_margin=0.01, floor_scale=10.0;
    Point lim_lo, lim_hi; lim_hi<<0.7,0.7,0.6, 1.6; lim_lo<<0.0,-0.7,0.1,-1.6;

    auto make_planner = [&](){
        return tsp::TaskSpacePlanner(
            m, coll_body,
            stddev_initial, stddev_min, stddev_max,
            inc, dec, elite, samples, checks,
            gd_iters, total_points, w_col, z_min,
            lim_lo, lim_hi, use_gd,
            sigma_floor, var_ema_beta, mean_lr, max_step_norm,
            floor_margin, floor_scale
        );
    };

    // start/end points
    const auto info = Utility::get_free_body_joint_info(coll_body, m);
    (void)info; // not used in benchmark
    Point q0 = Utility::get_body_point<Point>(m,d,start_body);
    Point qT = Utility::get_body_point<Point>(m,d,end_body);
    q0[2]+=0.02; qT[2]+=0.02;

    // --------- iteration benchmarks ---------
    {
        std::vector<int> iters = {1, 2, 5, 10};
        std::cout << "\n=== Iteration Benchmark (N=" << N << ") ===\n";
        for(int k : iters){
            const auto [S, n, L] = trials(N, /*warm=*/false, run_converged, make_planner, q0, qT, k);
            std::cout << "[iter=" << k << "] succ " << n << "/" << N
                      << "  mean " << S.mean_ms << " ms  std " << S.std_ms
                      << "  min " << S.min_ms << "  max " << S.max_ms;
            if(n) std::cout << "  avg length " << L << " m"; std::cout << "\n";
        }
    }

    mj_deleteData(d); mj_deleteModel(m);
    return 0;
}
