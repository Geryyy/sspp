#!/usr/bin/env python3
import argparse
import importlib.util
import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from vpsto.vpsto import VPSTO, VPSTOOptions
import mujoco as mj


# ---------------------------
# Utilities
# ---------------------------

@dataclass
class Stats:
    mean_ms: float = 0.0
    std_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0


def compute_stats(samples_ms: List[float]) -> Stats:
    if not samples_ms:
        return Stats()
    arr = np.asarray(samples_ms, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std(ddof=0))
    return Stats(mean_ms=mean, std_ms=std, min_ms=float(arr.min()), max_ms=float(arr.max()))


def load_hooks_module(hooks_path: str):
    spec = importlib.util.spec_from_file_location("user_hooks", hooks_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load hooks module from {hooks_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # required symbols
    for required in ("collision_penalty", "fk_point"):
        if not hasattr(mod, required):
            raise AttributeError(
                f"Hooks module must define `{required}`. "
                f"Missing in {hooks_path}."
            )
    return mod


def path_length_xyz(fk_point_fn, mj_model, q_traj: np.ndarray, n_samples: int = 50) -> float:
    """
    Approximate task-space path length by sampling the end-effector/body point along q_traj.
    q_traj: (T, ndof)
    """
    if q_traj.shape[0] < 2:
        return 0.0

    # If vpsto gives dense T points already, downsample to ~n_samples; else use as-is
    idx = np.linspace(0, q_traj.shape[0] - 1, num=min(n_samples, q_traj.shape[0]), dtype=int)
    pts = np.asarray([fk_point_fn(q_traj[i], mj_model) for i in idx], dtype=float)
    if pts.shape[1] >= 3:
        pts = pts[:, :3]
    dist = np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()
    return float(dist)


# ---------------------------
# Main benchmark
# ---------------------------

def make_loss(hooks_mod, vpsto, mj_model, lam_coll: float):
    """
    Returns a loss(candidates) function compatible with VPSTO:
    candidates: dict with keys 'pos' -> list of (T, ndof) arrays, 'T' -> list of scalars
    We use: cost_i = T_i + lam_coll * collision_penalty_i
    """
    def loss(candidates: dict) -> np.ndarray:
        # candidates['T'] is (pop_size,) array-like
        T = np.asarray(candidates["T"], dtype=float)
        # user-provided penalty per candidate (0 if feasible)
        penalties = hooks_mod.collision_penalty(candidates, mj_model)
        if penalties.shape != T.shape:
            raise ValueError(f"collision_penalty must return shape {T.shape}, got {penalties.shape}")
        return T + lam_coll * penalties
    return loss


def run_one_solve(vpsto, loss_fn, q0, qT, dqT=None):
    if dqT is None:
        dqT = np.zeros_like(q0)
    sol = vpsto.minimize(loss_fn, q0=q0, qT=qT, dqT=dqT)
    return sol


def collect_path_length_for_solution(hooks_mod, vpsto, mj_model, q_via_best, q0, qT) -> float:
    """
    Uses VPSTO's internal trajectory generator to get the time-parameterized joint path
    and then integrates task-space length of fk_point along it.
    """
    q_traj, _, _ = vpsto.vptraj.get_trajectory(q_via_best, q0, qT=qT, dqT=np.zeros_like(q0))
    q_traj = np.squeeze(q_traj, axis=0)  # (T, ndof)
    return path_length_xyz(hooks_mod.fk_point, mj_model, q_traj, n_samples=100)


def benchmark_phase(label: str,
                    make_vpsto_callable,
                    loss_fn_factory,
                    mj_model,
                    hooks_mod,
                    q0: np.ndarray,
                    qT: np.ndarray,
                    N: int,
                    warm: bool) -> Tuple[Stats, int, float]:
    """
    Runs N solves in either cold or warm mode.
    Returns (time_stats, successes, avg_path_len).
    success criterion: final penalty == 0 (collision-free).
    """
    times_ms: List[float] = []
    successes = 0
    path_len_sum = 0.0

    if warm:
        # single persistent optimizer instance for all warm runs
        vpsto = make_vpsto_callable()
        loss_fn = loss_fn_factory(vpsto)
        for i in range(N):
            t0 = time.perf_counter()
            sol = run_one_solve(vpsto, loss_fn, q0, qT)
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)

            # success check via final candidate penalty
            final_penalty = float(np.min(sol.loss_list[-1])) if len(sol.loss_list) > 0 else np.inf
            if final_penalty < 1e-9:  # collision-penalty==0 and no extra penalties
                successes += 1
                q_via_best = sol.p_best
                path_len_sum += collect_path_length_for_solution(hooks_mod, vpsto, mj_model, q_via_best, q0, qT)
    else:
        # fresh optimizer each solve
        for i in range(N):
            vpsto = make_vpsto_callable()
            loss_fn = loss_fn_factory(vpsto)
            t0 = time.perf_counter()
            sol = run_one_solve(vpsto, loss_fn, q0, qT)
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)

            final_penalty = float(np.min(sol.loss_list[-1])) if len(sol.loss_list) > 0 else np.inf
            if final_penalty < 1e-9:
                successes += 1
                q_via_best = sol.p_best
                path_len_sum += collect_path_length_for_solution(hooks_mod, vpsto, mj_model, q_via_best, q0, qT)

    stats = compute_stats(times_ms)
    avg_path_len = (path_len_sum / successes) if successes > 0 else 0.0

    print(f"\n[{label}] N={N}")
    print(f"  Successes: {successes}/{N}")
    print(f"  Time mean={stats.mean_ms:.3f} ms  std={stats.std_ms:.3f} ms  "
          f"min={stats.min_ms:.3f} ms  max={stats.max_ms:.3f} ms")
    if successes > 0:
        print(f"  Avg path length: {avg_path_len:.4f} m")
    return stats, successes, avg_path_len


def main():
    ap = argparse.ArgumentParser(description="VP-STO benchmark (cold vs warm) with MuJoCo + user hooks.")
    ap.add_argument("--xml", required=True, help="MuJoCo XML file path")
    ap.add_argument("--hooks", required=True, help="Path to hooks python file providing collision_penalty() and fk_point()")
    ap.add_argument("--q0", required=True, help="Start joints as comma-separated floats, e.g. '0.1,0.2,0.0'")
    ap.add_argument("--qT", required=True, help="Goal joints as comma-separated floats")
    ap.add_argument("--N", type=int, default=50, help="Num runs per phase (cold/warm)")
    # VPSTO options
    ap.add_argument("--ndof", type=int, required=True, help="Number of DoF")
    ap.add_argument("--N_via", type=int, default=2, help="Number of via points")
    ap.add_argument("--N_eval", type=int, default=50, help="Trajectory evaluation points")
    ap.add_argument("--pop_size", type=int, default=100, help="Population size")
    ap.add_argument("--max_iter", type=int, default=100, help="Max VPSTO iterations")
    ap.add_argument("--sigma_init", type=float, default=8.0, help="Initial sampling variance (per VPSTO)")
    ap.add_argument("--lam_coll", type=float, default=1e3, help="Collision penalty weight in loss")
    args = ap.parse_args()

    # Load hooks
    hooks_mod = load_hooks_module(args.hooks)

    # Load MuJoCo model
    m = mj.MjModel.from_xml_path(args.xml)
    d = mj.MjData(m)  # not strictly needed but handy if your hooks use it

    # Parse joint vectors
    q0 = np.array([float(x) for x in args.q0.split(",")], dtype=float)
    qT = np.array([float(x) for x in args.qT.split(",")], dtype=float)
    if q0.shape[0] != args.ndof or qT.shape[0] != args.ndof:
        raise ValueError(f"q0 and qT must have ndof={args.ndof} elements.")

    # Factory to build VPSTO with provided options
    def make_vpsto():
        opt = VPSTOOptions(ndof=args.ndof)
        opt.N_via = args.N_via
        opt.N_eval = args.N_eval
        opt.pop_size = args.pop_size
        opt.max_iter = args.max_iter
        opt.sigma_init = args.sigma_init
        opt.log = False
        return VPSTO(opt)

    # Loss factory (binds lam_coll & hooks & model)
    def loss_factory(vpsto_inst):
        return make_loss(hooks_mod, vpsto_inst, m, lam_coll=args.lam_coll)

    # Phase A: cold (fresh optimizer each run)
    benchmark_phase("Cold start (iterate=false)",
                    make_vpsto_callable=make_vpsto,
                    loss_fn_factory=loss_factory,
                    mj_model=m,
                    hooks_mod=hooks_mod,
                    q0=q0, qT=qT,
                    N=args.N, warm=False)

    # Phase B: warm (reuse optimizer instance)
    benchmark_phase("Warm start (iterate=true)",
                    make_vpsto_callable=make_vpsto,
                    loss_fn_factory=loss_factory,
                    mj_model=m,
                    hooks_mod=hooks_mod,
                    q0=q0, qT=qT,
                    N=args.N, warm=True)


if __name__ == "__main__":
    main()
