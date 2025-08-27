#!/usr/bin/env python3
import argparse
import importlib.util
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import mujoco as mj
from vpsto.vpsto import VPSTO, VPSTOOptions


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
    for required in ("collision_penalty", "fk_point"):
        if not hasattr(mod, required):
            raise AttributeError(f"Hooks module must define `{required}`. Missing in {hooks_path}.")
    return mod


def path_length_xyz(fk_point_fn, mj_model, q_traj: np.ndarray, n_samples: int = 100) -> float:
    """
    Approximate task-space path length (xyz only) along a joint trajectory.
    q_traj: (T, ndof)
    """
    if q_traj.shape[0] < 2:
        return 0.0
    idx = np.linspace(0, q_traj.shape[0] - 1, num=min(n_samples, q_traj.shape[0]), dtype=int)
    pts = np.asarray([fk_point_fn(q_traj[i], mj_model) for i in idx], dtype=float)
    pts = pts[:, :3] if pts.shape[1] >= 3 else pts
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


def make_loss(hooks_mod, vpsto, mj_model, lam_coll: float):
    """
    VP-STO loss(candidates) factory.
    candidates: dict with keys 'T' (shape [pop]) and 'pos' (list of (T, ndof)).
    We use loss = T + lam_coll * collision_penalty.
    """
    def loss(candidates: dict) -> np.ndarray:
        T = np.asarray(candidates["T"], dtype=float)
        penalties = hooks_mod.collision_penalty(candidates, mj_model)
        if penalties.shape != T.shape:
            raise ValueError(f"collision_penalty must return shape {T.shape}, got {penalties.shape}")
        return T + lam_coll * penalties
    return loss


def collect_path_length_for_solution(hooks_mod, vpsto, mj_model, q_via_best, q0, qT) -> float:
    """
    Use VPSTO's trajectory generator to get dense q_traj and measure xyz length.
    """
    q_traj, _, _ = vpsto.vptraj.get_trajectory(q_via_best, q0, qT=qT, dqT=np.zeros_like(q0))
    q_traj = np.squeeze(q_traj, axis=0)  # (T, ndof)
    return path_length_xyz(hooks_mod.fk_point, mj_model, q_traj, n_samples=200)


# ---------------------------
# Core runners
# ---------------------------

def run_one_solve(vpsto: VPSTO, loss_fn, q0: np.ndarray, qT: np.ndarray):
    """Full solve to VP-STO's internal stopping criterion."""
    return vpsto.minimize(loss_fn, q0=q0, qT=qT, dqT=np.zeros_like(q0))


def run_one_solve_budgeted(vpsto: VPSTO,
                            loss_fn,
                            q0: np.ndarray,
                            qT: np.ndarray,
                            budget_ms: float,
                            chunk_iter: int = 5,
                            tff_recorder: Optional[List[float]] = None):
    """
    Anytime mode: repeatedly extend max_iter in small chunks until wall-clock budget is exhausted.
    - Tracks best-so-far solution by min(loss_list[-1]).
    - Optionally records time-to-first-feasible (tff_recorder append).
    Returns the final VPSTO solution object (with current p_best).
    """
    assert budget_ms > 0.0 and chunk_iter > 0
    t_deadline = time.perf_counter() + budget_ms / 1000.0
    best_loss = np.inf
    saw_feasible = False

    while time.perf_counter() < t_deadline:
        old_iter = getattr(vpsto, "iter", 0)
        old_max = vpsto.opt.max_iter
        vpsto.opt.max_iter = old_iter + chunk_iter
        sol = vpsto.minimize(loss_fn, q0=q0, qT=qT, dqT=np.zeros_like(q0))
        vpsto.opt.max_iter = old_max

        if len(sol.loss_list) > 0:
            current = float(np.min(sol.loss_list[-1]))
            # TFF capture
            if (not saw_feasible) and current < 1e-9:
                saw_feasible = True
                if tff_recorder is not None:
                    tff_recorder.append((budget_ms - max(0.0, (t_deadline - time.perf_counter()) * 1000.0)))
            # best-so-far
            if current < best_loss:
                best_loss = current

        # Stop if VPSTO didn't advance (defensive)
        if getattr(vpsto, "iter", 0) <= old_iter:
            break

    return sol

def is_solution_feasible(sol, hooks_mod, vpsto, mj_model, q0, qT, lam_coll, tol=1e-9):
    """
    Success criterion: collision penalty == 0 (feasible trajectory).
    """
    if not hasattr(sol, "p_best"):
        return False

    q_traj, _, _ = vpsto.vptraj.get_trajectory(sol.p_best, q0, qT=qT, dqT=np.zeros_like(q0))
    q_traj = np.squeeze(q_traj, axis=0)
    candidates = {"pos": [q_traj], "T": np.asarray([float(len(q_traj))])}
    penalties = hooks_mod.collision_penalty(candidates, mj_model)
    return float(np.asarray(penalties).ravel()[0]) <= tol


def benchmark_phase(label: str,
                    make_vpsto_callable,
                    loss_fn_factory,
                    mj_model,
                    hooks_mod,
                    q0: np.ndarray,
                    qT: np.ndarray,
                    N: int,
                    warm: bool,
                    lam_coll: float) -> Tuple[Stats, int, float]:
    """
    Run N solves in either cold (fresh solver) or warm (persistent solver) mode.
    Returns (time_stats, successes, avg_path_len).
    """
    import time
    import numpy as np

    times_ms: List[float] = []
    successes = 0
    path_len_sum = 0.0

    if warm:
        vpsto = make_vpsto_callable()
        loss_fn = loss_fn_factory(vpsto)
        for _ in range(N):
            t0 = time.perf_counter()
            sol = vpsto.minimize(loss_fn, q0=q0, qT=qT, dqT=np.zeros_like(q0))
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)

            if is_solution_feasible(sol, hooks_mod, vpsto, mj_model, q0, qT, lam_coll):
                successes += 1
                q_traj, _, _ = vpsto.vptraj.get_trajectory(sol.p_best, q0, qT=qT, dqT=np.zeros_like(q0))
                q_traj = np.squeeze(q_traj, axis=0)
                idx = np.linspace(0, len(q_traj) - 1, num=min(100, len(q_traj)), dtype=int)
                pts = np.asarray([hooks_mod.fk_point(q_traj[i], mj_model) for i in idx])[:, :3]
                path_len_sum += float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
    else:
        for _ in range(N):
            vpsto = make_vpsto_callable()
            loss_fn = loss_fn_factory(vpsto)
            t0 = time.perf_counter()
            sol = vpsto.minimize(loss_fn, q0=q0, qT=qT, dqT=np.zeros_like(q0))
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)

            if is_solution_feasible(sol, hooks_mod, vpsto, mj_model, q0, qT, lam_coll):
                successes += 1
                q_traj, _, _ = vpsto.vptraj.get_trajectory(sol.p_best, q0, qT=qT, dqT=np.zeros_like(q0))
                q_traj = np.squeeze(q_traj, axis=0)
                idx = np.linspace(0, len(q_traj) - 1, num=min(100, len(q_traj)), dtype=int)
                pts = np.asarray([hooks_mod.fk_point(q_traj[i], mj_model) for i in idx])[:, :3]
                path_len_sum += float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())

    stats = compute_stats(times_ms)
    avg_path_len = (path_len_sum / successes) if successes > 0 else 0.0

    print(f"\n[{label}] N={N}")
    print(f"  Successes: {successes}/{N}")
    print(f"  Mean time: {stats.mean_ms:.3f} ms, Std: {stats.std_ms:.3f} ms")
    print(f"  Min/Max  : {stats.min_ms:.3f} / {stats.max_ms:.3f} ms")
    if successes > 0:
        print(f"  Avg path length: {avg_path_len:.4f} m")

    return stats, successes, avg_path_len



# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser(
        description="VP-STO benchmark: converged (best-effort) and anytime (fixed budget) runs."
    )
    ap.add_argument("--xml", required=True, help="MuJoCo XML file path")
    ap.add_argument("--hooks", required=True, help="Path to hooks python file (collision_penalty, fk_point)")
    ap.add_argument("--q0", required=True, help="Start joints as comma-separated floats, e.g. '0.1,0.2,0.0,1.57'")
    ap.add_argument("--qT", required=True, help="Goal joints as comma-separated floats")
    ap.add_argument("--ndof", type=int, required=True, help="Number of DoF")
    ap.add_argument("--N", type=int, default=50, help="Num runs per phase")
    ap.add_argument("--N_via", type=int, default=2, help="Number of via points")
    ap.add_argument("--N_eval", type=int, default=50, help="Trajectory evaluation points")
    ap.add_argument("--pop_size", type=int, default=100, help="Population size")
    ap.add_argument("--max_iter", type=int, default=100, help="Max VPSTO iterations (converged mode)")
    ap.add_argument("--sigma_init", type=float, default=8.0, help="Initial sampling variance (VPSTO)")
    ap.add_argument("--lam_coll", type=float, default=1e3, help="Collision penalty weight in loss")

    # Anytime (budgeted) options
    ap.add_argument("--budgets_ms", type=str, default="",
                    help="Comma-separated budgets in ms for anytime runs, e.g. '20,50,100'. Empty=skip anytime.")
    ap.add_argument("--chunk_iter", type=int, default=5,
                    help="Iteration chunk size per budget step for anytime runs.")

    args = ap.parse_args()

    # Load hooks & model
    hooks_mod = load_hooks_module(args.hooks)
    m = mj.MjModel.from_xml_path(args.xml)
    _ = mj.MjData(m)

    # Parse joint vectors
    q0 = np.array([float(x) for x in args.q0.split(",")], dtype=float)
    qT = np.array([float(x) for x in args.qT.split(",")], dtype=float)
    if q0.shape[0] != args.ndof or qT.shape[0] != args.ndof:
        raise ValueError(f"q0 and qT must have ndof={args.ndof} elements.")

    # VPSTO factory
    def make_vpsto():
        opt = VPSTOOptions(ndof=args.ndof)
        opt.N_via    = args.N_via
        opt.N_eval   = args.N_eval
        opt.pop_size = args.pop_size
        opt.max_iter = args.max_iter
        opt.sigma_init = args.sigma_init
        opt.log = False
        return VPSTO(opt)

    def loss_factory(vpsto_inst):
        return make_loss(hooks_mod, vpsto_inst, m, lam_coll=args.lam_coll)

    # ---- Converged runs (cold & warm) ----
    benchmark_phase("Converged (cold)", make_vpsto, loss_factory, m, hooks_mod,
                    q0=q0, qT=qT, N=args.N, warm=False, budget_ms=0.0)

    benchmark_phase("Converged (warm)", make_vpsto, loss_factory, m, hooks_mod,
                    q0=q0, qT=qT, N=args.N, warm=True, budget_ms=0.0)

    # ---- Anytime runs (optional) ----
    if args.budgets_ms.strip():
        budgets = [float(x) for x in args.budgets_ms.split(",") if x.strip()]
        for B in budgets:
            benchmark_phase(f"Anytime budget {B:.0f} ms (cold)",
                            make_vpsto, loss_factory, m, hooks_mod,
                            q0=q0, qT=qT, N=args.N, warm=False, budget_ms=B, chunk_iter=args.chunk_iter)

            benchmark_phase(f"Anytime budget {B:.0f} ms (warm)",
                            make_vpsto, loss_factory, m, hooks_mod,
                            q0=q0, qT=qT, N=args.N, warm=True, budget_ms=B, chunk_iter=args.chunk_iter)


if __name__ == "__main__":
    main()
