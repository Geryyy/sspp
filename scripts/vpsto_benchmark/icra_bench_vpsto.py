#!/usr/bin/env python3
import argparse, importlib.util, time
from dataclasses import dataclass
from typing import List, Tuple, Callable
import numpy as np
import mujoco as mj
from vpsto.vpsto import VPSTO, VPSTOOptions


# ---------- small utils ----------
@dataclass
class Stats:
    mean_ms: float = 0.0
    std_ms:  float = 0.0
    min_ms:  float = 0.0
    max_ms:  float = 0.0

def stats(ms: List[float]) -> Stats:
    if not ms: return Stats()
    a = np.asarray(ms, float)
    return Stats(float(a.mean()), float(a.std()), float(a.min()), float(a.max()))

def load_hooks(path: str):
    spec = importlib.util.spec_from_file_location("user_hooks", path)
    if not spec or not spec.loader:
        raise RuntimeError(f"Failed to load hooks from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name in ("collision_penalty", "fk_point"):
        if not hasattr(mod, name):
            raise AttributeError(f"hooks module must define `{name}`")
    return mod

def path_len_xyz(fk_point: Callable, mj_model: mj.MjModel, q_traj: np.ndarray, samples: int = 200) -> float:
    if q_traj.shape[0] < 2: return 0.0
    idx = np.linspace(0, q_traj.shape[0]-1, num=min(samples, q_traj.shape[0]), dtype=int)
    pts = np.asarray([fk_point(q_traj[i], mj_model) for i in idx], float)
    pts = pts[:, :3] if pts.shape[1] >= 3 else pts
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())

def make_loss(hooks, mj_model, lam_coll: float):
    # loss(candidates): T + λ * collision_penalty
    def loss(candidates: dict) -> np.ndarray:
        T = np.asarray(candidates["T"], float)
        pen = np.asarray(hooks.collision_penalty(candidates, mj_model), float)
        if pen.shape != T.shape:
            raise ValueError(f"collision_penalty shape {pen.shape} != T shape {T.shape}")
        return T + lam_coll * pen
    return loss

def traj_from_vias(vpsto: VPSTO, q_via_best: np.ndarray, q0: np.ndarray, qT: np.ndarray) -> np.ndarray:
    q_traj, _, _ = vpsto.vptraj.get_trajectory(q_via_best, q0, qT=qT, dqT=np.zeros_like(q0))
    return np.squeeze(q_traj, axis=0)  # (T, ndof)

def feasible_and_length(hooks, mj_model, vpsto: VPSTO, q_via_best: np.ndarray, q0: np.ndarray, qT: np.ndarray) -> Tuple[bool, float]:
    q_traj = traj_from_vias(vpsto, q_via_best, q0, qT)
    cand = {"pos": [q_traj], "T": np.asarray([float(len(q_traj))])}
    pen = float(np.asarray(hooks.collision_penalty(cand, mj_model)).ravel()[0])
    if pen <= 1e-9:
        return True, path_len_xyz(hooks.fk_point, mj_model, q_traj)
    return False, 0.0


# ---------- runners ----------
def run_converged(vpsto: VPSTO, loss_fn, q0: np.ndarray, qT: np.ndarray):
    t0 = time.perf_counter()
    sol = vpsto.minimize(loss_fn, q0=q0, qT=qT, dqT=np.zeros_like(q0))
    ms = (time.perf_counter() - t0) * 1000.0
    ok, L = feasible_and_length(hooks_mod, mj_model, vpsto, sol.p_best, q0, qT)
    return ms, ok, L

def run_anytime(vpsto: VPSTO, loss_fn, q0: np.ndarray, qT: np.ndarray, budget_ms: float, chunk_iter: int = 5):
    t_end = time.perf_counter() + budget_ms / 1000.0
    best_L = np.inf; saw_ok = False
    # start from current iter, extend in small chunks until time is up
    while time.perf_counter() < t_end:
        cur_iter = getattr(vpsto, "iter", 0)
        vpsto.opt.max_iter = cur_iter + chunk_iter
        sol = vpsto.minimize(loss_fn, q0=q0, qT=qT, dqT=np.zeros_like(q0))
        ok, L = feasible_and_length(hooks_mod, mj_model, vpsto, sol.p_best, q0, qT)
        if ok:
            saw_ok = True
            if L < best_L: best_L = L
        # defensive: if optimizer didn't move forward, bail
        if getattr(vpsto, "iter", 0) <= cur_iter: break
    used_ms = (t_end - (t_end - time.perf_counter())) * 1000.0  # approx actual
    return used_ms, saw_ok, (best_L if saw_ok else 0.0)


# ---------- main ----------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="VP-STO benchmark: Converged and Anytime (fixed budgets)")
    ap.add_argument("--xml",   required=True)
    ap.add_argument("--hooks", required=True)
    ap.add_argument("--ndof",  type=int, required=True)
    ap.add_argument("--q0",    required=True, help="comma-separated, e.g. 0.5,0.1,0.12,1.57")
    ap.add_argument("--qT",    required=True)

    ap.add_argument("--N",        type=int,   default=50)
    ap.add_argument("--N_via",    type=int,   default=2)
    ap.add_argument("--N_eval",   type=int,   default=50)
    ap.add_argument("--pop_size", type=int,   default=100)
    ap.add_argument("--max_iter", type=int,   default=100)
    ap.add_argument("--sigma_init", type=float, default=8.0)
    ap.add_argument("--lam_coll", type=float, default=1e3)

    ap.add_argument("--budgets_ms", type=str, default="", help="e.g. '10,20,50' (optional)")
    ap.add_argument("--chunk_iter", type=int, default=5)
    args = ap.parse_args()

    # hooks & model
    hooks_mod = load_hooks(args.hooks)
    mj_model  = mj.MjModel.from_xml_path(args.xml)
    _         = mj.MjData(mj_model)

    # joints
    q0 = np.array([float(x) for x in args.q0.split(",")], float)
    qT = np.array([float(x) for x in args.qT.split(",")], float)
    if q0.size != args.ndof or qT.size != args.ndof:
        raise ValueError(f"q0 and qT must have ndof={args.ndof}")

    # VPSTO factory + loss
    def make_vpsto():
        opt = VPSTOOptions(ndof=args.ndof)
        opt.N_via = args.N_via; opt.N_eval = args.N_eval
        opt.pop_size = args.pop_size; opt.max_iter = args.max_iter
        opt.sigma_init = args.sigma_init; opt.log = False
        return VPSTO(opt)

    loss_factory = lambda v: make_loss(hooks_mod, mj_model, args.lam_coll)

    # -------- Converged: cold vs warm --------
    for label, warm in (("Converged (cold)", False), ("Converged (warm)", True)):
        times, succ, sumL = [], 0, 0.0
        if warm:
            vpsto = make_vpsto(); loss_fn = loss_factory(vpsto)
            for _ in range(args.N):
                ms, ok, L = run_converged(vpsto, loss_fn, q0, qT)
                times.append(ms); succ += int(ok); sumL += L
        else:
            for _ in range(args.N):
                vpsto = make_vpsto(); loss_fn = loss_factory(vpsto)
                ms, ok, L = run_converged(vpsto, loss_fn, q0, qT)
                times.append(ms); succ += int(ok); sumL += L
        S = stats(times)
        print(f"\n[{label}] N={args.N}")
        print(f"  Successes: {succ}/{args.N}")
        print(f"  Time mean={S.mean_ms:.3f} ms  std={S.std_ms:.3f} ms  min={S.min_ms:.3f}  max={S.max_ms:.3f} ms")
        if succ: print(f"  Avg path length: {sumL/succ:.4f} m")

    # -------- Anytime (fixed budgets, optional) --------
    if args.budgets_ms.strip():
        budgets = [float(x) for x in args.budgets_ms.split(",") if x.strip()]
        print(f"\n=== Anytime (budgets ms: {','.join(str(int(b)) for b in budgets)}, N={args.N}) ===")
        for B in budgets:
            for label, warm in ((f"Budget {int(B)} ms (cold)", False), (f"Budget {int(B)} ms (warm)", True)):
                times, succ, sumL = [], 0, 0.0
                if warm:
                    vpsto = make_vpsto(); loss_fn = loss_factory(vpsto)
                    for _ in range(args.N):
                        ms, ok, L = run_anytime(vpsto, loss_fn, q0, qT, B, args.chunk_iter)
                        times.append(ms); succ += int(ok); sumL += L
                else:
                    for _ in range(args.N):
                        vpsto = make_vpsto(); loss_fn = loss_factory(vpsto)
                        ms, ok, L = run_anytime(vpsto, loss_fn, q0, qT, B, args.chunk_iter)
                        times.append(ms); succ += int(ok); sumL += L
                S = stats(times)
                print(f"  [{label}]  succ {succ}/{args.N}"
                      f"  mean {S.mean_ms:.3f} ms  std {S.std_ms:.3f}  min {S.min_ms:.3f}  max {S.max_ms:.3f} ms"
                      f"{'' if not succ else f'  avgL {sumL/succ:.4f} m'}")
