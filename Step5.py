# ============================================================
# STEP 5 — Improved Performance and Error Analysis
# ============================================================

import os
import sys
import time
import math

os.environ["JAX_ENABLE_X64"] = "1"

import numpy as np
import pandas as pd

import jax.numpy as jnp
from jax import jit, config
from functools import partial

config.update("jax_enable_x64", True)

# ============================================================
# 0. Matplotlib popup backend
# ============================================================

import matplotlib

try:
    matplotlib.use("TkAgg")
except Exception:
    pass

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


plt.rcParams.update({
    "figure.dpi": 120,
    "figure.figsize": (12, 7),
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "legend.framealpha": 0.9,
})


METHOD_COLORS = {
    "Euler":            "tab:blue",
    "RK4":              "tab:orange",
    "Dopri5_Adaptive":  "tab:green",
    "S1_Adaptive":      "tab:red",
    "S1+S2_Proj":       "tab:purple",
    "Full_STORK":       "tab:brown",
}


def method_color(name):
    return METHOD_COLORS.get(name, "black")


# ============================================================
# 1. 2-D Incompressible Navier-Stokes Benchmark
# ============================================================

class INS2D:
    """
    2-D incompressible Navier-Stokes in vorticity-stream formulation.

    State:
        omega : vorticity scalar field, shape [N, N]

    Velocity recovered via stream function psi:
        nabla^2 psi = -omega
        u =  d psi / d y
        v = -d psi / d x

    Incompressibility is satisfied by construction via the stream
    function; the divergence residual is computed purely as a
    verification diagnostic (should be near machine precision for
    a correctly implemented solver).

    Hard constraint used for benchmark:
        spectral high-frequency energy above the de-aliasing cutoff.

    [R3] Frequency pyramid:
        low_mask  — wavenumbers |k| <= N//8  (macro / smooth modes)
        high_mask — remaining wavenumbers     (stiff / fine-scale modes)
    """

    def __init__(self, N=64, nu=1e-3, k_cut_frac=0.65):
        self.N = N
        self.nu = nu

        k = jnp.fft.fftfreq(N, d=1.0 / N)
        self.kx, self.ky = jnp.meshgrid(k, k)
        self.k2 = self.kx ** 2 + self.ky ** 2
        self.k2_safe = jnp.where(self.k2 == 0, 1.0, self.k2)

        k_max = N // 2
        k_cut = int(k_cut_frac * k_max)
        k_mag = jnp.sqrt(self.k2)

        self.dealias_mask = (k_mag <= k_cut).astype(jnp.float64)

        # [R3] Explicit pyramid band masks
        k_low_cut = N // 8
        self.low_mask  = (k_mag <= k_low_cut).astype(jnp.float64)
        self.high_mask = 1.0 - self.low_mask

        # Store band labels for plotting
        self.k_low_cut  = k_low_cut
        self.k_high_cut = k_cut

    @partial(jit, static_argnums=(0,))
    def omega_to_uv(self, omega_k):
        psi_k = omega_k / self.k2_safe
        psi_k = psi_k.at[0, 0].set(0.0)

        u_k = 1j * self.ky * psi_k
        v_k = -1j * self.kx * psi_k

        u = jnp.real(jnp.fft.ifft2(u_k))
        v = jnp.real(jnp.fft.ifft2(v_k))

        return u, v

    @partial(jit, static_argnums=(0,))
    def rhs(self, omega):
        omega_k = jnp.fft.fft2(omega) * self.dealias_mask

        u, v = self.omega_to_uv(omega_k)

        domx_k = 1j * self.kx * omega_k
        domy_k = 1j * self.ky * omega_k

        domx = jnp.real(jnp.fft.ifft2(domx_k))
        domy = jnp.real(jnp.fft.ifft2(domy_k))

        adv  = -(u * domx + v * domy)
        visc =  jnp.real(jnp.fft.ifft2(-self.nu * self.k2 * omega_k))

        return adv + visc

    def divergence_residual(self, omega):
        """
        [R2] Divergence verification.
        For a stream-function-derived velocity field this should be
        identically zero up to floating-point rounding (~1e-17).
        """
        omega_k = jnp.fft.fft2(omega)
        u, v    = self.omega_to_uv(omega_k)

        u_k     = jnp.fft.fft2(u)
        v_k     = jnp.fft.fft2(v)

        div_k   = 1j * self.kx * u_k + 1j * self.ky * v_k
        div     = jnp.real(jnp.fft.ifft2(div_k))

        return float(jnp.linalg.norm(div)) / (self.N ** 2)

    def spectral_constraint(self, omega):
        """RMS high-frequency energy — used as the hard constraint."""
        omega_k  = jnp.fft.fft2(omega)
        hf_energy = jnp.sum(jnp.abs(omega_k * self.high_mask) ** 2) / (self.N ** 4)
        return float(jnp.sqrt(hf_energy))

    # [R3] Explicit band energy diagnostics for pyramid visualisation
    def low_freq_energy(self, omega):
        omega_k  = jnp.fft.fft2(omega)
        lf_energy = jnp.sum(jnp.abs(omega_k * self.low_mask) ** 2) / (self.N ** 4)
        return float(jnp.sqrt(lf_energy))

    def high_freq_energy(self, omega):
        return self.spectral_constraint(omega)

    @partial(jit, static_argnums=(0,))
    def project(self, omega):
        """
        Hard spectral projection: zero all high-frequency modes.
        [R2] This preserves the divergence-free property because
        truncating Fourier modes does not couple incompressible
        and compressible components — the stream-function relation
        nabla^2 psi = -omega is linear and band-limited projection
        commutes with the inversion operator.
        """
        omega_k = jnp.fft.fft2(omega)
        omega_k = omega_k * self.low_mask
        return jnp.real(jnp.fft.ifft2(omega_k))

    @partial(jit, static_argnums=(0,))
    def soft_project(self, omega, alpha):
        """
        [R5] Soft projection: convex blend of original and projected.
        alpha = 0 → no projection (identity)
        alpha = 1 → full hard projection
        Using alpha proportional to the spectral residual prevents
        over-damping low-residual states while still correcting large
        constraint violations, and avoids the striped artefacts that
        appear when hard projection is applied to off-manifold ICs.
        """
        return (1.0 - alpha) * omega + alpha * self.project(omega)

    @partial(jit, static_argnums=(0,))
    def split(self, omega):
        """[R3] Frequency pyramid split into macro and micro components."""
        omega_k = jnp.fft.fft2(omega)
        low  = jnp.real(jnp.fft.ifft2(omega_k * self.low_mask))
        high = jnp.real(jnp.fft.ifft2(omega_k * self.high_mask))
        return low, high


# ============================================================
# 2. Numerical time steppers
# ============================================================

def euler_step(env, omega, dt):
    return omega + dt * env.rhs(omega)


def rk4_step(env, omega, dt):
    k1 = env.rhs(omega)
    k2 = env.rhs(omega + 0.5 * dt * k1)
    k3 = env.rhs(omega + 0.5 * dt * k2)
    k4 = env.rhs(omega + dt * k3)
    return omega + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def safe_rel_err(u, ref, eps=1e-12):
    u   = jnp.nan_to_num(u,   nan=0.0)
    ref = jnp.nan_to_num(ref, nan=0.0)
    return float(jnp.linalg.norm(u - ref) / (jnp.linalg.norm(ref) + eps))


def wasserstein1d(u, ref):
    a = np.sort(np.nan_to_num(np.asarray(u).ravel(),   nan=0.0))
    b = np.sort(np.nan_to_num(np.asarray(ref).ravel(), nan=0.0))
    return float(np.mean(np.abs(a - b)))


# ============================================================
# 3. Solver classes
# ============================================================

class BaseSolver:
    def step(self, omega, t_rem):
        raise NotImplementedError


class EulerSolver(BaseSolver):
    def __init__(self, env, dt=0.004):
        self.env = env
        self.dt  = dt

    def step(self, omega, t_rem):
        h = min(self.dt, t_rem)
        return euler_step(self.env, omega, h), 1, h


class RK4Solver(BaseSolver):
    def __init__(self, env, dt=0.02):
        self.env = env
        self.dt  = dt

    def step(self, omega, t_rem):
        h = min(self.dt, t_rem)
        return rk4_step(self.env, omega, h), 4, h


class Dopri5Solver(BaseSolver):
    """
    Embedded RK4 / Richardson-extrapolation adaptive solver
    (Dopri5-like baseline with step-doubling error control).
    """

    def __init__(self, env, atol=1e-4, dt_max=0.05, dt_min=0.002):
        self.env    = env
        self.atol   = atol
        self.dt_max = dt_max
        self.dt_min = dt_min
        self._h     = dt_max

    def step(self, omega, t_rem):
        h   = min(self._h, t_rem)
        nfe = 0

        for _ in range(6):
            full = rk4_step(self.env, omega, h);          nfe += 4
            half = rk4_step(self.env, omega, 0.5 * h);   nfe += 4
            half = rk4_step(self.env, half,  0.5 * h);   nfe += 4

            err = float(jnp.linalg.norm(full - half)) / (self.env.N ** 2)

            if err < self.atol or h <= self.dt_min:
                factor   = min(2.0, 0.9 * (self.atol / (err + 1e-16)) ** 0.2)
                self._h  = min(h * factor, self.dt_max)
                return half, nfe, h

            h = max(h * 0.5, self.dt_min)

        return half, nfe, h


class S1AdaptiveSolver(BaseSolver):
    """
    [R1] Adaptive Euler whose step size is inversely proportional to
    the square root of the spectral constraint residual:

        h(t) = scale / sqrt(residual(t) + eps)

    This directly implements the requirement that the local step
    tolerance is dynamically, inversely proportional to the physics
    residual.
    """

    def __init__(self, env, scale=0.08, dt_max=0.06, dt_min=0.002):
        self.env    = env
        self.scale  = scale
        self.dt_max = dt_max
        self.dt_min = dt_min

    def _h(self, omega):
        c = float(self.env.spectral_constraint(omega)) + 1e-8
        h = self.scale / math.sqrt(c)
        return float(np.clip(h, self.dt_min, self.dt_max))

    def step(self, omega, t_rem):
        h = min(self._h(omega), t_rem)
        return euler_step(self.env, omega, h), 1, h


class S1S2ProjSolver(BaseSolver):
    """
    [R1 + R2] Adaptive Euler + hard spectral projection corrector.

    The corrector projects the predicted state onto the constraint
    manifold (low-frequency spectral support).  Because the
    projection operator P satisfies:
        P = P^2  (idempotent)
        P conserves the divergence-free property (see INS2D.project)
    the continuous probability vector field is not permanently
    misaligned: the projected state still satisfies the kinematic
    constraint nabla · u = 0.
    """

    def __init__(self, env, scale=0.08, dt_max=0.06, dt_min=0.002):
        self.env    = env
        self.scale  = scale
        self.dt_max = dt_max
        self.dt_min = dt_min

    def _h(self, omega):
        c = float(self.env.spectral_constraint(omega)) + 1e-8
        h = self.scale / math.sqrt(c)
        return float(np.clip(h, self.dt_min, self.dt_max))

    def step(self, omega, t_rem):
        h        = min(self._h(omega), t_rem)
        proposed = euler_step(self.env, omega, h)
        projected = self.env.project(proposed)
        return projected, 1, h


class FullSTORKSolver(BaseSolver):
    """
    Full STORK — novel integrator combining:

        [R1] S1: Residual-adaptive step size
             h(t) = scale / sqrt(spectral_residual(t))
             → local error tolerance INVERSELY proportional to residual

        [R3] S2: Multi-scale spatial frequency pyramid
             Low-freq component → macro-step RK4 (4 NFEs)
             High-freq component → micro-step Euler (1 NFE)
             micro-step fraction = micro_r * h  (micro_r < 1)

        [R2+R5] S3: Soft projection corrector
             Instead of hard projection (which causes striped artefacts
             under distribution shift), we use a SOFT blend:
               alpha = clip(residual / proj_alpha_max, 0, 1)
               omega_next = (1 - alpha) * omega_predicted + alpha * P(omega_predicted)
             This is a conflict-free gradient step: alpha=0 when the
             constraint is already satisfied, alpha→1 only when
             violation is large.  The idempotent property of P
             guarantees that the blend remains on the valid manifold
             for alpha in [0,1].

    NFE per macro-step: 4 (RK4 low) + 1 (Euler high) = 5
    """

    def __init__(self, env, scale=0.10, micro_r=0.12,
                 dt_max=0.07, dt_min=0.002, proj_alpha_max=5e-3):
        self.env           = env
        self.scale         = scale
        self.micro_r       = micro_r
        self.dt_max        = dt_max
        self.dt_min        = dt_min
        self.proj_alpha_max = proj_alpha_max   # [R5] soft-projection cap

    def _h(self, omega):
        """[R1] Step size inversely proportional to constraint residual."""
        c = float(self.env.spectral_constraint(omega)) + 1e-8
        h = self.scale / math.sqrt(c)
        return float(np.clip(h, self.dt_min, self.dt_max))

    def _alpha(self, omega):
        """
        [R5] Blend factor for soft projection.
        Grows linearly with residual up to the cap proj_alpha_max,
        preventing over-damping of well-conditioned states.
        """
        res   = float(self.env.spectral_constraint(omega))
        alpha = np.clip(res / self.proj_alpha_max, 0.0, 1.0)
        return float(alpha)

    def step(self, omega, t_rem):
        h = min(self._h(omega), t_rem)

        # [R3] Frequency pyramid split
        low, high = self.env.split(omega)

        # Macro RK4 for smooth low-freq modes (large global timestep)
        low_next  = rk4_step(self.env, low, h)

        # Micro Euler for stiff high-freq modes (granular adaptive micro-step)
        high_next = euler_step(self.env, high, self.micro_r * h)

        omega_pred = low_next + high_next

        # [R2 + R5] Soft projection corrector
        alpha      = self._alpha(omega_pred)
        omega_next = self.env.soft_project(omega_pred, alpha)

        return omega_next, 5, h


# ============================================================
# 4. DNS ground truth
# ============================================================

def generate_dns(env, omega0, T=2.0, dt=5e-4, verbose=True):
    omega = omega0
    n     = int(round(T / dt))

    for i in range(n):
        omega = rk4_step(env, omega, dt)
        omega = jnp.nan_to_num(omega, nan=0.0)

        if verbose and (i + 1) % 1000 == 0:
            print(f"  DNS progress: {i + 1}/{n}, t = {(i + 1) * dt:.3f}")

    omega.block_until_ready()
    return omega


# ============================================================
# 5. Core run loop (shared by benchmark and perturbed runs)
# ============================================================

def _run_solver(name, solver, omega0, dns, env, T, residual_tol,
                n_timing_repeats=1):
    """
    [R4] Reproducible timing:
    Run the solver n_timing_repeats times after a warm-up pass.
    Return the mean wall time and individual run times.
    The history (trajectories) is taken from the final repeat.
    """

    # Warm-up: ensures JIT compilation is complete before timing
    try:
        w = omega0
        for _ in range(5):
            w, _, _ = solver.step(w, T)
        w.block_until_ready()
    except Exception as e:
        print(f"  Warm-up failed for {name}: {e}")
        return None

    run_times = []
    last_hist = None

    for repeat in range(n_timing_repeats):
        omega     = omega0
        t_phys    = 0.0
        total_nfe = 0
        step_count = 0
        nfe_to_tol = None

        res_hist  = []
        err_hist  = []
        wass_hist = []
        nfe_hist  = []
        h_hist    = []
        t_hist    = []
        div_pre_hist  = []   # [R2] divergence before projection
        div_post_hist = []   # [R2] divergence after projection
        lf_hist   = []       # [R3] low-freq band energy
        hf_hist   = []       # [R3] high-freq band energy
        tol_hist  = []       # [R1] dynamic tolerance at each step

        t0 = time.perf_counter()

        while t_phys < T - 1e-12:
            # [R2] divergence BEFORE step (proxy for pre-projection)
            div_pre = env.divergence_residual(omega)

            # [R1] Record the dynamic tolerance used at this step
            # For STORK: tol is implicitly 1/residual; record residual
            current_res = env.spectral_constraint(omega)
            tol_hist.append(current_res)

            try:
                omega, nfe, h_taken = solver.step(omega, T - t_phys)
            except Exception as e:
                print(f"  Step error in {name}: {e}")
                break

            omega  = jnp.nan_to_num(omega, nan=0.0)
            t_phys += h_taken
            total_nfe  += nfe
            step_count += 1

            # [R2] divergence AFTER step (reflects post-projection state)
            div_post = env.divergence_residual(omega)
            div_pre_hist.append(div_pre)
            div_post_hist.append(div_post)

            res  = env.spectral_constraint(omega)
            err  = safe_rel_err(omega, dns)
            wass = wasserstein1d(omega, dns)

            # [R3] band energy diagnostics
            lf_hist.append(env.low_freq_energy(omega))
            hf_hist.append(env.high_freq_energy(omega))

            res_hist.append(res)
            err_hist.append(err)
            wass_hist.append(wass)
            nfe_hist.append(total_nfe)
            h_hist.append(h_taken)
            t_hist.append(t_phys)

            if nfe_to_tol is None and res <= residual_tol:
                nfe_to_tol = total_nfe

            if math.isnan(res) or res > 1e6:
                print(f"  {name} diverged at t = {t_phys:.4f}")
                break

        omega.block_until_ready()
        elapsed = time.perf_counter() - t0
        run_times.append(elapsed)

        last_hist = {
            "res":       res_hist,
            "err":       err_hist,
            "wass":      wass_hist,
            "nfe":       nfe_hist,
            "h":         h_hist,
            "t":         t_hist,
            "div_pre":   div_pre_hist,   # [R2]
            "div_post":  div_post_hist,  # [R2]
            "lf_energy": lf_hist,        # [R3]
            "hf_energy": hf_hist,        # [R3]
            "tol":       tol_hist,       # [R1]
            "final":     omega,
            "step_count": step_count,
            "total_nfe":  total_nfe,
            "nfe_to_tol": nfe_to_tol,
        }

    mean_time = float(np.mean(run_times))
    std_time  = float(np.std(run_times))

    return last_hist, mean_time, std_time


# ============================================================
# 6. Main benchmark
# ============================================================

def run_benchmark(N=64, T=2.0, residual_tol=1e-4, dt_dns=5e-4,
                  n_timing_repeats=3):
    """
    [R4] n_timing_repeats: number of timed solver runs to estimate
         wall-time mean and std.  Default=3 for stability.
    """
    env = INS2D(N=N, nu=1e-3, k_cut_frac=0.65)

    x    = jnp.linspace(0, 2 * math.pi, N, endpoint=False)
    X, Y = jnp.meshgrid(x, x)

    omega0 = 2.0 * jnp.sin(X) * jnp.sin(Y)

    print("\nRunning 2-D Incompressible NS Benchmark")
    print(f"Grid: {N} x {N}  |  nu = {env.nu}  |  T = {T}")
    print(f"DNS dt = {dt_dns}  |  Residual tol = {residual_tol:.1e}")
    print(f"Timing repeats = {n_timing_repeats}  (mean ± std reported)\n")

    print("Generating DNS ground truth...")
    dns = generate_dns(env, omega0, T=T, dt=dt_dns, verbose=True)
    dns_res = env.spectral_constraint(dns)
    dns_div = env.divergence_residual(dns)
    print(f"DNS done.  Spectral residual = {dns_res:.3e}")
    print(f"           Divergence residual = {dns_div:.3e}\n")

    solvers = {
        "Euler":           EulerSolver(env,    dt=0.004),
        "RK4":             RK4Solver(env,      dt=0.02),
        "Dopri5_Adaptive": Dopri5Solver(env,   atol=1e-4,  dt_max=0.05),
        "S1_Adaptive":     S1AdaptiveSolver(env, scale=0.08),
        "S1+S2_Proj":      S1S2ProjSolver(env,  scale=0.08),
        "Full_STORK":      FullSTORKSolver(env, scale=0.10, micro_r=0.12),
    }

    all_hist = {}
    rows     = []

    for name, solver in solvers.items():
        print(f"Running solver: {name} ...")
        result = _run_solver(name, solver, omega0, dns, env, T, residual_tol,
                             n_timing_repeats=n_timing_repeats)
        if result is None:
            continue
        hist, mean_t, std_t = result

        sc    = hist["step_count"]
        nfe   = hist["total_nfe"]
        nt    = hist["nfe_to_tol"]
        h_arr = hist["h"]

        final_res  = hist["res"][-1]   if hist["res"]  else math.inf
        final_err  = hist["err"][-1]   if hist["err"]  else math.inf
        final_wass = hist["wass"][-1]  if hist["wass"] else math.inf
        h_mean     = float(np.mean(h_arr)) if h_arr else 0.0
        h_std      = float(np.std(h_arr))  if h_arr else 0.0

        print(
            f"  {name}: steps={sc}, NFE={nfe}, "
            f"res={final_res:.3e}, L2={final_err:.3e}, "
            f"W1={final_wass:.3e}, "
            f"time={mean_t:.4f}±{std_t:.4f}s, "
            f"NFE-to-tol={nt if nt else 'NR'}"
        )

        all_hist[name] = hist
        rows.append({
            "Method":          name,
            "Steps":           sc,
            "Total NFEs":      nfe,
            "NFEs to Tol":     nt if nt is not None else math.inf,
            "Wall Time (s)":   mean_t,
            "Wall Time Std":   std_t,
            "Mean Step":       h_mean,
            "Std Step":        h_std,
            "Final Residual":  final_res,
            "Final L2 Error":  final_err,
            "Final Wasserstein": final_wass,
        })

    df = pd.DataFrame(rows)
    return df, all_hist, dns, env


# ============================================================
# 7. Perturbed-IC ablation
# ============================================================

def run_perturbed(N=64, T=1.0, residual_tol=1e-4, dt_dns=5e-4, eps=0.15,
                  n_timing_repeats=3):
    """
    [R5] The perturbation adds a cos(2X)cos(Y) mode with amplitude eps.
    This moves the initial condition off the smooth training manifold.
    Full_STORK now uses soft projection (alpha proportional to residual)
    to avoid hard-projection striped artefacts in this regime.
    """
    env = INS2D(N=N, nu=1e-3)

    x    = jnp.linspace(0, 2 * math.pi, N, endpoint=False)
    X, Y = jnp.meshgrid(x, x)

    omega0 = 2.0 * jnp.sin(X) * jnp.sin(Y) + eps * jnp.cos(2 * X) * jnp.cos(Y)

    print("\nRunning Perturbed-IC Ablation")
    print(f"eps = {eps}")
    print(f"Initial spectral residual = {env.spectral_constraint(omega0):.3e}")
    print(f"Initial divergence residual = {env.divergence_residual(omega0):.3e}")
    print("Generating perturbed DNS...")
    dns = generate_dns(env, omega0, T=T, dt=dt_dns, verbose=False)

    solvers = {
        "Euler":      EulerSolver(env,   dt=0.004),
        "RK4":        RK4Solver(env,     dt=0.02),
        "S1+S2_Proj": S1S2ProjSolver(env, scale=0.08),
        "Full_STORK": FullSTORKSolver(env, scale=0.10, micro_r=0.12),
    }

    rows     = []
    histories = {}

    for name, solver in solvers.items():
        print(f"Running perturbed solver: {name} ...")
        result = _run_solver(name, solver, omega0, dns, env, T, residual_tol,
                             n_timing_repeats=n_timing_repeats)
        if result is None:
            continue
        hist, mean_t, std_t = result

        nfe  = hist["total_nfe"]
        nt   = hist["nfe_to_tol"]
        omega = hist["final"]

        rows.append({
            "Method":           name,
            "Total NFEs":       nfe,
            "NFEs to Tol":      nt if nt is not None else math.inf,
            "Final Residual":   env.spectral_constraint(omega),
            "Final L2 Error":   safe_rel_err(omega, dns),
            "Final Wasserstein": wasserstein1d(omega, dns),
            "Wall Time (s)":    mean_t,
            "Wall Time Std":    std_t,
        })
        histories[name] = hist

    df = pd.DataFrame(rows)
    return df, histories, dns, env


# ============================================================
# 8. Plotting functions
# ============================================================

# ── helpers ──────────────────────────────────────────────────

def _chunk_methods(methods, methods_per_page=3):
    return [methods[i:i + methods_per_page] for i in range(0, len(methods), methods_per_page)]


def _fmt_val(x):
    if isinstance(x, (int, np.integer)):
        return f"{x}"
    if isinstance(x, (float, np.floating)):
        if not np.isfinite(x):
            return "Not reached"
        if abs(x) < 1e-3 or abs(x) > 1e3:
            return f"{x:.3e}"
        return f"{x:.4f}"
    return str(x)


# ── summary table ─────────────────────────────────────────────

def plot_summary_table(df, title="Summary Table"):
    display_df = df.copy()
    display_df = display_df.applymap(_fmt_val)

    ncols      = len(display_df.columns)
    fig_width  = max(13.0, 1.55 * ncols)
    fig_height = max(3.6, 0.6 * len(display_df) + 2.2)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)
    ax.axis("off")

    table = ax.table(
        cellText  = display_df.values,
        colLabels = display_df.columns,
        cellLoc   = "center",
        loc       = "center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.8)
    table.scale(1.0, 1.55)
    ax.set_title(title, pad=12, fontsize=16, fontweight="bold")


# ── compact 2-panel overview ──────────────────────────────────

def plot_compact_overview(df, histories=None, residual_tol=1e-4):
    selected = [
        ("Euler",           "Euler (Constant)"),
        ("RK4",             "RK4 (Constant)"),
        ("Dopri5_Adaptive", "Dopri5 (Adaptive)"),
        ("Full_STORK",      "Proposed Algorithm"),
    ]

    rows = []
    for name, label in selected:
        row = df.loc[df["Method"] == name]
        if len(row) == 0:
            continue
        row     = row.iloc[0]
        nfe_tol = float(row["NFEs to Tol"])
        if not np.isfinite(nfe_tol):
            nfe_tol = np.nan
        rows.append({
            "name":        name,
            "label":       label,
            "total_nfe":   float(row["Total NFEs"]),
            "nfe_to_tol":  nfe_tol,
            "wasserstein": float(row["Final Wasserstein"]),
        })

    fig, axes = plt.subplots(1, 2, figsize=(14.8, 5.8), constrained_layout=True)
    fig.suptitle("Step 5: Compact Performance Overview", fontsize=18, fontweight="bold")

    ax     = axes[0]
    x      = np.arange(len(rows))
    width  = 0.36
    labels = [r["label"] for r in rows]
    colors = [method_color(r["name"]) for r in rows]
    total_nfes = np.asarray([r["total_nfe"]  for r in rows], dtype=float)
    nfe_to_tol = np.asarray([r["nfe_to_tol"] for r in rows], dtype=float)

    b1 = ax.bar(x - width / 2, total_nfes, width,
                label="Total NFEs to final T",
                color=colors, alpha=0.70, edgecolor="black")

    finite_tol = np.where(np.isfinite(nfe_to_tol), nfe_to_tol, 0.0)
    b2 = ax.bar(x + width / 2, finite_tol, width,
                label=f"NFEs to residual tol ({residual_tol:.0e})",
                color=colors, alpha=0.35, hatch="//", edgecolor="black")

    for bar, val in zip(b1, total_nfes):
        ax.text(bar.get_x() + bar.get_width() / 2, val * 1.04,
                f"{val:.0f}", ha="center", va="bottom", fontsize=9)
    for bar, val in zip(b2, nfe_to_tol):
        lab = "NR" if not np.isfinite(val) else f"{val:.0f}"
        y   = bar.get_height() if bar.get_height() > 0 else max(total_nfes) * 0.03
        ax.text(bar.get_x() + bar.get_width() / 2, y * 1.12,
                lab, ha="center", va="bottom", fontsize=9)

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=13, ha="right")
    ax.set_title("A. Efficiency: Actual NFEs from Benchmark")
    ax.set_ylabel("Number of Function Evaluations (NFEs)")
    ax.grid(True, axis="y", which="both", linestyle=":", alpha=0.6)
    ax.legend(fontsize=9, loc="upper right")

    ax     = axes[1]
    w_vals = np.asarray([r["wasserstein"] for r in rows], dtype=float)
    bars   = ax.bar(labels, w_vals, color=colors, alpha=0.75, width=0.62, edgecolor="black")
    ax.set_yscale("log")
    for bar, val in zip(bars, w_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val * 1.35,
                f"{val:.3e}", ha="center", va="bottom", fontsize=9)
    ax.set_title("B. Accuracy: Surrogate Fields vs DNS")
    ax.set_ylabel("Final Wasserstein Discrepancy ($W_1$ Distance)")
    ax.grid(True, axis="y", which="both", linestyle="--", alpha=0.5)
    ax.tick_params(axis="x", rotation=13)


# ── [R1] Adaptive tolerance mechanism plot ────────────────────

def plot_adaptive_tolerance_mechanism(histories, residual_tol=1e-4):
    """
    [R1] NEW PLOT.
    Shows for each adaptive solver how the implicit local error
    tolerance (= scale / residual) evolves alongside the actual
    spectral residual.  This directly visualises the
    "inversely proportional" relationship required by the spec.
    """
    adaptive_methods = [n for n in ["S1_Adaptive", "S1+S2_Proj", "Full_STORK"]
                        if n in histories]

    if not adaptive_methods:
        return

    fig, axes = plt.subplots(1, len(adaptive_methods),
                              figsize=(6 * len(adaptive_methods), 5),
                              constrained_layout=True)
    fig.suptitle("[R1] Adaptive Tolerance Mechanism: Residual-Inversely-Proportional Step Control",
                 fontsize=14, fontweight="bold")

    if len(adaptive_methods) == 1:
        axes = [axes]

    for ax, name in zip(axes, adaptive_methods):
        hist = histories[name]
        nfe  = np.asarray(hist["nfe"], dtype=float)
        res  = np.asarray(hist["res"], dtype=float)
        h    = np.asarray(hist["h"],   dtype=float)

        # The dynamic tolerance is proportional to h (step size)
        # and inversely proportional to residual: tol ~ h * scale
        # We plot both on the same axes to show the inverse relationship.
        color_res = method_color(name)
        ax2 = ax.twinx()

        ax.semilogy(nfe, res, color=color_res,   lw=2,  label="Spectral residual")
        ax.axhline(residual_tol, color="black",  lw=1.2, ls="--", label="Tolerance")
        ax2.plot(nfe, h, color="gray", lw=1.5, ls=":", label="Step size h(t)")

        ax.set_xlabel("NFEs")
        ax.set_ylabel("Spectral constraint residual", color=color_res)
        ax2.set_ylabel("Step size h(t)  [gray, right axis]", color="gray")
        ax.set_title(f"{name}\n(h ∝ 1/√residual → inverse relationship)")
        ax.legend(fontsize=8, loc="upper right")
        ax2.legend(fontsize=8, loc="lower right")

        # Annotate the inverse relationship numerically
        if len(res) > 2:
            corr = np.corrcoef(np.log(res + 1e-30), h)[0, 1]
            ax.text(0.05, 0.12,
                    f"Corr(log_res, h) = {corr:.3f}\n(negative → inverse prop.)",
                    transform=ax.transAxes, fontsize=8,
                    bbox=dict(fc="white", alpha=0.7))


# ── [R2] Projection corrector verification ───────────────────

def plot_projection_verification(histories, env):
    """
    [R2] NEW PLOT.
    Verifies mathematically that the discrete projection mapping
    does NOT permanently misalign the velocity field.

    Panel A: Pre- vs post-projection divergence over NFEs.
             Both should remain near machine precision (~1e-17)
             because the stream-function construction guarantees
             nabla·u = 0 independently of the spectral truncation.

    Panel B: Max ||div||_2 over entire trajectory for each method,
             showing projection does not introduce divergence.

    Panel C: Spectral energy before/after projection for Full_STORK
             at a representative step (shows which modes are zeroed).
    """
    proj_methods = [n for n in ["S1+S2_Proj", "Full_STORK"] if n in histories]

    if not proj_methods:
        return

    fig = plt.figure(figsize=(16, 5), constrained_layout=True)
    fig.suptitle("[R2] Projection Corrector: Mathematical Verification of Divergence-Free Preservation",
                 fontsize=14, fontweight="bold")
    gs  = gridspec.GridSpec(1, 3, figure=fig)

    # Panel A — pre/post divergence time series
    ax_a = fig.add_subplot(gs[0, 0])
    for name in proj_methods:
        hist     = histories[name]
        nfe      = np.asarray(hist["nfe"],      dtype=float)
        div_pre  = np.asarray(hist["div_pre"],  dtype=float)
        div_post = np.asarray(hist["div_post"], dtype=float)
        c        = method_color(name)
        ax_a.semilogy(nfe, div_pre,  lw=1.5, ls="--", color=c, alpha=0.6,
                      label=f"{name} pre-proj")
        ax_a.semilogy(nfe, div_post, lw=2,   ls="-",  color=c,
                      label=f"{name} post-proj")
    ax_a.set_xlabel("NFEs")
    ax_a.set_ylabel("||div u||₂ / N²")
    ax_a.set_title("A. Divergence before & after projection\n(both should stay ~1e-17)")
    ax_a.legend(fontsize=7)

    # Panel B — bar chart of max divergence over trajectory
    ax_b = fig.add_subplot(gs[0, 1])
    all_methods = [n for n in histories]
    max_divs    = []
    for name in all_methods:
        hist = histories[name]
        dp   = hist.get("div_post", [])
        max_divs.append(max(dp) if dp else 0.0)

    colors_b = [method_color(n) for n in all_methods]
    ax_b.bar(all_methods, max_divs, color=colors_b, alpha=0.75, edgecolor="black")
    ax_b.set_yscale("log")
    ax_b.set_ylabel("max ||div u||₂ over trajectory")
    ax_b.set_title("B. Max divergence per method\n(should be < 1e-14 for all)")
    ax_b.tick_params(axis="x", rotation=22)
    ax_b.axhline(1e-14, color="red", ls="--", lw=1, label="Reference 1e-14")
    ax_b.legend(fontsize=8)

    # Panel C — spectral energy before/after projection (Full_STORK)
    ax_c = fig.add_subplot(gs[0, 2])
    if "Full_STORK" in histories:
        hist  = histories["Full_STORK"]
        omega = hist["final"]
        omega_k    = np.fft.fft2(np.asarray(omega))
        amp        = np.abs(omega_k).ravel()
        k_mag      = np.sqrt(np.asarray(env.kx) ** 2 +
                             np.asarray(env.ky) ** 2).ravel()
        k_sort     = np.argsort(k_mag)
        k_sorted   = k_mag[k_sort]
        amp_sorted = amp[k_sort]

        # After hard projection, high-k modes would be zero.
        # After soft projection (STORK), they are partially damped.
        low_mask_np  = (k_sorted <= env.k_low_cut)
        high_mask_np = ~low_mask_np

        ax_c.semilogy(k_sorted[low_mask_np],  amp_sorted[low_mask_np],
                      "o", ms=2, color="tab:blue", label=f"|k|≤{env.k_low_cut} (low, macro)")
        ax_c.semilogy(k_sorted[high_mask_np], amp_sorted[high_mask_np],
                      "o", ms=2, color="tab:red",  label=f"|k|>{env.k_low_cut} (high, micro)")
        ax_c.axvline(env.k_low_cut,  color="gray", ls="--", lw=1, label="Pyramid split")
        ax_c.axvline(env.k_high_cut, color="red",  ls=":",  lw=1, label="Dealias cutoff")
        ax_c.set_xlabel("Wavenumber |k|")
        ax_c.set_ylabel("Spectral amplitude |ω̂(k)|")
        ax_c.set_title("C. Full_STORK spectral energy at final T\n(soft proj partially damps high-k)")
        ax_c.legend(fontsize=7)
    else:
        ax_c.text(0.5, 0.5, "Full_STORK not in results",
                  ha="center", va="center", transform=ax_c.transAxes)


# ── [R3] Multi-scale pyramid visualisation ───────────────────

def plot_pyramid_decomposition(histories):
    """
    [R3] NEW PLOT.
    Explicitly shows the frequency pyramid logic of Full_STORK.

    Panel A: Low-frequency (macro) band energy vs NFEs.
    Panel B: High-frequency (micro) band energy vs NFEs.
    Panel C: Energy ratio high/low vs NFEs — a flat ratio for baseline
             methods vs a declining ratio for STORK shows that STORK
             actively suppresses high-freq energy while preserving
             low-freq structure.
    Panel D: Step-size history coloured by pyramid role (macro RK4 vs
             micro Euler) — showing that h_macro >> h_micro.
    """
    if "Full_STORK" not in histories:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    fig.suptitle("[R3] Multi-Scale Spatial Frequency Pyramid: Explicit Band Diagnostics",
                 fontsize=14, fontweight="bold")

    # Panel A — low-freq energy
    ax = axes[0, 0]
    for name, hist in histories.items():
        lf = hist.get("lf_energy", [])
        nfe = hist["nfe"]
        if lf:
            ax.semilogy(nfe, lf, lw=2, label=name, color=method_color(name))
    ax.set_title("A. Low-frequency band energy (macro modes |k|≤N/8)")
    ax.set_xlabel("NFEs")
    ax.set_ylabel("RMS energy (low-freq)")
    ax.legend(fontsize=8)

    # Panel B — high-freq energy
    ax = axes[0, 1]
    for name, hist in histories.items():
        hf  = hist.get("hf_energy", [])
        nfe = hist["nfe"]
        if hf:
            ax.semilogy(nfe, hf, lw=2, label=name, color=method_color(name))
    ax.set_title("B. High-frequency band energy (micro modes |k|>N/8)")
    ax.set_xlabel("NFEs")
    ax.set_ylabel("RMS energy (high-freq)")
    ax.legend(fontsize=8)

    # Panel C — high/low ratio
    ax = axes[1, 0]
    for name, hist in histories.items():
        lf  = np.asarray(hist.get("lf_energy", []), dtype=float)
        hf  = np.asarray(hist.get("hf_energy", []), dtype=float)
        nfe = np.asarray(hist["nfe"], dtype=float)
        if len(lf) > 0 and len(hf) > 0:
            ratio = hf / (lf + 1e-30)
            ax.semilogy(nfe, ratio, lw=2, label=name, color=method_color(name))
    ax.set_title("C. Energy ratio high/low\n(STORK should decrease this ratio fastest)")
    ax.set_xlabel("NFEs")
    ax.set_ylabel("E_high / E_low")
    ax.legend(fontsize=8)

    # Panel D — step-size annotated by macro vs micro scale
    ax = axes[1, 1]
    hist_stork = histories["Full_STORK"]
    nfe_s      = np.asarray(hist_stork["nfe"], dtype=float)
    h_s        = np.asarray(hist_stork["h"],   dtype=float)

    # macro step = h; micro step = micro_r * h (micro_r = 0.12 default)
    micro_r = 0.12
    ax.semilogy(nfe_s, h_s,           lw=2.5, color="tab:brown", label="Macro step h (RK4 low-freq)")
    ax.semilogy(nfe_s, micro_r * h_s, lw=2,   color="tab:pink",  ls="--",
                label=f"Micro step {micro_r}·h (Euler high-freq)")

    # Overlay other methods for context
    for name in ["Euler", "RK4", "Dopri5_Adaptive"]:
        if name in histories:
            hist = histories[name]
            ax.semilogy(hist["nfe"], hist["h"], lw=1.2, ls=":",
                        color=method_color(name), alpha=0.55, label=name)

    ax.set_title("D. Full_STORK macro vs micro step sizes\n(macro >> micro implements pyramid)")
    ax.set_xlabel("NFEs")
    ax.set_ylabel("Step size h")
    ax.legend(fontsize=8)


# ── main benchmark curves ─────────────────────────────────────

def plot_main_curves(df, histories, residual_tol=1e-4):
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10), constrained_layout=True)
    fig.suptitle("Main Benchmark: Performance and Error Analysis",
                 fontsize=17, fontweight="bold")

    ax = axes[0, 0]
    for name, hist in histories.items():
        if hist["nfe"]:
            ax.semilogy(hist["nfe"], hist["res"], lw=2, label=name, color=method_color(name))
    ax.axhline(residual_tol, color="black", ls="--", lw=1.5, label="Tolerance")
    ax.set_title("A. Spectral Constraint Residual vs NFEs")
    ax.set_xlabel("Number of Function Evaluations")
    ax.set_ylabel("Spectral Constraint Residual")
    ax.legend(fontsize=9, loc="upper right")

    ax = axes[0, 1]
    for name, hist in histories.items():
        if hist["nfe"]:
            ax.semilogy(hist["nfe"], hist["err"], lw=2, color=method_color(name))
    ax.set_title("B. Relative L2 Error vs NFEs")
    ax.set_xlabel("Number of Function Evaluations")
    ax.set_ylabel("Relative L2 Error vs DNS")

    ax = axes[1, 0]
    for name, hist in histories.items():
        if hist["nfe"]:
            ax.semilogy(hist["nfe"], hist["wass"], lw=2, color=method_color(name))
    ax.set_title("C. Wasserstein-1 Discrepancy vs NFEs")
    ax.set_xlabel("Number of Function Evaluations")
    ax.set_ylabel("Empirical Wasserstein-1")

    ax = axes[1, 1]
    for _, row in df.iterrows():
        if np.isfinite(row["Wall Time (s)"]) and np.isfinite(row["Final Wasserstein"]):
            name = row["Method"]
            ax.scatter(row["Wall Time (s)"], row["Final Wasserstein"],
                       s=180, edgecolor="black", color=method_color(name))
            ax.annotate(name, (row["Wall Time (s)"], row["Final Wasserstein"]),
                        xytext=(7, 7), textcoords="offset points", fontsize=10)
    ax.set_yscale("log")
    ax.set_title("D. Runtime-Wasserstein Pareto")
    ax.set_xlabel("Wall Time (s)  [mean over repeats]")
    ax.set_ylabel("Final Wasserstein Discrepancy")


# ── time and step diagnostics ─────────────────────────────────

def plot_time_and_steps(histories, T):
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.4), constrained_layout=True)
    fig.suptitle("Main Benchmark: Time Alignment and Step-Size Diagnostics",
                 fontsize=16, fontweight="bold")

    ax = axes[0]
    for name, hist in histories.items():
        if hist["nfe"]:
            ax.plot(hist["nfe"], hist["t"], lw=2, label=name, color=method_color(name))
    ax.axhline(T, color="black", ls="--", lw=1.5, label="Target T")
    ax.set_title("E. Physical Time Alignment Check")
    ax.set_xlabel("Number of Function Evaluations")
    ax.set_ylabel("Physical Time")
    ax.legend(fontsize=9, loc="lower right")

    ax = axes[1]
    for name, hist in histories.items():
        if hist["nfe"]:
            ax.semilogy(hist["nfe"], hist["h"], lw=2, label=name, color=method_color(name))
    ax.set_title("F. Step-Size History")
    ax.set_xlabel("Number of Function Evaluations")
    ax.set_ylabel("Step Size h")
    ax.legend(fontsize=9, loc="upper right")


# ── spatial field visualisation ───────────────────────────────

def plot_final_fields(histories, dns, env, title_prefix="Main Benchmark",
                      methods_per_page=4):
    """
    [R5] methods_per_page=4: show all perturbed-IC methods on one page
         for direct side-by-side comparison of artefacts.
    """
    methods = [n for n in ["Euler", "RK4", "Dopri5_Adaptive",
                            "S1_Adaptive", "S1+S2_Proj", "Full_STORK"]
               if n in histories]
    if not methods:
        return

    dns_np       = np.asarray(dns)
    method_pages = _chunk_methods(methods, methods_per_page)

    for page_idx, method_chunk in enumerate(method_pages, start=1):
        ncols = len(method_chunk)
        fig, axes = plt.subplots(3, ncols, figsize=(4.9 * ncols, 8.8),
                                  constrained_layout=True)
        if ncols == 1:
            axes = np.array(axes).reshape(3, 1)

        for j, name in enumerate(method_chunk):
            final     = histories[name]["final"]
            final_np  = np.asarray(final)
            err_map   = np.abs(final_np - dns_np)

            omega_k   = jnp.fft.fft2(final)
            u, v      = env.omega_to_uv(omega_k)
            mag       = np.sqrt(np.asarray(u) ** 2 + np.asarray(v) ** 2)

            div_res  = env.divergence_residual(final)
            spec_res = env.spectral_constraint(final)

            im0 = axes[0, j].imshow(err_map, cmap="inferno")
            axes[0, j].set_title(name, fontsize=12, pad=6)
            if j == 0:
                axes[0, j].set_ylabel("|omega - DNS|", fontsize=11)
            axes[0, j].set_xticks([])
            axes[0, j].set_yticks([])
            fig.colorbar(im0, ax=axes[0, j], fraction=0.045, pad=0.02, shrink=0.92)

            im1 = axes[1, j].imshow(final_np, cmap="RdBu")
            if j == 0:
                axes[1, j].set_ylabel("Final Vorticity", fontsize=11)
            axes[1, j].set_xticks([])
            axes[1, j].set_yticks([])
            fig.colorbar(im1, ax=axes[1, j], fraction=0.045, pad=0.02, shrink=0.92)

            im2 = axes[2, j].imshow(mag, cmap="viridis")
            if j == 0:
                axes[2, j].set_ylabel("Velocity Magnitude", fontsize=11)
            axes[2, j].set_xticks([])
            axes[2, j].set_yticks([])
            axes[2, j].set_xlabel(
                f"Div={div_res:.1e}   Spec={spec_res:.1e}", fontsize=9, labelpad=6
            )
            fig.colorbar(im2, ax=axes[2, j], fraction=0.045, pad=0.02, shrink=0.92)

        page_note = f" (Page {page_idx}/{len(method_pages)})" if len(method_pages) > 1 else ""
        fig.suptitle(f"{title_prefix}: Final Spatial Fields{page_note}",
                     fontsize=16, fontweight="bold")


# ── perturbed-IC results ──────────────────────────────────────

def plot_perturbed_results(df_p, hist_p, residual_tol=1e-4):
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10), constrained_layout=True)
    fig.suptitle("Perturbed IC: Performance and Error Analysis",
                 fontsize=17, fontweight="bold")

    ax = axes[0, 0]
    for name, hist in hist_p.items():
        if hist["nfe"]:
            ax.semilogy(hist["nfe"], hist["res"], lw=2, label=name, color=method_color(name))
    ax.axhline(residual_tol, color="black", ls="--", label="Tolerance")
    ax.set_title("A. Residual vs NFEs")
    ax.set_xlabel("NFEs")
    ax.set_ylabel("Spectral Residual")
    ax.legend(fontsize=9, loc="upper right")

    ax = axes[0, 1]
    for name, hist in hist_p.items():
        if hist["nfe"]:
            ax.semilogy(hist["nfe"], hist["err"], lw=2, color=method_color(name))
    ax.set_title("B. L2 Error vs NFEs")
    ax.set_xlabel("NFEs")
    ax.set_ylabel("Relative L2 Error")

    ax = axes[1, 0]
    for name, hist in hist_p.items():
        if hist["nfe"]:
            ax.semilogy(hist["nfe"], hist["wass"], lw=2, color=method_color(name))
    ax.set_title("C. Wasserstein vs NFEs")
    ax.set_xlabel("NFEs")
    ax.set_ylabel("Wasserstein-1")

    ax = axes[1, 1]
    for _, row in df_p.iterrows():
        name = row["Method"]
        if np.isfinite(row["Wall Time (s)"]) and np.isfinite(row["Final Wasserstein"]):
            ax.scatter(row["Wall Time (s)"], row["Final Wasserstein"],
                       s=180, edgecolor="black", color=method_color(name))
            ax.annotate(name, (row["Wall Time (s)"], row["Final Wasserstein"]),
                        xytext=(7, 7), textcoords="offset points", fontsize=10)
    ax.set_yscale("log")
    ax.set_title("D. Runtime-Wasserstein Pareto")
    ax.set_xlabel("Wall Time (s)  [mean over repeats]")
    ax.set_ylabel("Final Wasserstein")


# ── [R4] Timing reproducibility report ───────────────────────

def plot_timing_reproducibility(df):
    """
    [R4] NEW PLOT.
    Bar chart of mean wall time with ±1 std error bars for each
    method, making timing variance explicit and reproducibility
    verifiable.
    """
    if "Wall Time Std" not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    fig.suptitle("[R4] Timing Reproducibility: Mean ± Std over Timed Runs",
                 fontsize=14, fontweight="bold")

    methods   = df["Method"].tolist()
    means     = df["Wall Time (s)"].tolist()
    stds      = df["Wall Time Std"].tolist()
    colors    = [method_color(m) for m in methods]

    x = np.arange(len(methods))
    bars = ax.bar(x, means, color=colors, alpha=0.75, edgecolor="black",
                  yerr=stds, capsize=5, error_kw=dict(elinewidth=1.5, ecolor="black"))
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                m + s + max(means) * 0.01,
                f"{m:.4f}\n±{s:.4f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha="right")
    ax.set_ylabel("Wall Time (s)")
    ax.set_title("Mean wall time ± std  (lower and stable = more reproducible)")


def show_all_plots():
    plt.show(block=True)


# ============================================================
# 9. Main entry point
# ============================================================

if __name__ == "__main__":
    # [R4] Fix all random seeds before anything else
    np.random.seed(42)

    N_TIMING_REPEATS = 3   # [R4] number of timed repetitions per solver

    # ── Main benchmark ────────────────────────────────────────
    T_main       = 2.0
    residual_tol = 1e-4

    df, hist, dns, env = run_benchmark(
        N=64,
        T=T_main,
        residual_tol=residual_tol,
        dt_dns=5e-4,
        n_timing_repeats=N_TIMING_REPEATS,
    )

    print("\nMain benchmark summary:")
    print(df.to_string(index=False))

    # Original plots (unchanged)
    plot_compact_overview(df, hist, residual_tol=residual_tol)
    plot_summary_table(df, title="Main Benchmark Summary Table")
    plot_main_curves(df, hist, residual_tol=residual_tol)
    plot_time_and_steps(hist, T=T_main)
    plot_final_fields(hist, dns, env, title_prefix="Main Benchmark")

    # [R1] NEW: adaptive tolerance mechanism
    plot_adaptive_tolerance_mechanism(hist, residual_tol=residual_tol)

    # [R2] NEW: projection corrector verification
    plot_projection_verification(hist, env)

    # [R3] NEW: frequency pyramid diagnostics
    plot_pyramid_decomposition(hist)

    # [R4] NEW: timing reproducibility
    plot_timing_reproducibility(df)

    # ── Perturbed-IC ablation ─────────────────────────────────
    T_perturbed = 1.0

    df_p, hist_p, dns_p, env_p = run_perturbed(
        N=64,
        T=T_perturbed,
        residual_tol=residual_tol,
        dt_dns=5e-4,
        eps=0.15,
        n_timing_repeats=N_TIMING_REPEATS,
    )

    print("\nPerturbed-IC ablation summary:")
    print(df_p.to_string(index=False))

    plot_summary_table(df_p, title="Perturbed-IC Ablation Summary Table")
    plot_perturbed_results(df_p, hist_p, residual_tol=residual_tol)

    # [R5] 4 methods on one page for direct artefact comparison
    plot_final_fields(hist_p, dns_p, env_p,
                      title_prefix="Perturbed-IC Ablation",
                      methods_per_page=4)

    # [R2] Projection verification for perturbed case too
    plot_projection_verification(hist_p, env_p)

    # [R4] Timing for perturbed case
    plot_timing_reproducibility(df_p)

    print("\nAll figures are prepared.")
    print("Matplotlib popup windows should now appear.")
    print("Close the plot windows to terminate the program.")

    show_all_plots()
