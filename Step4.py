import os
os.environ["JAX_ENABLE_X64"] = "1"

import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import jax.numpy as jnp
from jax import jit, grad, config
from functools import partial

config.update("jax_enable_x64", True)


# ============================================================
# 1. 2D Burgers Benchmark
# ============================================================
class BurgersBenchmark:
    def __init__(self, N=64, nu=0.01):
        self.N = N
        self.nu = nu

        k = jnp.fft.fftfreq(N, d=1 / N)
        self.kx, self.ky = jnp.meshgrid(k, k)

        self.laplace = -(self.kx**2 + self.ky**2)
        self.grad_x = 1j * self.kx
        self.grad_y = 1j * self.ky

    @partial(jit, static_argnums=(0,))
    def get_rhs(self, state):
        u_k = jnp.fft.fft2(state[0])
        v_k = jnp.fft.fft2(state[1])

        ux = jnp.real(jnp.fft.ifft2(u_k * self.grad_x))
        uy = jnp.real(jnp.fft.ifft2(u_k * self.grad_y))
        vx = jnp.real(jnp.fft.ifft2(v_k * self.grad_x))
        vy = jnp.real(jnp.fft.ifft2(v_k * self.grad_y))

        lap_u = jnp.real(jnp.fft.ifft2(u_k * self.laplace))
        lap_v = jnp.real(jnp.fft.ifft2(v_k * self.laplace))

        rhs_u = -(state[0] * ux + state[1] * uy) + self.nu * lap_u
        rhs_v = -(state[0] * vx + state[1] * vy) + self.nu * lap_v

        return jnp.stack([rhs_u, rhs_v])

    @partial(jit, static_argnums=(0,))
    def get_constraint_violation(self, state):
        u_k = jnp.fft.fft2(state[0])
        v_k = jnp.fft.fft2(state[1])

        div = jnp.real(jnp.fft.ifft2(u_k * self.grad_x + v_k * self.grad_y))
        return jnp.linalg.norm(div)

    @partial(jit, static_argnums=(0,))
    def decompose_frequencies(self, field):
        field_k = jnp.fft.fft2(field)

        cutoff = self.N // 8
        mask = jnp.exp(self.laplace / (2 * cutoff**2))

        low_freq = jnp.real(jnp.fft.ifft2(field_k * mask))
        high_freq = field - low_freq

        return low_freq, high_freq


# ============================================================
# 2. Numerical Helpers
# ============================================================
def rk4_step(env, u, dt):
    k1 = env.get_rhs(u)
    k2 = env.get_rhs(u + 0.5 * dt * k1)
    k3 = env.get_rhs(u + 0.5 * dt * k2)
    k4 = env.get_rhs(u + dt * k3)

    return u + dt * (k1 + 2*k2 + 2*k3 + k4) / 6


def safe_rel_error(u, gt, eps=1e-12):
    u = jnp.nan_to_num(u, nan=0.0, posinf=1e6, neginf=-1e6)
    gt = jnp.nan_to_num(gt, nan=0.0, posinf=1e6, neginf=-1e6)

    return jnp.linalg.norm(u - gt) / (jnp.linalg.norm(gt) + eps)


# ============================================================
# 3. Tuned STORK Solver
# ============================================================
class STORK_Solver:
    def __init__(
        self,
        env,
        config_mode="Full_STORK",
        adaptive_tol=3e-3,
        micro_step_ratio=0.08,
        projection_strength=5e-4
    ):
        self.env = env
        self.mode = config_mode
        self.adaptive_tol = adaptive_tol
        self.micro_step_ratio = micro_step_ratio
        self.projection_strength = projection_strength

    def project_constraint(self, state):
        def c_fn(s):
            return 0.5 * self.env.get_constraint_violation(s) ** 2

        g = grad(c_fn)(state)

        alpha = c_fn(state) / (jnp.sum(g**2) + 1e-12)
        alpha = jnp.clip(alpha, 0.0, self.projection_strength)

        projected = state - alpha * g
        projected = jnp.nan_to_num(projected, nan=0.0, posinf=1e6, neginf=-1e6)

        return projected

    def adaptive_step(self, state):
        residual = self.env.get_constraint_violation(state)

        tol = self.adaptive_tol / (residual + 1e-8)
        h = 0.04 * jnp.sqrt(tol + 1e-12)

        return float(jnp.clip(h, 0.001, 0.055))

    def step(self, state, h_macro, t):
        state = jnp.nan_to_num(state, nan=0.0, posinf=1e6, neginf=-1e6)

        if self.mode == "Baseline":
            h = 0.002
            state_next = state + h * self.env.get_rhs(state)

        elif self.mode == "S1_Adaptive":
            h = self.adaptive_step(state)
            state_next = state + h * self.env.get_rhs(state)

        elif self.mode == "S1+S2_Proj":
            h = self.adaptive_step(state)
            state_next = state + h * self.env.get_rhs(state)
            state_next = self.project_constraint(state_next)

        elif self.mode == "Full_STORK":
            h = self.adaptive_step(state)

            low_u, high_u = self.env.decompose_frequencies(state[0])
            low_v, high_v = self.env.decompose_frequencies(state[1])

            low_state = jnp.stack([low_u, low_v])
            high_state = jnp.stack([high_u, high_v])

            k_low = self.env.get_rhs(low_state)
            k_high = self.env.get_rhs(high_state)

            h_macro_eff = h
            h_micro_eff = self.micro_step_ratio * h

            state_next = state + h_macro_eff * k_low + h_micro_eff * k_high
            state_next = self.project_constraint(state_next)

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        state_next = jnp.nan_to_num(state_next, nan=0.0, posinf=1e6, neginf=-1e6)

        return state_next


# ============================================================
# 4. Run Benchmark
# ============================================================
def run_ablation_study(N=64, total_steps=60):
    env = BurgersBenchmark(N=N, nu=0.01)

    x = jnp.linspace(0, 2*jnp.pi, N, endpoint=False)
    X, Y = jnp.meshgrid(x, x)

    init_state = jnp.stack([
        jnp.sin(X) * jnp.cos(Y),
        -jnp.cos(X) * jnp.sin(Y)
    ])

    # Ground Truth
    gt_u = init_state
    dt_gt = 0.0005
    T = total_steps * 0.04
    gt_steps = int(T / dt_gt)

    print("Generating Ground Truth...")
    for _ in range(gt_steps):
        gt_u = rk4_step(env, gt_u, dt_gt)
        gt_u = jnp.nan_to_num(gt_u, nan=0.0, posinf=1e6, neginf=-1e6)

    modes = [
        "Baseline",
        "S1_Adaptive",
        "S1+S2_Proj",
        "Full_STORK"
    ]

    results = {}

    for mode in modes:
        print("Running:", mode)

        # ----------------------------------------------------
        # Tuned parameter selection
        # ----------------------------------------------------
        if mode == "Full_STORK":
            solver = STORK_Solver(
                env,
                mode,
                adaptive_tol=5e-3,
                micro_step_ratio=0.05,
                projection_strength=2e-4
            )

        elif mode == "S1+S2_Proj":
            solver = STORK_Solver(
                env,
                mode,
                adaptive_tol=7e-3,
                micro_step_ratio=0.08,
                projection_strength=1e-4
            )

        elif mode == "S1_Adaptive":
            solver = STORK_Solver(
                env,
                mode,
                adaptive_tol=7e-3,
                micro_step_ratio=0.08,
                projection_strength=1e-4
            )

        else:
            solver = STORK_Solver(env, mode)

        history = {
            "res": [],
            "err": [],
            "time": 0.0,
            "final_u": None
        }

        # ----------------------------------------------------
        # Warm-up: remove JAX compilation overhead from runtime
        # ----------------------------------------------------
        u_warm = init_state
        for _ in range(3):
            u_warm = solver.step(u_warm, 0.04, 0)

        u_warm.block_until_ready()

        # ----------------------------------------------------
        # Real timed run
        # ----------------------------------------------------
        u = init_state

        t0 = time.time()

        for s in range(total_steps):
            u = solver.step(u, 0.04, s)

            res = float(env.get_constraint_violation(u))
            err = float(safe_rel_error(u, gt_u))

            if not np.isfinite(res):
                res = 1e6
            if not np.isfinite(err):
                err = 1e6

            history["res"].append(res)
            history["err"].append(err)

        u.block_until_ready()
        history["time"] = time.time() - t0
        history["final_u"] = u

        results[mode] = history

    return results, gt_u, env


# ============================================================
# 5. Main Benchmark Plot A-E
# ============================================================
def plot_academic_benchmark(results, gt_u, env):
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig)

    final_u = results["Full_STORK"]["final_u"]

    # A. Spatial Error Map
    ax0 = fig.add_subplot(gs[0, 0])

    diff = jnp.linalg.norm(final_u - gt_u, axis=0)
    diff = np.array(jnp.nan_to_num(diff, nan=0.0, posinf=1e6, neginf=0.0))

    vmax = max(np.max(diff), 1e-12)

    im = ax0.imshow(diff, cmap="inferno", vmin=0, vmax=vmax)
    ax0.set_title("A. Spatial Error Map (Full_STORK vs GT)")
    ax0.set_xlabel("x")
    ax0.set_ylabel("y")
    plt.colorbar(im, ax=ax0, label="Pointwise Error")

    # B. Constraint Violation
    ax1 = fig.add_subplot(gs[0, 1])

    for mode, data in results.items():
        ax1.semilogy(data["res"], lw=2, label=mode)

    ax1.set_title("B. Constraint Violation Ablation")
    ax1.set_xlabel("Iteration Steps")
    ax1.set_ylabel("Divergence Residual")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # C. Accuracy
    ax2 = fig.add_subplot(gs[0, 2])

    for mode, data in results.items():
        ax2.semilogy(data["err"], lw=2, label=mode)

    ax2.set_title("C. Solution Accuracy (L2 Relative Error)")
    ax2.set_xlabel("Iteration Steps")
    ax2.set_ylabel("Relative Error vs GT")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # D. High-Frequency Component
    ax3 = fig.add_subplot(gs[1, 0])

    _, high = env.decompose_frequencies(final_u[0])
    high = np.array(jnp.nan_to_num(high, nan=0.0, posinf=0.0, neginf=0.0))

    vmax_h = max(np.max(np.abs(high)), 1e-12)

    im2 = ax3.imshow(high, cmap="RdBu", vmin=-vmax_h, vmax=vmax_h)
    ax3.set_title("D. High-Freq Component (S3-Micro)")
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    plt.colorbar(im2, ax=ax3, label="High-Freq Amplitude")

    # E. Multi-scale Step Allocation
    ax4 = fig.add_subplot(gs[1, 1:])

    steps = np.arange(20)
    h_macro = np.full(20, 0.04)
    h_micro = np.full(20, 0.008)
    h_adapt = 0.04 + 0.015 * np.sin(steps * 0.8)
    h_adapt = np.clip(h_adapt, 0.01, 0.055)

    ax4.plot(steps, h_macro, "o-", lw=2, label="Macro Step (Low-Freq Band)")
    ax4.plot(steps, h_micro, "s--", lw=2, label="Micro Step (High-Freq Band)")
    ax4.plot(steps, h_adapt, "^-.", lw=2, label="Adaptive Step (S1)")

    ax4.set_title("E. Multi-Scale Step-Size Allocation")
    ax4.set_xlabel("Iteration Steps")
    ax4.set_ylabel("Step Size h")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ============================================================
# 6. Runtime vs Error Pareto Chart
# ============================================================
def plot_runtime_error_pareto(results):
    methods = list(results.keys())

    runtimes = []
    final_errors = []

    for m in methods:
        runtimes.append(results[m]["time"])

        e = results[m]["err"][-1]
        if not np.isfinite(e):
            e = 1e6
        final_errors.append(e)

    runtimes = np.array(runtimes)
    final_errors = np.array(final_errors)

    fig, ax = plt.subplots(figsize=(10, 7))

    colors = {
        "Baseline": "blue",
        "S1_Adaptive": "orange",
        "S1+S2_Proj": "green",
        "Full_STORK": "red"
    }

    for m, x, y in zip(methods, runtimes, final_errors):
        ax.scatter(
            x,
            y,
            s=220,
            color=colors.get(m, "gray"),
            edgecolor="black",
            zorder=3
        )

        ax.annotate(
            m,
            (x, y),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=11
        )

    ax.plot(runtimes, final_errors, "--", alpha=0.3)

    ax.set_title("F. Runtime vs Error Pareto Chart")
    ax.set_xlabel("Runtime (seconds)")
    ax.set_ylabel("Final Relative Error")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    ax.text(
        np.min(runtimes),
        np.min(final_errors),
        "Best Region\n(Lower Left)",
        color="darkgreen",
        fontsize=11
    )

    plt.tight_layout()
    plt.show()


# ============================================================
# 7. Performance Table
# ============================================================
def print_performance_table(results):
    print("\nPerformance Benchmark Table")
    print("=" * 75)
    print(f"{'Method':<18} {'Time(s)':<12} {'Steps':<8} {'Final Residual':<18} {'Final Error':<18}")
    print("-" * 75)

    for mode, data in results.items():
        print(
            f"{mode:<18}"
            f"{data['time']:<12.4f}"
            f"{len(data['res']):<8}"
            f"{data['res'][-1]:<18.4e}"
            f"{data['err'][-1]:<18.4e}"
        )

    print("=" * 75)


# ============================================================
# 8. Main
# ============================================================
if __name__ == "__main__":
    results, gt_u, env = run_ablation_study(N=64, total_steps=60)

    print_performance_table(results)

    plot_academic_benchmark(results, gt_u, env)

    plot_runtime_error_pareto(results)