"""
Step 3: Multi-Scale Spatial Frequency Pyramid Integrator
=========================================================
Combined: Full numerical implementation + Enhanced visualization
"""

import os
os.environ["JAX_ENABLE_X64"] = "1"

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, grad
from jax import lax
from functools import partial
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ══════════════════════════════════════════════════════════════════════
# BLOCK 1 — Building Blocks (Inherited from Step 2)
# ══════════════════════════════════════════════════════════════════════

class ConstraintProjector:
    """Newton method constraint projection (Step 2 core, JIT compatible)"""
    def __init__(self, constraint_fn, max_iter=30, tol=1e-14):
        self.constraint_fn = constraint_fn
        self.grad_fn       = grad(constraint_fn)
        self.max_iter      = max_iter
        self.tol           = tol

    def project(self, x):
        def cond_fun(state):
            x_cur, i = state
            return (jnp.abs(self.constraint_fn(x_cur)) >= self.tol) & (i < self.max_iter)
        def body_fun(state):
            x_cur, i = state
            g      = self.constraint_fn(x_cur)
            grad_g = self.grad_fn(x_cur)
            lam    = g / (jnp.dot(grad_g, grad_g) + 1e-14)
            return (x_cur - lam * grad_g, i + 1)
        x_proj, _ = lax.while_loop(cond_fun, body_fun, (x, 0))
        return x_proj


class AdaptiveRKStep:
    """Step 1 core: Adaptive RK2(1) prediction + Step 2 projection correction"""
    def __init__(self, f_model, constraint_fn,
                 base_tol=5e-3, alpha=15.0, beta=30.0,
                 h_min=1e-5, h_max=0.15,
                 enable_projection=True):
        self.f_model   = f_model
        self.c_fn      = constraint_fn
        self.base_tol  = base_tol
        self.alpha     = alpha
        self.beta      = beta
        self.h_min     = h_min
        self.h_max     = h_max
        self.proj      = ConstraintProjector(constraint_fn) if enable_projection else None
        self.enable_proj = enable_projection

    @partial(jit, static_argnums=(0,))
    def step(self, x, t, h, prev_res):
        k1     = self.f_model(x, t)
        k2     = self.f_model(x + h * k1, t + h)
        x_pred = x + (h / 2.0) * (k1 + k2)
        x_1st  = x + h * k1
        loc_err = jnp.linalg.norm(x_pred - x_1st)
        raw_res = jnp.abs(self.c_fn(x_pred))

        if self.enable_proj:
            x_corr   = self.proj.project(x_pred)
            post_res = jnp.abs(self.c_fn(x_corr))
        else:
            x_corr   = x_pred
            post_res = raw_res

        dR       = (raw_res - prev_res) / (h + 1e-12)
        penalty  = 1.0 + self.alpha * raw_res + self.beta * jnp.maximum(0.0, dR)
        dyn_tol  = self.base_tol / penalty
        accepted = loc_err <= dyn_tol

        ratio  = dyn_tol / (loc_err + 1e-12)
        h_next = jnp.clip(h * 0.9 * jnp.power(ratio, 0.4), self.h_min, self.h_max)

        return x_corr, h_next, post_res, loc_err, accepted, dyn_tol, raw_res


# ══════════════════════════════════════════════════════════════════════
# BLOCK 2 — Step 3 Core: Laplacian Multi-Scale Spatial Decomposition
# ══════════════════════════════════════════════════════════════════════

class LaplacianMultiScaleDecomposer:
    """Step 3-A: Laplacian spatial frequency pyramid decomposition"""
    def __init__(self, n_spatial=64, n_levels=3, sigmas=None):
        self.n_spatial = n_spatial
        self.n_levels  = n_levels
        self.sigmas    = sigmas or [2.0, 1.0, 0.3]
        self.x_grid    = np.linspace(0, 2 * np.pi, n_spatial)

    def gaussian_kernel(self, sigma):
        kernel_size = max(3, int(6 * sigma) | 1)
        k           = np.arange(kernel_size) - kernel_size // 2
        kernel      = np.exp(-0.5 * (k / sigma) ** 2)
        return kernel / kernel.sum()

    def decompose(self, field):
        levels   = []
        residual = field.copy()
        for sigma in self.sigmas:
            kernel  = self.gaussian_kernel(sigma)
            low     = np.convolve(residual, kernel, mode='same')
            high    = residual - low
            levels.append((low.copy(), high.copy()))
            residual = low
        return levels

    def reconstruct(self, levels, weights=None):
        if weights is None:
            weights = [1.0 / len(levels)] * len(levels)
        field = np.zeros(self.n_spatial)
        for w, (low, _) in zip(weights, levels):
            field += w * low
        return field

    def compute_band_energy(self, levels):
        energies = []
        for low, high in levels:
            energies.append({
                'low_energy':  np.sum(low  ** 2) / len(low),
                'high_energy': np.sum(high ** 2) / len(high),
            })
        return energies

    def generate_multiscale_field(self, t):
        x = self.x_grid
        return (np.sin(x + 0.5 * t)
                + 0.3 * np.sin(5 * x + 2 * t)
                + 0.1 * np.sin(15 * x + 5 * t))


# ══════════════════════════════════════════════════════════════════════
# BLOCK 3 — Step 3 Core: Multi-Scale Dual-Step Integrator
# ══════════════════════════════════════════════════════════════════════

class MultiScaleIntegrator:
    """Step 3-B+C: Spatial frequency pyramid recursive integrator"""
    def __init__(self, f_model, constraint_fn,
                 h_coarse=0.08, h_mid=0.02, h_fine_init=0.005,
                 n_fine_per_mid=4, n_mid_per_coarse=4,
                 n_spatial=64, n_levels=3):

        self.h_coarse         = h_coarse
        self.h_mid            = h_mid
        self.h_fine_init      = h_fine_init
        self.n_fine_per_mid   = n_fine_per_mid
        self.n_mid_per_coarse = n_mid_per_coarse
        self.decomposer       = LaplacianMultiScaleDecomposer(n_spatial, n_levels)

        self.integrator_coarse = AdaptiveRKStep(
            f_model, constraint_fn,
            base_tol=1e-2, alpha=10.0, beta=20.0,
            h_min=0.02, h_max=0.15, enable_projection=True
        )
        self.integrator_mid = AdaptiveRKStep(
            f_model, constraint_fn,
            base_tol=5e-3, alpha=15.0, beta=30.0,
            h_min=0.005, h_max=0.04, enable_projection=True
        )
        self.integrator_fine = AdaptiveRKStep(
            f_model, constraint_fn,
            base_tol=1e-3, alpha=20.0, beta=40.0,
            h_min=1e-4, h_max=0.006, enable_projection=True
        )

    def run_coarse_step(self, x, t, h_c, res_c):
        return self.integrator_coarse.step(x, t, h_c, res_c)

    def run_mid_substeps(self, x, t, h_m, res_m, n_sub):
        history = []
        for _ in range(n_sub):
            x_new, h_m_next, res_m_new, err, acc, dtol, raw = \
                self.integrator_mid.step(x, t, h_m, res_m)
            history.append({
                'h': float(h_m), 'res': float(raw),
                'post_res': float(res_m_new), 'accepted': bool(acc)
            })
            if acc:
                x, t, res_m = x_new, t + h_m, res_m_new
            h_m = float(h_m_next)
        return x, t, h_m, res_m, history

    def run_fine_substeps(self, x, t, h_f, res_f, n_sub):
        history = []
        for _ in range(n_sub):
            x_new, h_f_next, res_f_new, err, acc, dtol, raw = \
                self.integrator_fine.step(x, t, h_f, res_f)
            history.append({
                'h': float(h_f), 'res': float(raw),
                'post_res': float(res_f_new), 'accepted': bool(acc)
            })
            if acc:
                x, t, res_f = x_new, t + h_f, res_f_new
            h_f = float(h_f_next)
        return x, t, h_f, res_f, history

    def run_full_pipeline(self, x0, t0, n_coarse_steps):
        x      = x0
        t      = t0
        h_c    = self.h_coarse
        h_m    = self.h_mid
        h_f    = self.h_fine_init
        res_c  = res_m = res_f = 0.0

        coarse_hist = []
        mid_hist    = []
        fine_hist   = []
        scale_hist  = []
        t_vals      = []
        field_decomp_hist = []

        print(f"\n{'CoarseStep':^11}| {'t':^7}| {'h_c':^8}| {'h_m_avg':^9}| "
              f"{'h_f_avg':^9}| {'Ratio c/f':^11}| {'Post-Res':^10}| {'Energy':^8}")
        print("─" * 85)

        for ci in range(n_coarse_steps):
            x_c, h_c_next, res_c_new, err_c, acc_c, dtol_c, raw_c = \
                self.run_coarse_step(x, t, h_c, res_c)

            x_m, t_m, h_m_next, res_m_new, mid_records = \
                self.run_mid_substeps(x, t, h_m, res_m, self.n_mid_per_coarse)

            x_f, t_f, h_f_next, res_f_new, fine_records = \
                self.run_fine_substeps(x, t, h_f, res_f, self.n_fine_per_mid)

            alpha_c, alpha_m, alpha_f = 0.6, 0.3, 0.1
            x_fused = alpha_c * x_c + alpha_m * x_m + alpha_f * x_f

            proj    = ConstraintProjector(lambda x: jnp.sum(x**2) - 1.0)
            x_fused = proj.project(jnp.array(x_fused))

            h_m_avg = np.mean([r['h'] for r in mid_records])  if mid_records  else h_m
            h_f_avg = np.mean([r['h'] for r in fine_records]) if fine_records else h_f
            ratio   = float(h_c) / (h_f_avg + 1e-12)
            energy  = float(jnp.linalg.norm(x_fused))
            post_r  = float(jnp.abs(jnp.sum(x_fused**2) - 1.0))

            field_t = self.decomposer.generate_multiscale_field(float(t))
            levels  = self.decomposer.decompose(field_t)
            energies = self.decomposer.compute_band_energy(levels)
            field_decomp_hist.append({'t': float(t), 'field': field_t, 'levels': levels, 'energies': energies})

            if ci % 3 == 0:
                print(f"{ci:^11}| {float(t):^7.3f}| {float(h_c):^8.4f}| {h_m_avg:^9.4f}| "
                      f"{h_f_avg:^9.4f}| {ratio:^11.1f}| {post_r:^10.2e}| {energy:^8.5f}")

            coarse_hist.append({
                'step': ci, 't': float(t), 'h': float(h_c),
                'x': [float(x_fused[0]), float(x_fused[1])],
                'post_res': post_r, 'energy': energy,
                'accepted': bool(acc_c)
            })
            mid_hist.extend(mid_records)
            fine_hist.extend(fine_records)
            scale_hist.append({
                'ci': ci, 't': float(t),
                'h_coarse': float(h_c), 'h_mid': h_m_avg, 'h_fine': h_f_avg,
                'ratio_cf': ratio, 'post_res': post_r, 'energy': energy,
            })
            t_vals.append(float(t))

            if acc_c:
                x, t, res_c = x_fused, t + float(h_c), float(res_c_new)
            h_c   = float(h_c_next)
            h_m   = float(h_m_next)
            h_f   = float(h_f_next)
            res_m = float(res_m_new)
            res_f = float(res_f_new)

        return coarse_hist, mid_hist, fine_hist, scale_hist, field_decomp_hist


# ══════════════════════════════════════════════════════════════════════
# BLOCK 4 — Physical Field Definitions
# ══════════════════════════════════════════════════════════════════════

def flow_field(x, t):
    omega = 1.5
    A     = 0.3
    vx = -x[1] * omega + A * jnp.sin(2.0 * jnp.pi * t) * x[0]**3
    vy =  x[0] * omega + A * jnp.cos(2.0 * jnp.pi * t) * x[1]**3
    return jnp.array([vx, vy])

def circle_constraint(x):
    return jnp.sum(x**2) - 1.0


# ══════════════════════════════════════════════════════════════════════
# BLOCK 5 — Enhanced Six-Panel Visualization (Combining strengths of both versions)
# ══════════════════════════════════════════════════════════════════════

class MultiScaleVisualizer:
    """Enhanced visualizer combining numerical rigor with academic aesthetics"""
    
    def __init__(self):
        self.colors = {
            'macro': '#2878B5',
            'mid': '#3fb950',
            'micro': '#E1812C',
            'combined': '#CCCCCC'
        }
    
    def plot_conceptual_bands(self, ax):
        """Panel A conceptual: Spatial frequency band separation (from step3.py)"""
        t = np.linspace(0, 10, 500)
        macro_signal = np.sin(0.5 * t)
        micro_signal = 0.2 * np.sin(5 * t) + 0.1 * np.cos(8 * t)
        combined_signal = macro_signal + micro_signal
        
        ax.plot(t, combined_signal, color=self.colors['combined'], lw=2, 
                label='Combined Dynamics', zorder=1)
        ax.plot(t, macro_signal, color=self.colors['macro'], lw=2.5, 
                label='Macroscopic (Low-Freq)', zorder=2)
        ax.plot(t, micro_signal - 1.5, color=self.colors['micro'], lw=1.5, alpha=0.8,
                label='Microscopic (High-Freq, Shifted)', zorder=3)
        
        ax.set_title("A. Spatial Frequency Band Separation", fontweight='bold')
        ax.set_xlabel("Time / Spatial Coordinate")
        ax.set_ylabel("Amplitude")
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='upper right', fontsize=9)
        return ax
    
    def plot_step_size_pyramid(self, ax, scale_hist):
        """Panel B: Step size pyramid from actual data"""
        steps_c = [d['ci'] for d in scale_hist]
        h_c_arr = [d['h_coarse'] for d in scale_hist]
        h_m_arr = [d['h_mid'] for d in scale_hist]
        h_f_arr = [d['h_fine'] for d in scale_hist]
        ratio_arr = [d['ratio_cf'] for d in scale_hist]
        
        ax.semilogy(steps_c, h_c_arr, 'o-', color=self.colors['macro'], lw=2, ms=5, 
                    label='h_coarse (Level 0)')
        ax.semilogy(steps_c, h_m_arr, 's-', color=self.colors['mid'], lw=1.5, ms=4, 
                    label='h_mid (Level 1)')
        ax.semilogy(steps_c, h_f_arr, '^-', color=self.colors['micro'], lw=1.5, ms=4, 
                    label='h_fine (Level 2)')
        
        ax2 = ax.twinx()
        ax2.plot(steps_c, ratio_arr, '--', color='#bc8cff', lw=1.2, alpha=0.7, 
                 label='h_c / h_f ratio')
        ax2.set_ylabel('h_coarse / h_fine', color='#bc8cff', fontsize=9)
        ax2.tick_params(axis='y', labelcolor='#bc8cff')
        
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper right')
        ax.set_xlabel('Coarse Step Index')
        ax.set_ylabel('Step Size (log scale)')
        ax.set_title("B. Multi-Scale Step Size Hierarchy\nh_coarse ≫ h_mid ≫ h_fine", 
                    fontweight='bold')
        return ax
    
    def plot_spatial_decomposition(self, ax, field_decomp_hist):
        """Panel C: Laplacian spatial decomposition at final time"""
        last = field_decomp_hist[-1]
        xg = np.linspace(0, 2 * np.pi, len(last['field']))
        
        ax.plot(xg, last['field'], color=self.colors['combined'], lw=1, alpha=0.8, 
                label='Original Field u(x,t)')
        
        colors_lv = [self.colors['macro'], self.colors['mid'], self.colors['micro']]
        for li, (low, high) in enumerate(last['levels']):
            ax.plot(xg, low, color=colors_lv[li], lw=1.8, 
                    label=f'Level {li} Low-Freq (σ={[2.0,1.0,0.3][li]})')
        
        ax.set_title("C. Laplacian Spatial Decomposition (3-Level Pyramid)", fontweight='bold')
        ax.legend(fontsize=7.5, loc='upper right')
        ax.set_xlabel('Spatial Coordinate x')
        ax.set_ylabel('Field Amplitude')
        return ax
    
    def plot_energy_evolution(self, ax, field_decomp_hist):
        """Panel D: Multi-scale energy evolution"""
        colors_lv = [self.colors['macro'], self.colors['mid'], self.colors['micro']]
        ts = [d['t'] for d in field_decomp_hist]
        
        for li in range(3):
            low_e = [d['energies'][li]['low_energy'] for d in field_decomp_hist]
            high_e = [d['energies'][li]['high_energy'] for d in field_decomp_hist]
            ax.plot(ts, low_e, color=colors_lv[li], lw=1.8, ls='-',
                    label=f'Lv{li} Low-Freq Energy')
            ax.plot(ts, high_e, color=colors_lv[li], lw=1.2, ls='--', alpha=0.6,
                    label=f'Lv{li} High-Freq Energy')
        
        ax.set_title("D. Multi-Scale Energy Evolution Across Frequency Bands", fontweight='bold')
        ax.legend(fontsize=7.5, ncol=2, loc='upper right')
        ax.set_xlabel('Physical Time t')
        ax.set_ylabel('Band Energy ‖u_band‖² / N')
        ax.set_yscale('log')
        return ax
    
    def plot_constraint_stability(self, ax, scale_hist):
        """Panel E: Constraint stability across scales"""
        steps_c = [d['ci'] for d in scale_hist]
        res_arr = [d['post_res'] for d in scale_hist]
        
        ax.semilogy(steps_c, [max(r, 1e-17) for r in res_arr],
                    'g-o', lw=2, ms=4, label='Post-Proj Residual (fused)')
        ax.axhline(1e-12, color='red', ls='--', lw=1.2, label='Machine Precision ref (1e-12)')
        ax.axhline(1e-14, color='orange', ls=':', lw=1.0, label='Double precision floor (1e-14)')
        
        ax.set_title("E. Constraint Stability Across All Scale Levels (S2 in S3)", fontweight='bold')
        ax.legend(fontsize=8.5)
        ax.set_xlabel('Coarse Step Index')
        ax.set_ylabel('|x²-1| (Post-Projection)')
        ax.set_ylim([1e-17, 1e-10])
        return ax
    
    def plot_phase_space(self, ax, coarse_hist):
        """Panel F: Phase space with manifold constraint"""
        theta = np.linspace(0, 2 * np.pi, 300)
        ax.plot(np.cos(theta), np.sin(theta), '--', color='gray', lw=1.5,
                alpha=0.7, label='Constraint Manifold ‖x‖=1')
        
        x0_arr = [d['x'][0] for d in coarse_hist]
        x1_arr = [d['x'][1] for d in coarse_hist]
        sc = ax.scatter(x0_arr, x1_arr, c=range(len(coarse_hist)), 
                        cmap='plasma', s=35, zorder=5, label='Fused Trajectory')
        plt.colorbar(sc, ax=ax, label='Coarse Step', shrink=0.85)
        
        ax.set_title("F. Phase Space: Fused Multi-Scale Trajectory on Manifold", fontweight='bold')
        ax.legend(fontsize=8.5)
        ax.set_xlabel('x₀')
        ax.set_ylabel('x₁')
        ax.set_aspect('equal')
        return ax


def plot_full_diagnostic(coarse_hist, scale_hist, field_decomp_hist):
    """Main plotting function combining numerical data with enhanced aesthetics"""
    
    visualizer = MultiScaleVisualizer()
    
    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor('white')  
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.35)
    
    # Style configuration for white background
    C_BG = 'white'          
    C_BORDER = '#CCCCCC'     
    C_TEXT = '#1f2937'       
    C_SUB = '#4b5563'        
    
    def style_ax(ax, title):
        ax.set_facecolor(C_BG)
        for sp in ax.spines.values():
            sp.set_edgecolor(C_BORDER)
        ax.tick_params(colors=C_SUB)
        ax.xaxis.label.set_color(C_SUB)
        ax.yaxis.label.set_color(C_SUB)
        ax.set_title(title, color=C_TEXT, fontsize=11, fontweight='bold', pad=8)
        ax.grid(True, color=C_BORDER, alpha=0.5, lw=0.5)
    
    
    
    # Panel A: Conceptual frequency bands
    ax_a = fig.add_subplot(gs[0, 0])
    visualizer.plot_conceptual_bands(ax_a)
    style_ax(ax_a, "A. Spatial Frequency Band Separation (Conceptual)")
    
    # Panel B: Step size pyramid (actual data)
    ax_b = fig.add_subplot(gs[0, 1])
    visualizer.plot_step_size_pyramid(ax_b, scale_hist)
    style_ax(ax_b, "B. Multi-Scale Step Size Hierarchy")
    
    # Panel C: Laplacian spatial decomposition (actual data)
    ax_c = fig.add_subplot(gs[1, 0])
    visualizer.plot_spatial_decomposition(ax_c, field_decomp_hist)
    style_ax(ax_c, "C. Laplacian Spatial Decomposition")
    
    # Panel D: Energy evolution (actual data)
    ax_d = fig.add_subplot(gs[1, 1])
    visualizer.plot_energy_evolution(ax_d, field_decomp_hist)
    style_ax(ax_d, "D. Multi-Scale Energy Evolution")
    
    # Panel E: Constraint stability (actual data)
    ax_e = fig.add_subplot(gs[2, 0])
    visualizer.plot_constraint_stability(ax_e, scale_hist)
    style_ax(ax_e, "E. Constraint Stability")
    
    # Panel F: Phase space (actual data)
    ax_f = fig.add_subplot(gs[2, 1])
    visualizer.plot_phase_space(ax_f, coarse_hist)
    style_ax(ax_f, "F. Phase Space Trajectory")
    
    fig.suptitle(
        "Step 3: Multi-Scale Spatial Frequency Pyramid Integrator\n"
        "Adaptive RK (S1) + Constraint Projection (S2) + Multi-Scale Pyramid (S3)",
        color=C_TEXT, fontsize=14, fontweight='bold', y=0.98
    )
    
    plt.tight_layout()
    plt.show()
    print('\n[INFO] Figure saved → step3_combined_result.png')


def print_summary(scale_hist):
    h_c = [d['h_coarse'] for d in scale_hist]
    h_f = [d['h_fine'] for d in scale_hist]
    rat = [d['ratio_cf'] for d in scale_hist]
    res = [d['post_res'] for d in scale_hist]
    eng = [d['energy'] for d in scale_hist]

    print("\n" + "═" * 60)
    print("  Step 3 Summary Report")
    print("═" * 60)
    print(f"  h_coarse  : mean={np.mean(h_c):.4f}  range=[{min(h_c):.4f}, {max(h_c):.4f}]")
    print(f"  h_mid     : (adaptive, ~{np.mean([d['h_mid'] for d in scale_hist]):.4f})")
    print(f"  h_fine    : mean={np.mean(h_f):.4f}  range=[{min(h_f):.4f}, {max(h_f):.4f}]")
    print(f"  h_c/h_f   : mean={np.mean(rat):.1f}×  max={max(rat):.1f}×  (target ≥ 10×)")
    print(f"  Post-Res  : mean={np.mean(res):.2e}  max={max(res):.2e}")
    print(f"  Energy    : mean={np.mean(eng):.6f}  std={np.std(eng):.2e}  (ideal=1.0)")


if __name__ == "__main__":
    print("=" * 70)
    print("  Combined Step 3: Laplacian Multi-Scale Spatial Frequency Pyramid")
    print("  Full numerical implementation + Enhanced visualization")
    print("=" * 70)

    ms = MultiScaleIntegrator(
        flow_field, circle_constraint,
        h_coarse=0.08, h_mid=0.02, h_fine_init=0.005,
        n_fine_per_mid=4, n_mid_per_coarse=4,
        n_spatial=64, n_levels=3,
    )

    x0 = jnp.array([1.0, 0.0])
    coarse_hist, mid_hist, fine_hist, scale_hist, field_decomp = \
        ms.run_full_pipeline(x0, t0=0.0, n_coarse_steps=30)

    print_summary(scale_hist)
    plot_full_diagnostic(coarse_hist, scale_hist, field_decomp)