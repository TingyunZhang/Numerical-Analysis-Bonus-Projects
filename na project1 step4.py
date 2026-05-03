import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Step 4: Aggregate W2 Bound for Riemannian Flow Matching
# ============================================================

def curvature_amplification(kappa, T, curvature_type="negative"):
    """
    Curvature-dependent Jacobi amplification factor.

    Parameters
    ----------
    kappa : float or np.ndarray
        Curvature magnitude. We assume |K| <= kappa^2.
    T : float
        Integration horizon, usually T = 1 - tau.
    curvature_type : str
        "negative", "zero", or "positive".

    Returns
    -------
    A : float or np.ndarray
        Curvature amplification factor.
    """

    kappa = np.asarray(kappa, dtype=float)
    x = kappa * T

    # Avoid division by zero
    eps = 1e-12

    if curvature_type == "negative":
        A = np.where(np.abs(x) < eps, 1.0, np.sinh(x) / x)

    elif curvature_type == "zero":
        A = np.ones_like(x)

    elif curvature_type == "positive":
        # Valid before conjugate points: kappa*T < pi
        A = np.where(np.abs(x) < eps, 1.0, np.sin(x) / x)
        A = np.maximum(A, 0.0)

    else:
        raise ValueError("curvature_type must be 'negative', 'zero', or 'positive'.")

    return A


def statistical_error(d, N, C0=1.0):
    """
    Finite-sample Wasserstein statistical error.

    E_stat <= C0 * sqrt(d) * N^{-1/d}
    """
    return C0 * np.sqrt(d) * N ** (-1.0 / d)


def field_error(d, N, kappa, tau, curvature_type="negative", C1=1.0):
    """
    Neural vector field L2 approximation error accumulated by the flow.

    E_field <= C1 * A_kappa(1-tau) * sqrt(d/N) * log(1/tau)
    """
    T = 1.0 - tau
    A = curvature_amplification(kappa, T, curvature_type)
    return C1 * A * np.sqrt(d / N) * np.log(1.0 / tau)


def numerical_error(h, p, kappa, tau, curvature_type="negative", C2=1.0):
    """
    Numerical truncation error for a p-th order solver.

    E_num <= (C2/p) * A_kappa(1-tau) * h^p * (tau^{-p} - 1)
    """
    T = 1.0 - tau
    A = curvature_amplification(kappa, T, curvature_type)
    return (C2 / p) * A * (h ** p) * (tau ** (-p) - 1.0)


def aggregate_W2_bound(
    d,
    N,
    kappa,
    h,
    tau,
    p=4,
    curvature_type="negative",
    C0=1.0,
    C1=1.0,
    C2=1.0
):
    """
    Master inequality:

    W2 <= E_stat + E_field + E_num
    """

    E_stat = statistical_error(d, N, C0)
    E_field = field_error(d, N, kappa, tau, curvature_type, C1)
    E_num = numerical_error(h, p, kappa, tau, curvature_type, C2)

    total = E_stat + E_field + E_num

    return total, E_stat, E_field, E_num


# ============================================================
# Baseline parameters
# ============================================================

d0 = 10
N0 = 5000
kappa0 = 1.0
h0 = 0.01
tau0 = 0.05
p0 = 4

C0 = 0.3
C1 = 0.5
C2 = 0.1


# ============================================================
# Print baseline decomposition
# ============================================================

total, E_stat, E_field, E_num = aggregate_W2_bound(
    d=d0,
    N=N0,
    kappa=kappa0,
    h=h0,
    tau=tau0,
    p=p0,
    curvature_type="negative",
    C0=C0,
    C1=C1,
    C2=C2
)

print("Baseline aggregate W2 bound")
print("-------------------------------------")
print(f"d       = {d0}")
print(f"N       = {N0}")
print(f"kappa   = {kappa0}")
print(f"h       = {h0}")
print(f"tau     = {tau0}")
print(f"solver order p = {p0}")
print("-------------------------------------")
print(f"E_stat  = {E_stat:.6e}")
print(f"E_field = {E_field:.6e}")
print(f"E_num   = {E_num:.6e}")
print(f"Total W2 bound = {total:.6e}")

# ============================================================
# Master inequality (explicit closed-form expression)
# ============================================================
#
#   W2(mu_hat, mu*) <=
#
#       C0 * sqrt(d) * N^{-1/d}                              (E_stat)
#     + C1 * A_kappa(1-tau) * sqrt(d/N) * log(1/tau)         (E_field)
#     + (C2/p) * A_kappa(1-tau) * h^p * (tau^{-p} - 1)       (E_num)
#
#   where  A_kappa(T) = sinh(kappa*T) / (kappa*T)   [negative curvature]
#                     = 1                            [zero curvature]
#                     = sin(kappa*T)  / (kappa*T)   [positive curvature]
#
#   and T = 1 - tau  is the effective integration horizon.
#
# Symbolic rendering with baseline constants:

T0 = 1.0 - tau0
A0 = np.sinh(kappa0 * T0) / (kappa0 * T0)

print()
print("Master inequality (closed form, negative curvature):")
print("  W2 <= C0*sqrt(d)*N^{-1/d}  +  C1*A(kappa,tau)*sqrt(d/N)*log(1/tau)"
      "  +  (C2/p)*A(kappa,tau)*h^p*(tau^{-p}-1)")
print()
print("Term structure at baseline parameters:")
print(f"  A_kappa(1-tau) = sinh({kappa0}*{T0:.2f}) / ({kappa0}*{T0:.2f}) = {A0:.6f}")
print(f"  E_stat  = {C0}*sqrt({d0})*{N0}^(-1/{d0})                   = {E_stat:.6e}")
print(f"  E_field = {C1}*{A0:.4f}*sqrt({d0}/{N0})*log(1/{tau0})      = {E_field:.6e}")
print(f"  E_num   = ({C2}/{p0})*{A0:.4f}*{h0}^{p0}*({tau0}^(-{p0})-1) = {E_num:.6e}")
print(f"  Total   = {total:.6e}")
print()
print(f"  Dominant term: E_stat ({100*E_stat/total:.1f}%)"
      f"  |  E_field ({100*E_field/total:.1f}%)"
      f"  |  E_num ({100*E_num/total:.2f}%)")
print()
print("Structural note on kappa:")
print("  kappa enters ONLY via A(kappa,tau) — it affects E_field and E_num, but NOT E_stat.")
print(f"  E_stat = C0*sqrt(d)*N^(-1/d) is kappa-independent => {E_stat:.6e} (constant w.r.t. kappa)")
print(f"  At kappa~0 (flat): A=1.000 => E_field+E_num floor = {E_field + E_num:.6e}")
kappa_high = 3.0
A_high = float(curvature_amplification(kappa_high, 1.0 - tau0, "negative"))
_, _, Ef_hi, En_hi = aggregate_W2_bound(d0, N0, kappa_high, h0, tau0, p0,
                                         "negative", C0, C1, C2)
print(f"  At kappa=3 (neg):  A={A_high:.4f} => E_field+E_num = {Ef_hi + En_hi:.6e}"
      f"  (x{(Ef_hi+En_hi)/(E_field+E_num):.1f} amplification vs flat)")


# ============================================================
# Figure 1: Effect of sample size N
# ============================================================

N_values = np.logspace(2, 6, 100)
bounds_N_neg = []
bounds_N_zero = []
bounds_N_pos = []

for N in N_values:
    bounds_N_neg.append(
        aggregate_W2_bound(d0, N, kappa0, h0, tau0, p0,
                           "negative", C0, C1, C2)[0]
    )
    bounds_N_zero.append(
        aggregate_W2_bound(d0, N, kappa0, h0, tau0, p0,
                           "zero", C0, C1, C2)[0]
    )
    bounds_N_pos.append(
        aggregate_W2_bound(d0, N, kappa0, h0, tau0, p0,
                           "positive", C0, C1, C2)[0]
    )

plt.figure(figsize=(7, 5))
plt.loglog(N_values, bounds_N_neg, label="Negative curvature")
plt.loglog(N_values, bounds_N_zero, label="Zero curvature")
plt.loglog(N_values, bounds_N_pos, label="Positive curvature")
plt.xlabel("Sample size N")
plt.ylabel("Aggregate W2 bound")
plt.title("Effect of sample size N on aggregate W2 error")
plt.legend()
plt.grid(True, which="both", alpha=0.3)
plt.show()


# ============================================================
# Figure 2: Effect of dimension d
# ============================================================

d_values = np.arange(2, 101)
bounds_d_neg = []
bounds_d_zero = []
bounds_d_pos = []

for d in d_values:
    bounds_d_neg.append(
        aggregate_W2_bound(d, N0, kappa0, h0, tau0, p0,
                           "negative", C0, C1, C2)[0]
    )
    bounds_d_zero.append(
        aggregate_W2_bound(d, N0, kappa0, h0, tau0, p0,
                           "zero", C0, C1, C2)[0]
    )
    bounds_d_pos.append(
        aggregate_W2_bound(d, N0, kappa0, h0, tau0, p0,
                           "positive", C0, C1, C2)[0]
    )

plt.figure(figsize=(7, 5))
plt.plot(d_values, bounds_d_neg, label="Negative curvature")
plt.plot(d_values, bounds_d_zero, label="Zero curvature")
plt.plot(d_values, bounds_d_pos, label="Positive curvature")
plt.xlabel("Intrinsic dimension d")
plt.ylabel("Aggregate W2 bound")
plt.title("Effect of intrinsic dimension d on aggregate W2 error")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


# ============================================================
# Figure 3: Effect of curvature magnitude kappa
# ============================================================

kappa_values = np.linspace(0.01, 3.0, 100)
bounds_k_neg = []
bounds_k_pos = []

for kappa in kappa_values:
    bounds_k_neg.append(
        aggregate_W2_bound(d0, N0, kappa, h0, tau0, p0,
                           "negative", C0, C1, C2)[0]
    )
    bounds_k_pos.append(
        aggregate_W2_bound(d0, N0, kappa, h0, tau0, p0,
                           "positive", C0, C1, C2)[0]
    )

fig3, ax3 = plt.subplots(figsize=(8, 6))

ax3.plot(kappa_values, bounds_k_neg, label="Negative curvature")
ax3.plot(kappa_values, bounds_k_pos, label="Positive curvature")

# --- Decompose at kappa=0.01 (near-flat) and kappa=3.0 (high curvature) ---
for kappa_mark, xoff, yoff in [(0.5, 0.05, 0.004), (2.5, -0.55, -0.012)]:
    _, Es_neg, Ef_neg, En_neg = aggregate_W2_bound(
        d0, N0, kappa_mark, h0, tau0, p0, "negative", C0, C1, C2)
    _, Es_pos, Ef_pos, En_pos = aggregate_W2_bound(
        d0, N0, kappa_mark, h0, tau0, p0, "positive", C0, C1, C2)
    A_neg = curvature_amplification(kappa_mark, 1.0 - tau0, "negative")
    A_pos = curvature_amplification(kappa_mark, 1.0 - tau0, "positive")
    txt = (
        f"κ = {kappa_mark}\n"
        f"E_stat = {Es_neg:.3f}  [κ-independent]\n"
        f"A(κ,τ): neg={A_neg:.3f}, pos={A_pos:.3f}\n"
        f"E_field: neg={Ef_neg:.3f}, pos={Ef_pos:.3f}\n"
        f"E_num:  neg={En_neg:.2e}, pos={En_pos:.2e}"
    )
    wneg = sum(aggregate_W2_bound(d0, N0, kappa_mark, h0, tau0, p0,
                                   "negative", C0, C1, C2)[:1])
    ax3.annotate(
        txt,
        xy=(kappa_mark, wneg),
        xytext=(kappa_mark + xoff, wneg + yoff),
        fontsize=7.5,
        color="#333333",
        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray",
                  alpha=0.85, lw=0.6),
        arrowprops=dict(arrowstyle="->", lw=0.8, color="gray"),
    )

ax3.set_xlabel("Curvature magnitude kappa")
ax3.set_ylabel("Aggregate W2 bound")
ax3.set_title("Effect of curvature magnitude on aggregate W2 error")
ax3.legend(loc="upper left")
ax3.grid(True, alpha=0.3)

# --- E_stat baseline: horizontal dashed line showing the kappa-independent floor ---
E_stat_baseline = statistical_error(d0, N0, C0)
ax3.axhline(E_stat_baseline, color="gray", linestyle="--", linewidth=1.0,
            alpha=0.7, label=f"E_stat floor = {E_stat_baseline:.3f} (κ-independent)")
ax3.text(0.02, E_stat_baseline + 0.002, r"$E_{stat}$ floor (κ-independent)",
         fontsize=8, color="gray", va="bottom")
ax3.legend(loc="upper left", fontsize=8)

# --- In-axes structural note (replaces truncated fig.text) ---
note = (
    "Structural note:\n"
    r"κ enters bound only via $A(\kappa,\tau)$, affecting $E_{field}$ and $E_{num}$." + "\n"
    r"$E_{stat} \propto \sqrt{d}\,N^{-1/d}$ is κ-independent (dashed line)." + "\n"
    "Gap at κ→0 is the constant $E_{stat}$ floor."
)
ax3.text(0.98, 0.05, note,
         transform=ax3.transAxes,
         fontsize=7.8, color="#444444", style="italic",
         ha="right", va="bottom",
         bbox=dict(boxstyle="round,pad=0.4", fc="#f7f7f7", ec="#aaaaaa",
                   alpha=0.92, lw=0.6))

plt.tight_layout()
plt.show()


# ============================================================
# Figure 4: Effect of time step h
# ============================================================

h_values = np.logspace(-4, -1, 100)
bounds_h_p1 = []
bounds_h_p2 = []
bounds_h_p4 = []

for h in h_values:
    bounds_h_p1.append(
        aggregate_W2_bound(d0, N0, kappa0, h, tau0, 1,
                           "negative", C0, C1, C2)[0]
    )
    bounds_h_p2.append(
        aggregate_W2_bound(d0, N0, kappa0, h, tau0, 2,
                           "negative", C0, C1, C2)[0]
    )
    bounds_h_p4.append(
        aggregate_W2_bound(d0, N0, kappa0, h, tau0, 4,
                           "negative", C0, C1, C2)[0]
    )

plt.figure(figsize=(7, 5))
plt.loglog(h_values, bounds_h_p1, label="Euler p=1")
plt.loglog(h_values, bounds_h_p2, label="RK2 p=2")
plt.loglog(h_values, bounds_h_p4, label="RK4 p=4")
plt.xlabel("Step size h")
plt.ylabel("Aggregate W2 bound")
plt.title("Effect of solver step size h on aggregate W2 error")
plt.legend()
plt.grid(True, which="both", alpha=0.3)
plt.show()


# ============================================================
# Figure 5: Effect of early stopping tau
# ============================================================

tau_values = np.logspace(-3, -0.3, 100)
bounds_tau_neg = []
bounds_tau_zero = []
bounds_tau_pos = []

for tau in tau_values:
    bounds_tau_neg.append(
        aggregate_W2_bound(d0, N0, kappa0, h0, tau, p0,
                           "negative", C0, C1, C2)[0]
    )
    bounds_tau_zero.append(
        aggregate_W2_bound(d0, N0, kappa0, h0, tau, p0,
                           "zero", C0, C1, C2)[0]
    )
    bounds_tau_pos.append(
        aggregate_W2_bound(d0, N0, kappa0, h0, tau, p0,
                           "positive", C0, C1, C2)[0]
    )

plt.figure(figsize=(7, 5))
plt.loglog(tau_values, bounds_tau_neg, label="Negative curvature")
plt.loglog(tau_values, bounds_tau_zero, label="Zero curvature")
plt.loglog(tau_values, bounds_tau_pos, label="Positive curvature")
plt.xlabel("Early stopping tau")
plt.ylabel("Aggregate W2 bound")
plt.title("Effect of early stopping tau on aggregate W2 error")
plt.legend()
plt.grid(True, which="both", alpha=0.3)
plt.show()


# ============================================================
# Figure 6: Error decomposition as tau varies
# ============================================================

E_stat_list = []
E_field_list = []
E_num_list = []
E_total_list = []

for tau in tau_values:
    total, Es, Ef, En = aggregate_W2_bound(
        d=d0,
        N=N0,
        kappa=kappa0,
        h=h0,
        tau=tau,
        p=p0,
        curvature_type="negative",
        C0=C0,
        C1=C1,
        C2=C2
    )
    E_stat_list.append(Es)
    E_field_list.append(Ef)
    E_num_list.append(En)
    E_total_list.append(total)

plt.figure(figsize=(7, 5))
plt.loglog(tau_values, E_stat_list, label="Statistical error")
plt.loglog(tau_values, E_field_list, label="Field approximation error")
plt.loglog(tau_values, E_num_list, label="Numerical truncation error")
plt.loglog(tau_values, E_total_list, label="Total bound", linewidth=2)
plt.xlabel("Early stopping tau")
plt.ylabel("Error contribution")
plt.title("Error decomposition under negative curvature")
plt.legend()
plt.grid(True, which="both", alpha=0.3)
plt.show()


# ============================================================
# Figure 7a: Joint heatmap over d and N  (E_stat dominates)
# ============================================================

d_grid = np.arange(2, 51)
N_grid = np.logspace(2, 6, 60)

D_mat, N_mat = np.meshgrid(d_grid, N_grid)
Z_dN = np.zeros_like(D_mat, dtype=float)

for i in range(D_mat.shape[0]):
    for j in range(D_mat.shape[1]):
        Z_dN[i, j] = aggregate_W2_bound(
            d=int(D_mat[i, j]),
            N=N_mat[i, j],
            kappa=kappa0,
            h=h0,
            tau=tau0,
            p=p0,
            curvature_type="negative",
            C0=C0, C1=C1, C2=C2
        )[0]

plt.figure(figsize=(7, 5))
contour_dN = plt.contourf(D_mat, N_mat, np.log10(Z_dN), levels=30)
plt.yscale("log")
plt.xlabel("Intrinsic dimension d")
plt.ylabel("Sample size N")
plt.title(r"log10 W2 bound over d and N  ($E_{stat} \propto \sqrt{d}\,N^{-1/d}$ joint surface)")
plt.colorbar(contour_dN, label="log10 W2 bound")
plt.grid(True, which="both", alpha=0.2)
plt.tight_layout()
plt.show()


# ============================================================
# Figure 7b: Joint heatmap over kappa and tau (negative vs positive)
# ============================================================

kappa_grid = np.linspace(0.01, 3.0, 60)
tau_grid2  = np.logspace(-3, -0.3, 60)

KAP, TAU2 = np.meshgrid(kappa_grid, tau_grid2)
Z_ktau_neg = np.zeros_like(KAP, dtype=float)
Z_ktau_pos = np.zeros_like(KAP, dtype=float)

for i in range(KAP.shape[0]):
    for j in range(KAP.shape[1]):
        Z_ktau_neg[i, j] = aggregate_W2_bound(
            d=d0, N=N0, kappa=KAP[i, j], h=h0, tau=TAU2[i, j],
            p=p0, curvature_type="negative", C0=C0, C1=C1, C2=C2
        )[0]
        Z_ktau_pos[i, j] = aggregate_W2_bound(
            d=d0, N=N0, kappa=KAP[i, j], h=h0, tau=TAU2[i, j],
            p=p0, curvature_type="positive", C0=C0, C1=C1, C2=C2
        )[0]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, Z_k, subtitle in zip(
    axes,
    [Z_ktau_neg, Z_ktau_pos],
    ["negative curvature (sinh amplification)",
     "positive curvature (sin suppression)"]
):
    cf = ax.contourf(KAP, TAU2, np.log10(Z_k), levels=30)
    ax.set_yscale("log")
    ax.set_xlabel("Curvature magnitude kappa")
    ax.set_ylabel("Early stopping tau")
    ax.set_title(f"log10 W2 bound\n{subtitle}")
    fig.colorbar(cf, ax=ax, label="log10 W2 bound")
    ax.grid(True, which="both", alpha=0.2)
fig.suptitle(r"Joint $(\kappa,\,\tau)$ surface: cross-effect of curvature and early stopping",
             fontsize=12)
plt.tight_layout()
plt.show()


# ============================================================
# Figure 7c: Heatmap over h and tau
# ============================================================

h_grid = np.logspace(-4, -1, 80)
tau_grid = np.logspace(-3, -0.3, 80)

H, TAU = np.meshgrid(h_grid, tau_grid)
Z = np.zeros_like(H)

for i in range(TAU.shape[0]):
    for j in range(H.shape[1]):
        Z[i, j] = aggregate_W2_bound(
            d=d0,
            N=N0,
            kappa=kappa0,
            h=H[i, j],
            tau=TAU[i, j],
            p=p0,
            curvature_type="negative",
            C0=C0,
            C1=C1,
            C2=C2
        )[0]

plt.figure(figsize=(7, 5))
contour = plt.contourf(H, TAU, np.log10(Z), levels=30)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Step size h")
plt.ylabel("Early stopping tau")
plt.title("log10 Aggregate W2 bound over h and tau  (Fig 7c)")
plt.colorbar(contour, label="log10 W2 bound")
plt.show()


# ============================================================
# Figure 7d: Joint heatmap over h and d
# ============================================================
# Shows how the numerical truncation term E_num ~ h^p interacts
# with the statistical term E_stat ~ sqrt(d)*N^{-1/d} as d grows.
# At small h, E_stat dominates => curves are d-driven (vertical bands).
# At large h, E_num dominates => curves are h-driven (horizontal bands).

h_grid2 = np.logspace(-4, -1, 60)
d_grid2  = np.arange(2, 51)

H2, D2 = np.meshgrid(h_grid2, d_grid2)
Z_hd = np.zeros_like(H2, dtype=float)

for i in range(D2.shape[0]):
    for j in range(H2.shape[1]):
        Z_hd[i, j] = aggregate_W2_bound(
            d=int(D2[i, j]),
            N=N0,
            kappa=kappa0,
            h=H2[i, j],
            tau=tau0,
            p=p0,
            curvature_type="negative",
            C0=C0, C1=C1, C2=C2
        )[0]

fig7d, ax7d = plt.subplots(figsize=(8, 5))
cf7d = ax7d.contourf(H2, D2, np.log10(Z_hd), levels=30)
ax7d.set_xscale("log")
ax7d.set_xlabel("Step size h")
ax7d.set_ylabel("Intrinsic dimension d")
ax7d.set_title(
    r"log10 W2 bound over $h$ and $d$  (Fig 7d)"
    "\n"
    r"Left region: $E_{stat}$-dominated (d-driven vertical bands) | "
    r"Right: $E_{num}$-dominated (h-driven horizontal bands)"
)
fig7d.colorbar(cf7d, ax=ax7d, label="log10 W2 bound")
ax7d.grid(True, which="both", alpha=0.2)

# Annotate the two regimes
ax7d.text(2e-4, 45, r"$E_{stat}$-dominated" + "\n(d drives cost)",
          fontsize=8, color="white", ha="left", va="top",
          bbox=dict(boxstyle="round,pad=0.3", fc="navy", alpha=0.6, lw=0))
ax7d.text(5e-2, 10, r"$E_{num}$-dominated" + "\n(h drives cost)",
          fontsize=8, color="white", ha="left", va="top",
          bbox=dict(boxstyle="round,pad=0.3", fc="darkgreen", alpha=0.6, lw=0))

plt.tight_layout()
plt.show()