import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Arc

# ==========================================
# Global figure style configuration
# ==========================================
plt.rcParams.update({
    "figure.dpi": 150,                
    "figure.figsize": (14, 6),       #
    "font.size": 8,                  
    "axes.titlesize": 8,             
    "axes.titleweight": "bold",       
    "axes.spines.top": False,         
    "axes.spines.right": False,       
    "legend.framealpha": 0.9,         
})

class ProjectionMechanismVisualizer:
    def __init__(self):
        self.fig, self.axes = plt.subplots(1, 2)
        self.fig.suptitle("Step 2: Physics-Constrained Flow Matching - Step Projection Mechanism", 
                          fontsize=12, fontweight='bold', y=0.98)
        
        # Define the physical constraint manifold (e.g., energy-conserving circle x^2 + y^2 = R^2)
        self.R = 1.0 
        
    def plot_microscopic_mechanism(self):
        """Panel A: Microscopic Mechanism - Demonstrates the geometry of single-step integration and orthogonal projection"""
        ax = self.axes[0]
        ax.set_aspect('equal')
        
        # 1. Draw the constraint manifold
        theta = np.linspace(0, np.pi/2, 100)
        ax.plot(self.R * np.cos(theta), self.R * np.sin(theta), 
                color='#888888', lw=2.5, linestyle='--', label='Constraint Manifold $\mathcal{M}$ (e.g. $H(x)=0$)')
        
        # 2. Define key points
        # Current point x_n (on the manifold)
        x_n = np.array([np.cos(1.2), np.sin(1.2)]) * self.R
        
        # Vector field direction (tangential)
        v_n = np.array([-np.sin(1.2), np.cos(1.2)]) 
        
        # Euler predictor point x_n+1^* (deviates from manifold)
        h = 0.4 # Exaggerated step size for visibility
        x_pred = x_n + h * v_n
        
        # Projected point x_n+1 (pulled back to manifold)
        x_proj = x_pred / np.linalg.norm(x_pred) * self.R
        
        # 3. Plot points and vectors
        # Current point
        ax.plot(x_n[0], x_n[1], 'ko', markersize=7)
        ax.annotate("$x_t$", xy=x_n, xytext=(x_n[0]-0.1, x_n[1]-0.1), fontsize=12, fontweight='bold')
        
        # Unconstrained Step (ODE Predictor)
        ax.annotate("", xy=x_pred, xytext=x_n,
                    arrowprops=dict(arrowstyle="->", color='#E1812C', lw=2.5))
        ax.plot(x_pred[0], x_pred[1], 'o', color='#E1812C', markersize=7)
        ax.annotate("$\widetilde{x}_{t+\Delta t}$\n(Unconstrained\nPredictor)", 
                    xy=x_pred, xytext=(x_pred[0]+0.05, x_pred[1]), color='#E1812C', fontsize=11)
        
        # Projection Step (Corrector)
        ax.annotate("", xy=x_proj, xytext=x_pred,
                    arrowprops=dict(arrowstyle="->", color='#32B897', lw=2.5, linestyle='--'))
        ax.plot(x_proj[0], x_proj[1], 'o', color='#32B897', markersize=7)
        ax.annotate("$x_{t+\Delta t}$\n(Projected\nState)", 
                    xy=x_proj, xytext=(x_proj[0]+0.02, x_proj[1]-0.15), color='#1B7A63', fontsize=11)
        
        # 4. Panel decorations
        ax.set_xlim(0, 1.2)
        ax.set_ylim(0, 1.5)
        ax.set_title("A. Microscopic View: Predictor-Corrector Geometry")
        ax.set_xlabel("State Dimension 1")
        ax.set_ylabel("State Dimension 2")
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='upper right', fontsize=9)

    def plot_macroscopic_trajectory(self):
        """Panel B: Macroscopic Trajectory - Compares long-term flow matching effects with vs. without projection mechanism"""
        ax = self.axes[1]
        ax.set_aspect('equal')
        
        # 1. Draw the complete constraint manifold background
        circle = Circle((0, 0), self.R, fill=False, edgecolor='#888888', linestyle='--', lw=2, zorder=1)
        ax.add_patch(circle)
        
        # 2. Simulate long-term flow matching integration (with projection vs. without projection)
        n_steps = 50
        h = 0.15
        
        # Initial states
        x_unconstrained = np.array([1.0, 0.0])
        x_constrained = np.array([1.0, 0.0])
        
        traj_unconst = [x_unconstrained.copy()]
        traj_const = [x_constrained.copy()]
        
        for _ in range(n_steps):
            # Vector field (tangential rotational flow)
            v_u = np.array([-x_unconstrained[1], x_unconstrained[0]])
            v_c = np.array([-x_constrained[1], x_constrained[0]])
            
            # Unconstrained flow (error accumulation leads to deviation from manifold)
            x_unconstrained = x_unconstrained + h * v_u
            
            # Constrained flow (take a step first, then project)
            x_pred = x_constrained + h * v_c
            x_constrained = x_pred / np.linalg.norm(x_pred) * self.R # Project back onto circle of radius R
            
            traj_unconst.append(x_unconstrained.copy())
            traj_const.append(x_constrained.copy())
            
        traj_unconst = np.array(traj_unconst)
        traj_const = np.array(traj_const)
        
        # 3. Plot trajectories
        ax.plot(traj_unconst[:, 0], traj_unconst[:, 1], 'o-', color='#E1812C', 
                markersize=4, alpha=0.7, lw=1.5, label='Naive Flow (Constraint Violated)')
        
        ax.plot(traj_const[:, 0], traj_const[:, 1], 'o-', color='#32B897', 
                markersize=4, alpha=0.9, lw=2, label='Projected Flow (Constraint Satisfied)')
        
        # Mark the starting point
        ax.plot(1.0, 0.0, 'k*', markersize=12, label='Start State $x_0$', zorder=5)
        
        # 4. Panel decorations
        limit = 2.0
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_title("B. Macroscopic View: Long-term Integration Trajectory")
        ax.set_xlabel("State Dimension 1")
        # Hide Y-axis label on the right plot for cleaner appearance
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='lower left', fontsize=9)

    def render(self):
        self.plot_microscopic_mechanism()
        self.plot_macroscopic_trajectory()
        plt.tight_layout()
        plt.show()

# =====================================================================
# Execution entry point
# =====================================================================
if __name__ == "__main__":
    visualizer = ProjectionMechanismVisualizer()
    visualizer.render()