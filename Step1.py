import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict

# ==========================================
# Global figure style configuration
# ==========================================
plt.rcParams.update({
    "figure.dpi": 150,                
    "figure.figsize": (12, 8.5),      
    "font.size": 11,                  
    "axes.titlesize": 14,             
    "axes.titleweight": "bold",       
    "axes.spines.top": False,         
    "axes.spines.right": False,       
    "legend.framealpha": 0.9,         
    "legend.edgecolor": "#CCCCCC"     
})

# ==========================================
# Core Class 1: Numerical Solver Simulator (Data Engine)
# ==========================================
class AdaptiveSolverSimulator:
    def __init__(self, n_steps: int = 40, random_seed: int = 42):
        self.n_steps = n_steps
        self.steps = np.arange(n_steps)
        np.random.seed(random_seed)

    def generate_ideal_run(self) -> Dict[str, np.ndarray]:
        h_vals = 0.065 * np.exp(-self.steps / 8.0) + 0.02
        noise = np.random.normal(0, 0.0005, size=self.n_steps)
        res_vals = 0.008 * np.power(self.steps, 1.1) + noise
        res_vals = np.clip(res_vals, 0.002, None) 
        tol_vals = 0.0005 / (res_vals + 0.05)
        
        return {
            "steps": self.steps,
            "h": h_vals,
            "residual": res_vals,
            "tolerance": tol_vals
        }

# ==========================================
# Core Class 2: Academic Figure Renderer (Visualization Engine)
# ==========================================
class DiagnosticsVisualizer:
    def __init__(self, data: Dict[str, np.ndarray]):
        self.data = data
        self.fig, self.axes = plt.subplots(2, 1)
        self.x = data["steps"]

    def plot_step_dynamics(self):
        ax = self.axes[0]
        y_h = self.data["h"]
        
        ax.plot(self.x, y_h, marker='o', markersize=6, markerfacecolor='white', 
                markeredgewidth=1.5, color='#2878B5', lw=2, label='Step Size (h)')
        
        ax.set_title("A. Step Size Dynamics (Ideal Convergence)")
        ax.set_ylabel("Step Size Value")
        ax.set_ylim(0, 0.10)
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)
        ax.legend(loc='upper right')

    def plot_mirror_relationship(self):
        ax_left = self.axes[1]
        ax_right = ax_left.twinx() 
        
        ax_right.spines['right'].set_visible(True)
        ax_right.spines['right'].set_color('#888888')
        
        y_res = self.data["residual"]
        y_tol = self.data["tolerance"]
        
        ax_left.set_title("B. Mirror Relationship: Residual vs Dynamic Tolerance")
        
        color_res = '#E1812C'
        line1 = ax_left.plot(self.x, y_res, color=color_res, lw=2.5, label='Physics Residual (R)')
        ax_left.set_ylabel("Residual $R$", color=color_res, fontweight='bold')
        ax_left.tick_params(axis='y', colors=color_res)
        
        color_tol = '#32B897'
        line2 = ax_right.plot(self.x, y_tol, color=color_tol, ls='--', lw=2.5, label='Dynamic Tolerance (Tol)')
        
        ax_right.fill_between(self.x, 0, y_tol, color=color_tol, alpha=0.1, label='Safe Tolerance Zone')
        
        ax_right.set_ylabel("Tolerance $Tol$", color=color_tol, fontweight='bold')
        ax_right.tick_params(axis='y', colors=color_tol)
        ax_right.set_ylim(0, max(y_tol) * 1.2) 
        
        lines, labels = ax_left.get_legend_handles_labels()
        lines2, labels2 = ax_right.get_legend_handles_labels()
        ax_left.legend(lines + lines2, labels + labels2, loc='center right')
        
        self._add_curved_annotations(ax_left, ax_right, y_res, y_tol)

    def _add_curved_annotations(self, ax_left, ax_right, y_res, y_tol):
        # Fix: Explicitly declare the color_tol variable here
        color_tol = '#32B897' 
        
        points = [
            (5,  "Low Residual",  0.15, "High Tolerance\n(Allows large steps)", -0.0015),
            (25, "Rising Residual", 0.15, "Tightening Tolerance", -0.0015)
        ]
        
        for x_idx, r_text, r_off, t_text, t_off in points:
            arrow_props = dict(arrowstyle="-|>", color='#555555', 
                               connectionstyle="arc3,rad=0.2", lw=1.2)
            
            ax_left.annotate(r_text, xy=(x_idx, y_res[x_idx]), 
                             xytext=(x_idx - 2, y_res[x_idx] + r_off),
                             arrowprops=arrow_props, fontsize=10, 
                             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#E1812C", alpha=0.8))
            
            # 这里调用了 color_tol，现在它能正常找到了
            ax_right.annotate(t_text, xy=(x_idx, y_tol[x_idx]), 
                              xytext=(x_idx + 2, y_tol[x_idx] + t_off),
                              arrowprops=dict(arrowstyle="-|>", color=color_tol, connectionstyle="arc3,rad=-0.2", lw=1.2), 
                              fontsize=10, color='#1B7A63',
                              bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color_tol, alpha=0.8))

    def render(self):
        self.plot_step_dynamics()
        self.plot_mirror_relationship()
        
        for ax in self.axes:
            ax.set_xticks(np.arange(0, 45, 5))
        self.axes[1].set_xlabel("Integration Steps", fontweight='bold')
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    simulator = AdaptiveSolverSimulator(n_steps=40)
    sim_data = simulator.generate_ideal_run()
    
    visualizer = DiagnosticsVisualizer(sim_data)
    visualizer.render()