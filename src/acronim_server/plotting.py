import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Dict, Any

def generate_faithfulness_plots(
    results: Dict[str, Any], 
    save_dir: str, 
    token: str
) -> Dict[str, str]:
    """
    Generates PNG plots for Deletion, Insertion, and Fidelity.
    Returns a dictionary of filenames/paths.
    """
    os.makedirs(save_dir, exist_ok=True)
    plot_paths = {}
    
    # Common Style settings
    plt.style.use('seaborn-v0_8-whitegrid') # Or 'ggplot' if seaborn is not available
    colors = {'ours': '#2563eb', 'random': '#dc2626', 'scatter': '#4f46e5'} # Blue, Red, Indigo

    # ---------------------------------------------------------
    # 1. C-Deletion Plot
    # ---------------------------------------------------------
    if "c_deletion" in results:
        data = results["c_deletion"]
        traj_ours = data["trajectory_ours"]
        traj_rand = data["trajectory_random"]
        
        plt.figure(figsize=(6, 4))
        x_axis = np.linspace(0, len(traj_ours)-1, len(traj_ours))
        
        plt.plot(x_axis, traj_ours, label=f"Ours (AUC={data['auc_ours']:.2f})", color=colors['ours'], linewidth=2)
        plt.plot(x_axis, traj_rand, label=f"Random (AUC={data['auc_random']:.2f})", color=colors['random'], linestyle='--', linewidth=2)
        
        plt.title(f"C-Deletion: '{token}'")
        plt.xlabel("Concepts Removed")
        plt.ylabel("Logit (Evidence)")
        plt.legend()
        plt.tight_layout()
        
        filename = f"plot_deletion_{token}.png"
        path = os.path.join(save_dir, filename)
        plt.savefig(path, dpi=100)
        plt.close()
        plot_paths["c_deletion_plot"] = filename

    # ---------------------------------------------------------
    # 2. C-Insertion Plot
    # ---------------------------------------------------------
    if "c_insertion" in results:
        data = results["c_insertion"]
        traj_ours = data["trajectory_ours"]
        traj_rand = data["trajectory_random"]
        
        plt.figure(figsize=(6, 4))
        x_axis = np.linspace(0, len(traj_ours)-1, len(traj_ours))
        
        plt.plot(x_axis, traj_ours, label=f"Ours (AUC={data['auc_ours']:.2f})", color=colors['ours'], linewidth=2)
        plt.plot(x_axis, traj_rand, label=f"Random (AUC={data['auc_random']:.2f})", color=colors['random'], linestyle='--', linewidth=2)
        
        plt.title(f"C-Insertion: '{token}'")
        plt.xlabel("Concepts Inserted")
        plt.ylabel("Logit (Evidence)")
        plt.legend(loc='lower right')
        plt.tight_layout()
        
        filename = f"plot_insertion_{token}.png"
        path = os.path.join(save_dir, filename)
        plt.savefig(path, dpi=100)
        plt.close()
        plot_paths["c_insertion_plot"] = filename

    # ---------------------------------------------------------
    # 3. C-µFidelity Scatter Plot
    # ---------------------------------------------------------
    if "c_fidelity" in results and "predicted_deltas" in results["c_fidelity"]:
        data = results["c_fidelity"]
        x = data["predicted_deltas"] # Expected Drop
        y = data["actual_deltas"]    # Actual Drop
        
        plt.figure(figsize=(6, 4))
        
        # Scatter points
        plt.scatter(x, y, alpha=0.6, color=colors['scatter'], edgecolors='w', s=50)
        
        # Fit trend line
        if len(x) > 1:
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            plt.plot(x, p(x), "r--", alpha=0.5, label=f"Trend (Corr={data['correlation']:.2f})")
        
        plt.title(f"C-µFidelity: '{token}'")
        plt.xlabel("Expected Drop (Importance Sum)")
        plt.ylabel("Actual Drop (Model Output)")
        plt.legend()
        plt.tight_layout()
        
        filename = f"plot_fidelity_{token}.png"
        path = os.path.join(save_dir, filename)
        plt.savefig(path, dpi=100)
        plt.close()
        plot_paths["c_fidelity_plot"] = filename

    return plot_paths