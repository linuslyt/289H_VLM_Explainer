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

import matplotlib.pyplot as plt
import numpy as np
import os

def plot_dataset_benchmark(results: dict, save_dir: str):
    """
    Plots aggregated curves for Deletion and Insertion from dataset-level stats.
    """
    token = results.get("token", "unknown")
    n_samples = results.get("samples_n", 0)
    os.makedirs(save_dir, exist_ok=True)
    
    plot_files = {}
    # Styles
    colors = {'ours': '#2563eb', 'random': '#dc2626', 'hist': '#4f46e5'}

    # --- 1. Aggregated Deletion Plot ---
    if "deletion" in results:
        data = results["deletion"]
        curve_ours = data["curve_ours_mean"]
        curve_rand = data.get("curve_rand_mean", []) # Handle missing if not updated yet
        
        plt.figure(figsize=(6, 4))
        x_axis = np.arange(len(curve_ours))
        
        # Plot Ours
        plt.plot(x_axis, curve_ours, label=f"Ours (Mean AUC={data['auc_ours_mean']:.2f})", 
                 color=colors['ours'], linewidth=2)
        
        # Plot Random (if available)
        if curve_rand:
            plt.plot(x_axis, curve_rand, label=f"Random (Mean AUC={data['auc_rand_mean']:.2f})", 
                     color=colors['random'], linestyle='--', linewidth=2)
            
        plt.title(f"Deletion Benchmark: '{token}' (N={n_samples})")
        plt.xlabel("Concepts Removed")
        plt.ylabel("Avg Logit (Evidence)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        del_filename = f"benchmark_deletion_{token}.png"
        plt.savefig(os.path.join(save_dir, del_filename), dpi=100)
        plt.close()
        plot_files["deletion_plot"] = del_filename

    # --- 2. Aggregated Insertion Plot ---
    if "insertion" in results:
        data = results["insertion"]
        curve_ours = data["curve_ours_mean"]
        curve_rand = data.get("curve_rand_mean", [])
        
        plt.figure(figsize=(6, 4))
        x_axis = np.arange(len(curve_ours))
        
        plt.plot(x_axis, curve_ours, label=f"Ours (Mean AUC={data['auc_ours_mean']:.2f})", 
                 color=colors['ours'], linewidth=2)
        
        if curve_rand:
            plt.plot(x_axis, curve_rand, label=f"Random (Mean AUC={data['auc_rand_mean']:.2f})", 
                     color=colors['random'], linestyle='--', linewidth=2)
            
        plt.title(f"Insertion Benchmark: '{token}' (N={n_samples})")
        plt.xlabel("Concepts Inserted")
        plt.ylabel("Avg Logit (Evidence)")
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        ins_filename = f"benchmark_insertion_{token}.png"
        plt.savefig(os.path.join(save_dir, ins_filename), dpi=100)
        plt.close()
        plot_files["insertion_plot"] = ins_filename
    
    # --- 3. Fidelity Histogram ---
    if "fidelity" in results and "raw_correlations" in results["fidelity"]:
        data = results["fidelity"]
        corrs = data["raw_correlations"]
        mean_corr = data["correlation_mean"]
        
        if len(corrs) > 0:
            plt.figure(figsize=(6, 4))
            
            # Plot Histogram of correlations
            # Bins=10 or 'auto' works well for N=50
            plt.hist(corrs, bins=10, color=colors['hist'], alpha=0.7, edgecolor='black')
            
            # Add vertical line for Mean
            plt.axvline(mean_corr, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_corr:.2f}')
            
            plt.title(f"Fidelity Distribution: '{token}' (N={n_samples})")
            plt.xlabel("Pearson Correlation (Higher is Better)")
            plt.ylabel("Number of Samples")
            plt.legend()
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            
            filename = f"benchmark_fidelity_{token}.png"
            plt.savefig(os.path.join(save_dir, filename), dpi=100)
            plt.close()
            plot_files["fidelity_plot"] = filename


    return plot_files