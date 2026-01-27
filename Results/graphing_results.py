import json
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import norm
from matplotlib.ticker import FormatStrFormatter



def plot_comparative_histograms(avg_file, eval_log_files, model_names, specie, dataset, output_dir="figures"):
    """
    Plot comparative histograms of model evaluation metrics (AUROC, AUPRC, Balanced Acc, F1 (Weighted))
    for each label across multiple models, formatted for black-and-white printing.
    """

    os.makedirs(output_dir, exist_ok=True)

    # Load averaged metrics
    with open(avg_file, "r") as f:
        avg_data = json.load(f)

    # Load evaluation logs
    eval_logs = []
    for file in eval_log_files:
        with open(file, "r") as f:
            eval_logs.append(json.load(f))

    # Only two styles: solid & hatched
    facecolors = ["#ff7f0e", "#1f77b4"]    # gray, white
    hatches = ["", "///"]             # plain, hatched

    # Labels (per-class metrics; skip mean_metrics)
    labels = [k for k in avg_data.keys() if k != "mean_metrics"]

    label_map = {
        "BALANCED_ACCURACY": "Balanced Acc",
        "WEIGHTED_F1": "F1 (Weighted)",
        "AUROC": "AUROC",
        "AUPRC": "AUPRC"
    }

    for label in labels:
        metrics = ["AUROC", "AUPRC", "BALANCED_ACCURACY", "WEIGHTED_F1"]
        short_metrics = [label_map[m] for m in metrics]

        means, errors = [], []

        # First model: averaged run
        avg_metrics = avg_data[label]
        model_means = [avg_metrics[m]["mean"] for m in metrics]
        model_stds = [avg_metrics[m]["std"] for m in metrics]
        means.append(model_means)
        errors.append(model_stds)

        # Other models: raw logs
        for log in eval_logs:
            per_label_list = [d for d in log["metrics_per_label"] if d["label"].lower() == label.lower()]
            if per_label_list:
                d = per_label_list[0]
                values = [d["auroc"], d["auprc"], d["balanced_accuracy"], d["weighted_f1"]]
            else:
                values = [np.nan] * 4

            means.append(values)
            errors.append([0.0] * 4)

        means = np.array(means)
        errors = np.array(errors)

        # Plot
        x = np.arange(len(metrics))
        width = 0.8 / len(model_names)

        fig, ax = plt.subplots(figsize=(3.35, 2.8), dpi=300)

        for i in range(len(model_names)):
            ax.bar(
            x + i * width - (len(model_names)-1)/2 * width,
            means[i],
            width=width,
            yerr=errors[i] if i == 0 else None,
            capsize=2,
            hatch=hatches[i % 2],
            edgecolor="black",
            facecolor=facecolors[i % 2],
            linewidth=0.8
            )



        ax.set_title(f"{specie} - {label}", fontsize=9, weight="bold", pad=5)
        ax.set_xticks(x)
        ax.set_xticklabels(short_metrics, rotation=25, ha="right", fontsize=7)
        ax.set_ylabel("Score", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.yaxis.grid(True, linestyle="--", alpha=0.6)

        # Save
        fname = os.path.join(output_dir, f"{specie}_{dataset}_{label.replace(' ', '_')}_comparison.png")
        plt.savefig(fname, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved {fname}")




def plot_paper_figure_four_panel(avg_file, eval_log_files, model_names, antibiotics, set, output_dir="figures"):
    """
    Create a 4-panel figure (full page width) using grayscale + hatch bars,
    no model names, no legend, B/W printing compatible.
    """

    
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11
    })



    os.makedirs(output_dir, exist_ok=True)

    with open(avg_file, "r") as f:
        avg_data = json.load(f)

    eval_logs = []
    for file in eval_log_files:
        with open(file, "r") as f:
            eval_logs.append(json.load(f))

    metrics = ["AUROC", "AUPRC", "BALANCED_ACCURACY", "WEIGHTED_F1"]
    label_map = {
        "BALANCED_ACCURACY": "Balanced Acc",
        "WEIGHTED_F1": "F1 (Weighted)",
        "AUROC": "AUROC",
        "AUPRC": "AUPRC"
    }
    short_metrics = [label_map[m] for m in metrics]


    # Two consistent bar styles
    facecolors = ["#ff7f0e", "#1f77b4"]
    hatches    = ["", "///"]

    fig, axes = plt.subplots(2, 2, figsize=(6.7, 6.0), dpi=300)
    axes = axes.flatten()

    for ax, drug in zip(axes, antibiotics):

        # Averaged metrics
        avg_metrics = avg_data[drug]
        avg_means = [avg_metrics[m]["mean"] for m in metrics]
        avg_stds  = [avg_metrics[m]["std"] for m in metrics]

        model_matrix = [avg_means]
        model_errors = [avg_stds]

        for log in eval_logs:
            entry = [d for d in log["metrics_per_label"] if d["label"].lower() == drug.lower()]
            if entry:
                d = entry[0]
                values = [d["auroc"], d["auprc"], d["balanced_accuracy"], d["weighted_f1"]]
            else:
                values = [np.nan] * 4

            model_matrix.append(values)
            model_errors.append([0.0] * 4)

        model_matrix = np.array(model_matrix)
        model_errors = np.array(model_errors)

        x = np.arange(len(metrics))
        width = 0.8 / len(model_names)

        for i in range(len(model_names)):
            ax.bar(
            x + i * width - (len(model_names)-1)/2 * width,
            model_matrix[i],
            width=width,
            hatch=hatches[i % 2],
            edgecolor="black",
            facecolor=facecolors[i % 2],
            linewidth=0.8,
            yerr=model_errors[i] if i == 0 else None,
            capsize=2
        )

        ax.set_title(f"E. coli – {drug}", fontsize=8, weight = "bold")
        ax.set_xticks(x)
        ax.set_xticklabels(short_metrics, rotation=25, fontsize=7)
        ax.set_ylim(0, 1.05)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_ylabel("Score", fontsize=7)

    plt.tight_layout()
    fname = os.path.join(output_dir, f"Figure_Paper_Ecoli_Set{set}_4panel.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"figure saved: {fname}")




def plot_uncertainty_distributions(log_file, specie, dataset, output_dir="uncertainty_plots"):
    """
    Plot Gaussian distributions of TP, TN, FP, FN uncertainties per label
    using B/W-friendly line styles. Legends removed as per journal instructions.
    """


    # Load JSON log
    with open(log_file, "r") as f:
        data = json.load(f)

    metrics = data["metrics_per_label"]
    os.makedirs(output_dir, exist_ok=True)

    # B/W-friendly line styles
    line_styles = {"TP": "solid", "TN": "dashed", "FP": "dotted", "FN": "dashdot"}
    colors = {
    "TN": "#1f77b4",   # blue
    "TP": "#2ca02c",   # green
    "FN": "#ff7f0e",   # orange
    "FP": "#d62728",   # red
    }

    for entry in metrics:
        label = entry["label"]

        groups = {
            "TP": (entry["tp_mean_uncertainty"], entry["tp_std_uncertainty"]),
            "TN": (entry["tn_mean_uncertainty"], entry["tn_std_uncertainty"]),
            "FP": (entry["fp_mean_uncertainty"], entry["fp_std_uncertainty"]),
            "FN": (entry["fn_mean_uncertainty"], entry["fn_std_uncertainty"]),
        }

        valid_means = [v[0] for v in groups.values() if not np.isnan(v[0])]
        valid_stds = [v[1] for v in groups.values() if not np.isnan(v[1])]
        if not valid_means or not valid_stds:
            print(f"Skipping {label}: no valid uncertainty values")
            continue

        xmin = min(valid_means) - 3 * max(valid_stds)
        xmax = max(valid_means) + 3 * max(valid_stds)
        x = np.linspace(xmin, xmax, 500)

        fig, ax = plt.subplots(figsize=(3.35, 2.5), dpi=300)
        for group, (mean, std) in groups.items():
            if std == 0 or np.isnan(mean) or np.isnan(std):
                continue
            y = norm.pdf(x, mean, std)
            ax.plot(x, y, linestyle=line_styles[group], linewidth=1.5, color=colors[group])

        ax.set_title(f"{specie} – {label}", fontsize=10, weight="bold")
        ax.set_xlabel("Uncertainty", fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        fname = os.path.join(output_dir, f"{specie}_{dataset}_{label.replace(' ', '_')}_uncertainty_BW.png")
        plt.savefig(fname, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved {fname}")






def plot_uncertainty_four_panel(log_file, specie, dataset, labels_to_plot, output_dir="uncertainty_plots"):
    """
    Create a 4-panel figure of uncertainty distributions (TP, TN, FP, FN)
    using B/W-friendly line styles. Legends removed.
    """

    # Load JSON log
    with open(log_file, "r") as f:
        data = json.load(f)

    metrics = data["metrics_per_label"]


    # Ensure output directory exists (only once)
    os.makedirs(output_dir, exist_ok=True)

    line_styles = {"TP": "solid", "TN": "dashed", "FP": "dotted", "FN": "dashdot"}
    colors = {
    "TN": "#1f77b4",   # blue
    "TP": "#2ca02c",   # green
    "FN": "#ff7f0e",   # orange
    "FP": "#d62728",   # red
    }

    fig, axes = plt.subplots(2, 2, figsize=(6.7, 6.0), dpi=300)
    axes = axes.flatten()

    for ax, label in zip(axes, labels_to_plot):
        entry = next((m for m in metrics if m["label"] == label), None)
        if entry is None:
            continue

        groups = {
            "TP": (entry["tp_mean_uncertainty"], entry["tp_std_uncertainty"]),
            "TN": (entry["tn_mean_uncertainty"], entry["tn_std_uncertainty"]),
            "FP": (entry["fp_mean_uncertainty"], entry["fp_std_uncertainty"]),
            "FN": (entry["fn_mean_uncertainty"], entry["fn_std_uncertainty"]),
        }

        valid_means = [v[0] for v in groups.values() if not np.isnan(v[0])]
        valid_stds = [v[1] for v in groups.values() if not np.isnan(v[1])]
        if not valid_means or not valid_stds:
            continue

        xmin = min(valid_means) - 3 * max(valid_stds)
        xmax = max(valid_means) + 3 * max(valid_stds)
        x = np.linspace(xmin, xmax, 500)

        for group, (mean, std) in groups.items():
            if std == 0 or np.isnan(mean) or np.isnan(std):
                continue
            y = norm.pdf(x, mean, std)
            ax.plot(x, y, linestyle=line_styles[group], linewidth=1.5, color=colors[group])

        ax.set_title(f"{specie} – {label}", fontsize=8, weight="bold")
        ax.set_xlabel("Uncertainty", fontsize=7)
        ax.set_ylabel("Density", fontsize=7)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    plt.tight_layout()
    fname = os.path.join(output_dir, f"{specie}_{dataset}_4panel_uncertainty_BW.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Figure saved: {fname}")



antibiotics = [
        "Ciprofloxacin",
        "Ceftriaxone",
        "Piperacillin-Tazobactam",
        "Tobramycin"
    ]



# set A

avg_file = "Results/SetA_results/ec_lasttwolayer_bayesian/combined_metrics.json"
eval_log_files = ["Results/SetA_results/ec_detlogs/ecoli_2015_2018_detmodel_seta_logs.json"]
plot_comparative_histograms(avg_file, eval_log_files, ["Last two Layer Bayesian", "Deterministic"], specie="E. coli", dataset="Set A", output_dir="Results/SetA_results/figures")
plot_paper_figure_four_panel(avg_file, eval_log_files, ["Last two Layer Bayesian", "Deterministic"], antibiotics, set = "A", output_dir="Results/SetA_results/figures")

avg_file = "Results/SetA_results/kp_lasttwolayer_bayesian/combined_metrics.json"
eval_log_files = ["Results/SetA_results/kp_detlogs/kp_2015_2018_detmodel_seta_logs.json"]
plot_comparative_histograms(avg_file, eval_log_files, ["Last two Layer Bayesian", "Deterministic"], specie="K. pneumoniae", dataset="Set A", output_dir="Results/SetA_results/figures")

avg_file = "Results/SetA_results/sa_lasttwolayer_bayesian/combined_metrics.json"
eval_log_files = ["Results/SetA_results/sa_detlogs/sa_2015_2018_detmodel_seta_logs.json"]
plot_comparative_histograms(avg_file, eval_log_files, ["Last two Layer Bayesian", "Deterministic"], specie="S. aureus", dataset="Set A", output_dir="Results/SetA_results/figures")

plot_uncertainty_distributions("Results/SetA_results/ec_lasttwolayer_bayesian/ecoli_2015_2018_lasttwolayerbayes_seta_run5.json", specie="E. coli", dataset="Set A", output_dir="Results/SetA_results/uncertainty_plots")
plot_uncertainty_four_panel("Results/SetA_results/ec_lasttwolayer_bayesian/ecoli_2015_2018_lasttwolayerbayes_seta_run5.json", specie="E. coli", dataset="Set A", labels_to_plot=antibiotics, output_dir="Results/SetA_results/uncertainty_plots")
plot_uncertainty_distributions("Results/SetA_results/Kp_lasttwolayer_bayesian/kp_2015_2018_lasttwolayerbayes_seta_run5.json", specie="K. pneumoniae", dataset="Set A", output_dir="Results/SetA_results/uncertainty_plots")
plot_uncertainty_distributions("Results/SetA_results/sa_lasttwolayer_bayesian/sa_2015_2018_lasttwolayerbayes_seta_run5.json", specie="S. aureus", dataset="Set A", output_dir="Results/SetA_results/uncertainty_plots")


# set B


antibiotics = [
        "Ciprofloxacin",
        "Ceftriaxone",
        "Piperacillin-Tazobactam",
        "Cefepime"
    ]

avg_file = "Results/SetB_results/ec_lasttwolayer_bayesian/combined_metrics.json"
eval_log_files = ["Results/setB_results/ec_detlogs/ecoli_2015_2018_detmodel_zeroshot_setB.json"]
plot_comparative_histograms(avg_file, eval_log_files, ["Last two Layer Bayesian", "Deterministic"], specie="E. coli", dataset="Set B", output_dir="Results/SetB_results/figures")
plot_paper_figure_four_panel(avg_file, eval_log_files, ["Last two Layer Bayesian", "Deterministic"], antibiotics, set = "B", output_dir="Results/SetB_results/figures")


avg_file = "Results/SetB_results/kp_lasttwolayer_bayesian/combined_metrics.json"
eval_log_files = ["Results/setB_results/kp_detlogs/kp_2015_2018_detmodel_zeroshot_setB.json"]
plot_comparative_histograms(avg_file, eval_log_files, ["Last two Layer Bayesian", "Deterministic"], specie="K. pneumoniae", dataset="Set B", output_dir="Results/SetB_results/figures")

avg_file = "Results/SetB_results/sa_lasttwolayer_bayesian/combined_metrics.json"
eval_log_files = ["Results/setB_results/sa_detlogs/sa_2015_2018_detmodel_zeroshot_setB.json"]
plot_comparative_histograms(avg_file, eval_log_files, ["Last two Layer Bayesian", "Deterministic"], specie="S. aureus", dataset="Set B", output_dir="Results/SetB_results/figures")

plot_uncertainty_distributions("Results/setB_results/ec_lasttwolayer_bayesian\ecoli_2015_2018_lasttwolayerbayesmodel_zeroshot_setB_run5.json", specie="E. coli", dataset="Set B", output_dir="Results/SetB_results/uncertainty_plots")
plot_uncertainty_four_panel("Results/setB_results/ec_lasttwolayer_bayesian\ecoli_2015_2018_lasttwolayerbayesmodel_zeroshot_setB_run5.json", specie="E. coli", dataset="Set B", labels_to_plot = antibiotics, output_dir="Results/SetB_results/uncertainty_plots")
plot_uncertainty_distributions("Results/setB_results/kp_lasttwolayer_bayesian/kp_2015_2018_lasttwolayerbayesmodel_zeroshot_setB_run5.json", specie="K. pneumoniae", dataset="Set B", output_dir="Results/SetB_results/uncertainty_plots")
plot_uncertainty_distributions("Results/setB_results/sa_lasttwolayer_bayesian/sa_2015_2018_lasttwolayerbayesmodel_zeroshot_setB_run5.json", specie="S. aureus", dataset="Set B", output_dir="Results/SetB_results/uncertainty_plots")


# set C
avg_file = "Results/SetC_results/ec_lasttwolayer_bayesian/combined_metrics.json"
eval_log_files = ["Results/setC_results/ec_detlogs/ecoli_2015_2018_detmodel_finetuned_setc.json"]
plot_comparative_histograms(avg_file, eval_log_files, ["Last two Layer Bayesian", "Deterministic"], specie="E. coli", dataset="Set C", output_dir="Results/SetC_results/figures")

avg_file = "Results/SetC_results/sa_lasttwolayer_bayesian/combined_metrics.json"
eval_log_files = ["Results/setC_results/sa_detlogs/sa_2015_2018_detmodel_finetuned_setc.json"]
plot_comparative_histograms(avg_file, eval_log_files, ["Last two Layer Bayesian", "Deterministic"], specie="S. aureus", dataset="Set C", output_dir="Results/SetC_results/figures")

plot_uncertainty_distributions("Results/setC_results/ec_lasttwolayer_bayesian/ecoli_2015_2018_lasttwolayerbayes_setc_finetuned_run5.json", specie="E. coli", dataset="Set C", output_dir="Results/SetC_results/uncertainty_plots")
plot_uncertainty_distributions("Results/setC_results/sa_lasttwolayer_bayesian/sa_2015_2018_lasttwolayerbayes_setc_finetuned_run5.json", specie="S. aureus", dataset="Set C", output_dir="Results/SetC_results/uncertainty_plots")


# set D
avg_file = "Results/SetD_results/ec_lasttwolayer_bayesian/combined_metrics.json"
eval_log_files = ["Results/setD_results/ec_detlogs/ecoli_2015_2018_detmodel_finetuned_setd.json"]
plot_comparative_histograms(avg_file, eval_log_files, ["Last two Layer Bayesian", "Deterministic"], specie="E. coli", dataset="Set D", output_dir="Results/SetD_results/figures")

avg_file = "Results/SetD_results/kp_lasttwolayer_bayesian/combined_metrics.json"
eval_log_files = ["Results/setD_results/kp_detlogs/kp_2015_2018_detmodel_finetuned_setD.json"]
plot_comparative_histograms(avg_file, eval_log_files, ["Last two Layer Bayesian", "Deterministic"], specie="K. pneumoniae", dataset="Set D", output_dir="Results/SetD_results/figures")

avg_file = "Results/SetD_results/sa_lasttwolayer_bayesian/combined_metrics.json"
eval_log_files = ["Results/setD_results/sa_detlogs/sa_2015_2018_detmodel_zeroshot_setD.json"]
plot_comparative_histograms(avg_file, eval_log_files, ["Last two Layer Bayesian", "Deterministic"], specie="S. aureus", dataset="Set D", output_dir="Results/SetD_results/figures")

plot_uncertainty_distributions("Results/setD_results/ec_lasttwolayer_bayesian/ecoli_2015_2018_lasttwolayerbayes_setd_finetuned_run5.json", specie="E. coli", dataset="Set D", output_dir="Results/SetD_results/uncertainty_plots")
plot_uncertainty_distributions("Results/setD_results/kp_lasttwolayer_bayesian/kp_2015_2018_lasttwolayerbayes_setD_finetuned_run5.json", specie="K. pneumoniae", dataset="Set D", output_dir="Results/SetD_results/uncertainty_plots")
plot_uncertainty_distributions("Results/setD_results/sa_lasttwolayer_bayesian/sa_2015_2018_lasttwolayerbayesmodel_zeroshot_setD_run5.json", specie="S. aureus", dataset="Set D", output_dir="Results/SetD_results/uncertainty_plots")


# UMG
avg_file = "Results/UMG_results/ec_lasttwolayer_bayesian/combined_metrics.json"
eval_log_files = ["Results/UMG_results/ec_detlogs/ecoli_2015_2018_detmodel_finetuned_setumg.json"]
plot_comparative_histograms(avg_file, eval_log_files, ["Last two Layer Bayesian", "Deterministic"], specie="E. coli", dataset="UMG", output_dir="Results/UMG_results/figures")

avg_file = "Results/UMG_results/kp_lasttwolayer_bayesian/combined_metrics.json"
eval_log_files = ["Results/UMG_results/kp_detlogs/kp_2015_2018_detmodel_finetuned_setumg.json"]
plot_comparative_histograms(avg_file, eval_log_files, ["Last two Layer Bayesian", "Deterministic"], specie="K. pneumoniae", dataset="UMG", output_dir="Results/UMG_results/figures")

avg_file = "Results/UMG_results/sa_lasttwolayer_bayesian/combined_metrics.json"
eval_log_files = ["Results/UMG_results/sa_detlogs/sa_2015_2018_detmodel_zeroshot_setUMG.json"]
plot_comparative_histograms(avg_file, eval_log_files, ["Last two Layer Bayesian", "Deterministic"], specie="S. aureus", dataset="UMG", output_dir="Results/UMG_results/figures")

plot_uncertainty_distributions("Results/UMG_results/ec_lasttwolayer_bayesian/ecoli_2015_2018_lasttwolayerbayes_setumg_finetuned_run5.json", specie="E. coli", dataset="UMG", output_dir="Results/UMG_results/uncertainty_plots")
plot_uncertainty_distributions("Results/UMG_results/kp_lasttwolayer_bayesian/kp_2015_2018_lasttwolayerbayes_setumg_finetuned_run5.json", specie="K. pneumoniae", dataset="UMG", output_dir="Results/UMG_results/uncertainty_plots")
plot_uncertainty_distributions("Results/UMG_results/sa_lasttwolayer_bayesian/sa_2015_2018_lasttwolayerbayesmodel_zeroshot_setUMG_run5.json", specie="S. aureus", dataset="UMG", output_dir="Results/UMG_results/uncertainty_plots")
