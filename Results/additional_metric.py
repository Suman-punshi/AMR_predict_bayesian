import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    matthews_corrcoef, accuracy_score, precision_score, recall_score,
    balanced_accuracy_score, roc_auc_score, average_precision_score,
    f1_score, roc_curve, precision_recall_curve
)


def _load_eval_json(json_file):
    """
    Loads either a deterministic (evaluate_multilabel) or Bayesian
    (evaluate_multilabel_bayesiansetup) evaluation JSON and returns
    a common (y_true, y_probs, y_pred, label_names) representation.
    """
    with open(json_file, "r") as f:
        data = json.load(f)

    label_names = data["label_names"]

    # Deterministic format: y_true / y_probs / y_pred stored directly
    if "y_true" in data and "y_probs" in data:
        y_true = np.array(data["y_true"])
        y_probs = np.array(data["y_probs"])
        y_pred = np.array(data["y_pred"]) if "y_pred" in data else (y_probs > 0.5).astype(int)
        model_type = "deterministic"

    # Bayesian format: reconstruct from per_sample list
    elif "per_sample" in data:
        y_true = np.array([s["true"] for s in data["per_sample"]])
        y_probs = np.array([s["probs"] for s in data["per_sample"]])
        y_pred = np.array([s["pred"] for s in data["per_sample"]])
        model_type = "bayesian"

    else:
        raise ValueError(
            "JSON format not recognized. Expected either "
            "'y_true'/'y_probs'/'y_pred' or 'per_sample'."
        )

    return y_true, y_probs, y_pred, label_names, model_type


def evaluate_extended_metrics(
    json_file,
    species,
    set_name,
    model_type=None,
    output_dir="Results",
    plot_curves=True,
    figure_width_mm=170,
    figure_height_mm=150,
    dpi=300
):
    """
    Computes extended per-label classification metrics (MCC, accuracy, precision,
    recall, balanced accuracy, AUROC, AUPRC, F1) from either a deterministic or
    Bayesian evaluation JSON file, saves them as an organized CSV, and plots
    ROC and PRC curves per label.

    Args:
        json_file (str): Path to the evaluation JSON (det or bayesian format).
        species (str): e.g. "ecoli", "saureus"
        set_name (str): e.g. "SetA", "SetB", "UMG"
        model_type (str): Optional override for "deterministic"/"bayesian" naming
            in output files. If None, auto-detected from JSON structure.
        output_dir (str): Base directory to save results into.
        plot_curves (bool): Whether to generate and save ROC/PRC figures.
        figure_width_mm, figure_height_mm (float): Journal-compliant figure size.
        dpi (int): Resolution for saved figures.

    Returns:
        pd.DataFrame: Per-label metrics table.
    """

    y_true, y_probs, y_pred, label_names, detected_model_type = _load_eval_json(json_file)

    if model_type is None:
        model_type = detected_model_type

    num_labels = len(label_names)
    results = []

    for i in range(num_labels):
        true_i = y_true[:, i]
        probs_i = y_probs[:, i]
        pred_i = y_pred[:, i]

        results.append({
            "Antibiotic": label_names[i],
            "Accuracy": accuracy_score(true_i, pred_i),
            "Balanced_Accuracy": balanced_accuracy_score(true_i, pred_i),
            "Precision": precision_score(true_i, pred_i, zero_division=0),
            "Recall": recall_score(true_i, pred_i, zero_division=0),
            "F1_Weighted": f1_score(true_i, pred_i, average="weighted", zero_division=0),
            "MCC": matthews_corrcoef(true_i, pred_i),
            "AUROC": roc_auc_score(true_i, probs_i),
            "AUPRC": average_precision_score(true_i, probs_i)
        })

    results_df = pd.DataFrame(results)

    print(f"\nExtended Evaluation Metrics — {species} | {set_name} | {model_type}\n")
    print(results_df.to_string(index=False))

    # ---- Save CSV, organized by species/set/model ----
    species_dir = os.path.join(output_dir, species)
    os.makedirs(species_dir, exist_ok=True)

    csv_filename = f"{species}_{set_name}_{model_type}_extended_metrics.csv"
    csv_path = os.path.join(species_dir, csv_filename)
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved metrics to: {csv_path}")

    # ---- ROC and PRC curves ----
    if plot_curves:
        fig_width_in = figure_width_mm / 25.4
        fig_height_in = figure_height_mm / 25.4

        n_cols = min(num_labels, 3)
        n_rows = int(np.ceil(num_labels / n_cols))

        # ROC figure
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width_in, fig_height_in))
        axes = np.array(axes).flatten()

        for i in range(num_labels):
            fpr, tpr, _ = roc_curve(y_true[:, i], y_probs[:, i])
            auc_val = roc_auc_score(y_true[:, i], y_probs[:, i])

            ax = axes[i]
            ax.plot(fpr, tpr, color="darkorange", linewidth=1.2,
                    label=f"AUROC = {auc_val:.3f}")
            ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=0.8)
            ax.set_xlabel("False Positive Rate", fontsize=8)
            ax.set_ylabel("True Positive Rate", fontsize=8)
            ax.set_title(label_names[i], fontsize=9, fontweight="bold")
            ax.legend(fontsize=7, loc="lower right")
            ax.tick_params(axis='both', which='major', labelsize=7)
            ax.grid(True, linewidth=0.3)

        # Hide unused subplots
        for j in range(num_labels, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        roc_path = os.path.join(species_dir, f"{species}_{set_name}_{model_type}_ROC.png")
        plt.savefig(roc_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {roc_path}")

        # PRC figure
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width_in, fig_height_in))
        axes = np.array(axes).flatten()

        for i in range(num_labels):
            precision, recall, _ = precision_recall_curve(y_true[:, i], y_probs[:, i])
            ap_val = average_precision_score(y_true[:, i], y_probs[:, i])

            ax = axes[i]
            ax.plot(recall, precision, color="steelblue", linewidth=1.2,
                    label=f"AUPRC = {ap_val:.3f}")
            ax.set_xlabel("Recall", fontsize=8)
            ax.set_ylabel("Precision", fontsize=8)
            ax.set_title(label_names[i], fontsize=9, fontweight="bold")
            ax.legend(fontsize=7, loc="lower left")
            ax.tick_params(axis='both', which='major', labelsize=7)
            ax.grid(True, linewidth=0.3)

        for j in range(num_labels, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        prc_path = os.path.join(species_dir, f"{species}_{set_name}_{model_type}_PRC.png")
        plt.savefig(prc_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {prc_path}")

    return results_df




det_results = evaluate_extended_metrics(
    json_file="Results/SetA_results/ec_detlogs/ecoli_2015_2018_detmodel_seta_logs.json",
    species="ec",
    set_name="SetA",
    model_type="deterministic"

)
bayes_results = evaluate_extended_metrics(
    json_file="Results/SetA_results/ec_lasttwolayer_bayesian/ecoli_2015_2018_lasttwolayerbayes_seta_run1.json",
    species="ec",
    set_name="SetA",
    model_type="bayesian"
)


det_results = evaluate_extended_metrics(
    json_file="Results/SetA_results/sa_detlogs/sa_2015_2018_detmodel_seta_logs.json",
    species="sa",
    set_name="SetA",
    model_type="deterministic"

)
bayes_results = evaluate_extended_metrics(
    json_file="Results/SetA_results/sa_lasttwolayer_bayesian/sa_2015_2018_lasttwolayerbayes_seta_run1.json",
    species="sa",
    set_name="SetA",
    model_type="bayesian"
)



det_results = evaluate_extended_metrics(
    json_file="Results/setB_results/ec_detlogs/ecoli_2015_2018_detmodel_zeroshot_setB.json",
    species="ec",
    set_name="SetB",
    model_type="deterministic"

)
bayes_results = evaluate_extended_metrics(
    json_file="Results/setB_results/ec_lasttwolayer_bayesian/ecoli_2015_2018_lasttwolayerbayesmodel_zeroshot_setB_run1.json",
    species="ec",
    set_name="SetB",
    model_type="bayesian"
)




det_results = evaluate_extended_metrics(
    json_file="Results/setB_results/sa_detlogs/sa_2015_2018_detmodel_zeroshot_setB.json",
    species="sa",
    set_name="SetB",
    model_type="deterministic"

)
bayes_results = evaluate_extended_metrics(
    json_file="Results/setB_results/sa_lasttwolayer_bayesian/sa_2015_2018_lasttwolayerbayesmodel_zeroshot_setB_run1.json",
    species="sa",
    set_name="SetB",
    model_type="bayesian"
)



det_results = evaluate_extended_metrics(
    json_file="Results/setC_results/ec_detlogs/ecoli_2015_2018_detmodel_finetuned_setc.json",
    species="ec",
    set_name="SetC",
    model_type="deterministic"

)
bayes_results = evaluate_extended_metrics(
    json_file="Results/setC_results/ec_lasttwolayer_bayesian/ecoli_2015_2018_lasttwolayerbayes_setc_finetuned_run1.json",
    species="ec",
    set_name="SetC",
    model_type="bayesian"
)


det_results = evaluate_extended_metrics(
    json_file="Results/setC_results/sa_detlogs/sa_2015_2018_detmodel_finetuned_setc.json",
    species="sa",
    set_name="SetC",
    model_type="deterministic"

)
bayes_results = evaluate_extended_metrics(
    json_file="Results/setC_results/sa_lasttwolayer_bayesian/sa_2015_2018_lasttwolayerbayes_setc_finetuned_run1.json",
    species="sa",
    set_name="SetC",
    model_type="bayesian"
)



det_results = evaluate_extended_metrics(
    json_file="Results/setD_results/ec_detlogs/ecoli_2015_2018_detmodel_finetuned_setd.json",
    species="ec",
    set_name="SetD",
    model_type="deterministic"

)
bayes_results = evaluate_extended_metrics(
    json_file="Results/setD_results/ec_lasttwolayer_bayesian/ecoli_2015_2018_lasttwolayerbayes_setd_finetuned_run1.json",
    species="ec",
    set_name="SetD",
    model_type="bayesian"
)



det_results = evaluate_extended_metrics(
    json_file="Results/setD_results/sa_detlogs/sa_2015_2018_detmodel_zeroshot_setD.json",
    species="sa",
    set_name="SetD",
    model_type="deterministic"

)
bayes_results = evaluate_extended_metrics(
    json_file="Results/setD_results/sa_lasttwolayer_bayesian/sa_2015_2018_lasttwolayerbayesmodel_zeroshot_setD_run1.json",
    species="sa",
    set_name="SetD",
    model_type="bayesian"
)




det_results = evaluate_extended_metrics(
    json_file="Results/UMG_results/ec_detlogs/ecoli_2015_2018_detmodel_finetuned_setumg.json",
    species="ec",
    set_name="UMG",
    model_type="deterministic"

)
bayes_results = evaluate_extended_metrics(
    json_file="Results/UMG_results/ec_lasttwolayer_bayesian/ecoli_2015_2018_lasttwolayerbayes_setumg_finetuned_run1.json",
    species="ec",
    set_name="UMG",
    model_type="bayesian"
)



det_results = evaluate_extended_metrics(
    json_file="Results/UMG_results/sa_detlogs/sa_2015_2018_detmodel_zeroshot_setUMG.json",
    species="sa",
    set_name="UMG",
    model_type="deterministic"

)
bayes_results = evaluate_extended_metrics(
    json_file="Results/UMG_results/sa_lasttwolayer_bayesian/sa_2015_2018_lasttwolayerbayesmodel_zeroshot_setUMG_run1.json",
    species="sa",
    set_name="UMG",
    model_type="bayesian"
)