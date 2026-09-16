import json
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

def calculate_brier_ece(
    json_file,
    species,
    set_name,
    model_type,
    n_bins=10,
    output_dir="Results"
):
    """
    Calculate per-label Brier score and ECE from either deterministic
    or Bayesian evaluation JSON files.

    Saves results as a CSV organized by species/set/model, e.g.:
        results/{species}/{species}_{set_name}_{model_type}_calibration.csv

    Parameters
    ----------
    json_file : str
        Path to the input JSON file.
    species : str
        e.g. "ecoli", "saureus"
    set_name : str
        e.g. "setA", "setB", "setC", "setD", "UMG"
    model_type : str
        e.g. "deterministic", "bayesian"
    n_bins : int
        Number of bins for ECE calculation.
    output_dir : str
        Base directory to save results into.
    """

    with open(json_file, "r") as f:
        data = json.load(f)

    label_names = data["label_names"]

    # Handle deterministic JSON
    if "y_true" in data and "y_probs" in data:

        y_true = np.array(data["y_true"])
        y_probs = np.array(data["y_probs"])

    # Handle Bayesian JSON
    elif "per_sample" in data:

        y_true = np.array([
            sample["true"] for sample in data["per_sample"]
        ])

        y_probs = np.array([
            sample["probs"] for sample in data["per_sample"]
        ])

    else:
        raise ValueError(
            "JSON format not recognized. Expected either "
            "'y_true'/'y_probs' or 'per_sample'."
        )

    # Calculate Brier + ECE per label
    results = []

    for i, label in enumerate(label_names):

        true = y_true[:, i]
        probs = y_probs[:, i]

        # Brier score
        brier = brier_score_loss(true, probs)

        # ECE
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

        ece = 0.0

        for b in range(n_bins):

            if b == n_bins - 1:
                mask = (
                    (probs >= bin_edges[b]) &
                    (probs <= bin_edges[b + 1])
                )
            else:
                mask = (
                    (probs >= bin_edges[b]) &
                    (probs < bin_edges[b + 1])
                )

            if not np.any(mask):
                continue

            mean_prob = np.mean(probs[mask])
            empirical_freq = np.mean(true[mask])
            bin_fraction = np.mean(mask)

            ece += bin_fraction * abs(
                empirical_freq - mean_prob
            )

        results.append({
            "Species": species,
            "Set": set_name,
            "Model": model_type,
            "Label": label,
            "Brier Score": brier,
            "ECE": ece
        })

    results_df = pd.DataFrame(results)

    print(f"\nPer-Label Calibration Metrics — {species} | {set_name} | {model_type}\n")
    print(results_df.to_string(index=False))

    # Build organized output path: results/{species}/{species}_{set}_{model}_calibration.csv
    set_dir = os.path.join(output_dir, set_name)
    os.makedirs(set_dir, exist_ok=True)

    filename = f"{species}_{set_name}_{model_type}_calibration.csv"
    filepath = os.path.join(set_dir, filename)

    results_df.to_csv(filepath, index=False)
    print(f"\nSaved to: {filepath}")

    return results_df


print("\nCalculating Brier Score and ECE for SetA E. coli Results for deterministic evaluation\n")
calculate_brier_ece("Results/SetA_results/ec_detlogs/ecoli_2015_2018_detmodel_seta_logs.json", "ecoli", "SetA_results", "Deterministic", n_bins=10)
print("\nCalculating Brier Score and ECE for SetA E. coli Results for bayesian evaluation\n")
calculate_brier_ece("Results/SetA_results/ec_lasttwolayer_bayesian/ecoli_2015_2018_lasttwolayerbayes_seta_run1.json", "ecoli", "SetA_results", "Bayesian", n_bins=10)

print("\nCalculating Brier Score and ECE for SetA S. aureus Results for deterministic evaluation\n")
calculate_brier_ece("Results\SetA_results\sa_detlogs\sa_2015_2018_detmodel_seta_logs.json", "sa", "SetA_results", "Deterministic", n_bins=10)
print("\nCalculating Brier Score and ECE for SetA S. aureus Results for bayesian evaluation\n")
calculate_brier_ece("Results\SetA_results\sa_lasttwolayer_bayesian\sa_2015_2018_lasttwolayerbayes_seta_run1.json", "sa", "SetA_results", "Bayesian", n_bins=10)



print("\nCalculating Brier Score and ECE for SetB E. coli Results for deterministic evaluation\n")
calculate_brier_ece("Results/SetB_results/ec_detlogs/ecoli_2015_2018_detmodel_zeroshot_setB.json", "ecoli", "setB_results", "Deterministic", n_bins=10)
print("\nCalculating Brier Score and ECE for SetB Results for bayesian evaluation\n")
calculate_brier_ece("Results/setB_results/ec_lasttwolayer_bayesian/ecoli_2015_2018_lasttwolayerbayesmodel_zeroshot_setB_run1.json", "ecoli", "setB_results", "Bayesian", n_bins=10)

print("\nCalculating Brier Score and ECE for SetB S. aureus Results for deterministic evaluation\n")
calculate_brier_ece("Results\setB_results\sa_detlogs\sa_2015_2018_detmodel_zeroshot_setB.json", "sa", "setB_results", "Deterministic", n_bins=10)
print("\nCalculating Brier Score and ECE for SetB S. aureus Results for bayesian evaluation\n")
calculate_brier_ece("Results\setB_results\sa_lasttwolayer_bayesian\sa_2015_2018_lasttwolayerbayesmodel_zeroshot_setB_run1.json", "sa", "setB_results", "Bayesian", n_bins=10)



print("\nCalculating Brier Score and ECE for SetC E. coli Results for deterministic evaluation\n")
calculate_brier_ece("Results/SetC_results/ec_detlogs/ecoli_2015_2018_detmodel_finetuned_setc.json", "ecoli", "setC_results", "Deterministic", n_bins=10)
print("\nCalculating Brier Score and ECE for SetC E. coli Results for bayesian evaluation\n")
calculate_brier_ece("Results/setC_results/ec_lasttwolayer_bayesian/ecoli_2015_2018_lasttwolayerbayes_setc_finetuned_run1.json", "ecoli", "setC_results", "Bayesian", n_bins=10)

print("\nCalculating Brier Score and ECE for SetC S. aureus Results for deterministic evaluation\n")
calculate_brier_ece("Results/SetC_results/sa_detlogs/sa_2015_2018_detmodel_finetuned_setc.json", "sa", "setC_results", "Deterministic", n_bins=10)
print("\nCalculating Brier Score and ECE for SetC S. aureus Results for bayesian evaluation\n")
calculate_brier_ece("Results/setC_results/sa_lasttwolayer_bayesian/sa_2015_2018_lasttwolayerbayes_setc_finetuned_run1.json", "sa", "setC_results", "Bayesian", n_bins=10)


print("\nCalculating Brier Score and ECE for SetD E. coli Results for deterministic evaluation\n")
calculate_brier_ece("Results/setD_results/ec_detlogs/ecoli_2015_2018_detmodel_finetuned_setd.json", "ecoli", "setD_results", "Deterministic", n_bins=10)
print("\nCalculating Brier Score and ECE for SetD E. coli Results for bayesian evaluation\n")
calculate_brier_ece("Results/setD_results/ec_lasttwolayer_bayesian/ecoli_2015_2018_lasttwolayerbayes_setd_finetuned_run1.json", "ecoli", "setD_results", "Bayesian", n_bins=10)

print("\nCalculating Brier Score and ECE for SetD S. aureus Results for deterministic evaluation\n")
calculate_brier_ece("Results/setD_results/sa_detlogs/sa_2015_2018_detmodel_zeroshot_setD.json", "sa", "setD_results", "Deterministic", n_bins=10)
print("\nCalculating Brier Score and ECE for SetD S. aureus Results for bayesian evaluation\n")
calculate_brier_ece("Results/setD_results/sa_lasttwolayer_bayesian/sa_2015_2018_lasttwolayerbayesmodel_zeroshot_setD_run1.json", "sa", "setD_results", "Bayesian", n_bins=10)


print("\nCalculating Brier Score and ECE for UMG E. coli Results for deterministic evaluation\n")
calculate_brier_ece("Results/UMG_results/ec_detlogs/ecoli_2015_2018_detmodel_finetuned_setumg.json", "ecoli", "setUMG_results", "Deterministic", n_bins=10)
print("\nCalculating Brier Score and ECE for UMG E. coli Results for bayesian evaluation\n")
calculate_brier_ece("Results/UMG_results/ec_lasttwolayer_bayesian/ecoli_2015_2018_lasttwolayerbayes_setumg_finetuned_run1.json", "ecoli", "setUMG_results", "Bayesian", n_bins=10)


print("\nCalculating Brier Score and ECE for UMG S. aureus Results for deterministic evaluation\n")
calculate_brier_ece("Results/UMG_results/sa_detlogs/sa_2015_2018_detmodel_zeroshot_setUMG.json", "sa", "setUMG_results", "Deterministic", n_bins=10)
print("\nCalculating Brier Score and ECE for UMG S. aureus Results for bayesian evaluation\n")
calculate_brier_ece("Results/UMG_results/sa_lasttwolayer_bayesian/sa_2015_2018_lasttwolayerbayesmodel_zeroshot_setUMG_run1.json", "sa", "setUMG_results", "Bayesian", n_bins=10)
