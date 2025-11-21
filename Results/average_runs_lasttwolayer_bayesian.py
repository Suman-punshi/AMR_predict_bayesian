import json
import numpy as np
import os
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    balanced_accuracy_score, f1_score, hamming_loss
)



def aggregate_eval_logs(input_dir, output_dir, output_file="combined_metrics.json"):
    """
    Recompute per-label metrics (AUROC, AUPRC, balanced accuracy, weighted F1)
    from saved per-sample logs in each run, then compute mean & std across runs.
    """

    json_files = [os.path.join(input_dir, f)
                  for f in os.listdir(input_dir)
                  if f.endswith(".json")]

    if not json_files:
        raise ValueError(f"No JSON files found in {input_dir}")

    # Will store list of: metrics_per_run[run_idx][label_name] = {metric: value}
    per_run_metrics = []

    for file in json_files:
        with open(file, "r") as f:
            data = json.load(f)

        label_names = data["label_names"]
        L = len(label_names)

        # Reconstruct y_true and y_probs from per-sample logs
        y_true = []
        y_probs = []

        for sample in data["per_sample"]:
            y_true.append(sample["true"])
            y_probs.append(sample["probs"])

        y_true = np.array(y_true)
        y_probs = np.array(y_probs)
        y_pred = (y_probs > 0.5).astype(int)

        # Compute metrics per label
        run_metrics = {}

        for i, label in enumerate(label_names):
            try:
                auc_roc = roc_auc_score(y_true[:, i], y_probs[:, i])
            except:
                auc_roc = float("nan")
            try:
                auc_prc = average_precision_score(y_true[:, i], y_probs[:, i])
            except:
                auc_prc = float("nan")

            bal_acc = balanced_accuracy_score(y_true[:, i], y_pred[:, i])
            f1 = f1_score(y_true[:, i], y_pred[:, i], average='weighted')

            run_metrics[label] = {
                "auroc": auc_roc,
                "auprc": auc_prc,
                "balanced_accuracy": bal_acc,
                "weighted_f1": f1,
            }

        per_run_metrics.append(run_metrics)

    # === Compute mean & std across runs ===
    output = {}

    for label in label_names:
        # Collect values across runs
        auc_roc_vals = [run[label]["auroc"] for run in per_run_metrics]
        auc_prc_vals = [run[label]["auprc"] for run in per_run_metrics]
        bal_vals     = [run[label]["balanced_accuracy"] for run in per_run_metrics]
        f1_vals      = [run[label]["weighted_f1"] for run in per_run_metrics]

        output[label] = {
            "AUROC": {
                "mean": float(np.nanmean(auc_roc_vals)),
                "std": float(np.nanstd(auc_roc_vals, ddof=1))
            },
            "AUPRC": {
                "mean": float(np.nanmean(auc_prc_vals)),
                "std": float(np.nanstd(auc_prc_vals, ddof=1))
            },
            "BALANCED_ACCURACY": {
                "mean": float(np.mean(bal_vals)),
                "std": float(np.std(bal_vals, ddof=1))
            },
            "WEIGHTED_F1": {
                "mean": float(np.mean(f1_vals)),
                "std": float(np.std(f1_vals, ddof=1))
            }
        }

    # Save final aggregated metrics
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, output_file)

    with open(out_path, "w") as f:
        json.dump(output, f, indent=4)

    print(f"\nSaved corrected aggregated results to: {out_path}")
    return output



# average ecoli set a
aggregate_eval_logs("Results/SetA_results/ec_lasttwolayer_bayesian", "Results/SetA_results/ec_lasttwolayer_bayesian", output_file="combined_metrics.json")

# average ecoli set b
aggregate_eval_logs("Results/SetB_results/ec_lasttwolayer_bayesian", "Results/SetB_results/ec_lasttwolayer_bayesian", output_file="combined_metrics.json")

# average ecoli set c
aggregate_eval_logs("Results/SetC_results/ec_lasttwolayer_bayesian", "Results/SetC_results/ec_lasttwolayer_bayesian", output_file="combined_metrics.json")

# average ecoli set D
aggregate_eval_logs("Results/SetD_results/ec_lasttwolayer_bayesian", "Results/SetD_results/ec_lasttwolayer_bayesian", output_file="combined_metrics.json")

# average ecoli set UMG
aggregate_eval_logs("Results/UMG_results/ec_lasttwolayer_bayesian", "Results/UMG_results/ec_lasttwolayer_bayesian", output_file="combined_metrics.json")




# average kp set a
aggregate_eval_logs("Results/SetA_results/kp_lasttwolayer_bayesian", "Results/SetA_results/kp_lasttwolayer_bayesian", output_file="combined_metrics.json")

# average kp set b
aggregate_eval_logs("Results/SetB_results/kp_lasttwolayer_bayesian", "Results/SetB_results/kp_lasttwolayer_bayesian", output_file="combined_metrics.json")

# average kp set D
aggregate_eval_logs("Results/SetD_results/kp_lasttwolayer_bayesian", "Results/SetD_results/kp_lasttwolayer_bayesian", output_file="combined_metrics.json")

# average kp set UMG
aggregate_eval_logs("Results/UMG_results/kp_lasttwolayer_bayesian", "Results/UMG_results/kp_lasttwolayer_bayesian", output_file="combined_metrics.json")



# average sa set a
aggregate_eval_logs("Results/SetA_results/sa_lasttwolayer_bayesian", "Results/SetA_results/sa_lasttwolayer_bayesian", output_file="combined_metrics.json")

# average sa set b
aggregate_eval_logs("Results/SetB_results/sa_lasttwolayer_bayesian", "Results/SetB_results/sa_lasttwolayer_bayesian", output_file="combined_metrics.json")

# average sa set c
aggregate_eval_logs("Results/SetC_results/sa_lasttwolayer_bayesian", "Results/SetC_results/sa_lasttwolayer_bayesian", output_file="combined_metrics.json")

# average sa set D
aggregate_eval_logs("Results/SetD_results/sa_lasttwolayer_bayesian", "Results/SetD_results/sa_lasttwolayer_bayesian", output_file="combined_metrics.json")

# average sa set D
aggregate_eval_logs("Results/UMG_results/sa_lasttwolayer_bayesian", "Results/UMG_results/sa_lasttwolayer_bayesian", output_file="combined_metrics.json")