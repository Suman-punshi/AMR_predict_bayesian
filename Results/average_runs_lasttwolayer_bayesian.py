import json
import numpy as np
import os

def aggregate_eval_logs(input_dir, output_dir, output_file="combined_metrics.json"):
    """
    Aggregate multiple evaluation JSON logs from a directory and compute mean and std 
    for each label and overall metrics.

    Parameters:
    - input_dir (str): Directory containing evaluation JSON files (one per run).
    - output_dir (str): Directory where aggregated results JSON will be saved.
    - output_file (str): Filename for aggregated results JSON (default: "combined_metrics.json").
    """
    # Find all JSON files in the directory
    json_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".json")]
    if not json_files:
        raise ValueError(f"No JSON files found in {input_dir}")

    # Containers
    per_label_metrics = {}
    mean_metrics = { "auroc": [], "auprc": [], "balanced_accuracy": [], "weighted_f1": [], "hamming_loss": [] }

    for file in json_files:
        with open(file, "r") as f:
            data = json.load(f)

        # Collect per-label metrics
        for entry in data["metrics_per_label"]:
            label = entry["label"]
            if label not in per_label_metrics:
                per_label_metrics[label] = { "auroc": [], "auprc": [], "balanced_accuracy": [], "weighted_f1": [] }
            
            per_label_metrics[label]["auroc"].append(entry["auroc"])
            per_label_metrics[label]["auprc"].append(entry["auprc"])
            per_label_metrics[label]["balanced_accuracy"].append(entry["balanced_accuracy"])
            per_label_metrics[label]["weighted_f1"].append(entry["weighted_f1"])

        # Collect mean metrics
        for k in mean_metrics.keys():
            if k in data["mean_metrics"]:
                mean_metrics[k].append(data["mean_metrics"][k])

    # Compute averages and stds
    results = {}
    for label, metrics in per_label_metrics.items():
        results[label] = {}
        for m, values in metrics.items():
            results[label][m.upper()] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            }

    # Overall mean metrics
    results["mean_metrics"] = {}
    for m, values in mean_metrics.items():
        results["mean_metrics"][m.upper()] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        }

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_file)

    # Save aggregated results
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Saved aggregated results to {output_path}")
    return results




# average ecoli set a
aggregate_eval_logs("Results/SetA_results/ec_lasttwolayer_bayesian", "Results/SetA_results/ec_lasttwolayer_bayesian", output_file="combined_metrics.json")

# average ecoli set b
aggregate_eval_logs("Results/SetB_results/ec_lasttwolayer_bayesian", "Results/SetB_results/ec_lasttwolayer_bayesian", output_file="combined_metrics.json")

# average ecoli set c
aggregate_eval_logs("Results/SetC_results/ec_lasttwolayer_bayesian", "Results/SetC_results/ec_lasttwolayer_bayesian", output_file="combined_metrics.json")

# average ecoli set D
aggregate_eval_logs("Results/SetD_results/ec_lasttwolayer_bayesian", "Results/SetD_results/ec_lasttwolayer_bayesian", output_file="combined_metrics.json")

# average ecoli set D
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

# # average sa set b
# aggregate_eval_logs("Results/SetB_results/sa_lasttwolayer_bayesian", "Results/SetB_results/sa_lasttwolayer_bayesian", output_file="combined_metrics.json")

# # average sa set c
# aggregate_eval_logs("Results/SetC_results/sa_lasttwolayer_bayesian", "Results/SetC_results/sa_lasttwolayer_bayesian", output_file="combined_metrics.json")

# # average sa set D
# aggregate_eval_logs("Results/SetD_results/sa_lasttwolayer_bayesian", "Results/SetD_results/sa_lasttwolayer_bayesian", output_file="combined_metrics.json")

# # average sa set D
# aggregate_eval_logs("Results/UMG_results/sa_lasttwolayer_bayesian", "Results/UMG_results/sa_lasttwolayer_bayesian", output_file="combined_metrics.json")