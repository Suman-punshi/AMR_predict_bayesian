import json
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score
)

# from statsmodels.stats.contingency_tables import mcnemar
# from statsmodels.stats.multitest import multipletests

import os


# load the deterministic JSON file and return label names, y_true, y_probs, and y_pred
def load_deterministic_json(path):
    with open(path, "r") as f:
        data = json.load(f)

    y_true = np.asarray(data["y_true"])
    y_probs = np.asarray(data["y_probs"])
    y_pred = np.asarray(data["y_pred"])

    return data["label_names"], y_true, y_probs, y_pred

# load the bayesian JSON file and return label names, y_true, y_probs, and y_pred
def load_bayesian_json(path):
    with open(path, "r") as f:
        data = json.load(f)

    per_sample = data["per_sample"]

    y_true = np.asarray([x["true"] for x in per_sample])
    y_probs = np.asarray([x["probs"] for x in per_sample])
    y_pred = np.asarray([x["pred"] for x in per_sample])

    return data["label_names"], y_true, y_probs, y_pred


# paired permutation test comparing two models evaluated on exactly the same samples
def paired_permutation_test(
    y_true,
    probs_a,
    probs_b,
    metric="auroc",
    n_permutations=10000,
    random_state=42
):
    """
    Paired permutation test comparing two models evaluated
    on exactly the same samples.
    probs_a and probs_b are the predicted probabilities for the positive class

    Returns:
        observed_diff
        p_value
    """

    rng = np.random.default_rng(random_state)
    # convert the labels and probabilities to numpy arrays
    y_true = np.asarray(y_true)
    probs_a = np.asarray(probs_a)
    probs_b = np.asarray(probs_b)
    # if the metric is not recognized, raise an error
    if metric == "auroc":
        metric_fn = roc_auc_score
    elif metric == "auprc":
        metric_fn = average_precision_score
    else:
        raise ValueError("metric must be 'auroc' or 'auprc'")
    # compute the observed difference in the metric between the two models
    observed_a = metric_fn(y_true, probs_a)
    observed_b = metric_fn(y_true, probs_b)
    observed_diff = observed_a - observed_b
    # initialize an array to store the null distribution of differences
    null_diffs = np.empty(n_permutations)
    # for each permutation, randomly swap the predictions of the two models for each sample
    for i in tqdm(range(n_permutations), desc="Running permutation tests"):

        # Randomly decide which model's prediction
        # each sample contributes.
        swap = rng.integers(0, 2, size=len(y_true)).astype(bool)
        # copy the probabilities and swap the predictions for the samples where swap is True
        perm_a = probs_a.copy()
        perm_b = probs_b.copy()
        perm_a[swap] = probs_b[swap]
        perm_b[swap] = probs_a[swap]
        # get the fake scores for the two models and compute the difference
        score_a = metric_fn(y_true, perm_a)
        score_b = metric_fn(y_true, perm_b)
        # get the difference and store it in the null distribution
        null_diffs[i] = score_a - score_b

    # Two-sided p-value
    p_value = (
        np.sum(np.abs(null_diffs) >= np.abs(observed_diff)) + 1
    ) / (n_permutations + 1)

    return observed_diff, p_value


def run_test_per_antibiotic(
    det_labels,
    det_true,
    det_probs,
    bayes_labels,
    bayes_true,
    bayes_probs,
    species,
    set_name,
    output_dir="Results"
):
    """
    Run paired permutation tests for AUROC and AUPRC
    for each antibiotic, comparing deterministic vs. bayesian models.

    Saves results as a CSV organized by species and set name, e.g.:
        results/{species}/{species}_{set_name}_results.csv

    Returns a DataFrame with the results.
    """

    if not np.array_equal(det_labels, bayes_labels):
        raise ValueError("Label names do not match between deterministic and bayesian models.")

    results = []

    for i, label in enumerate(det_labels):

        # AUROC
        auroc_diff, auroc_p = paired_permutation_test(
            det_true[:, i],
            det_probs[:, i],
            bayes_probs[:, i],
            metric="auroc",
            n_permutations=10000,
            random_state=42
        )

        # AUPRC
        auprc_diff, auprc_p = paired_permutation_test(
            det_true[:, i],
            det_probs[:, i],
            bayes_probs[:, i],
            metric="auprc",
            n_permutations=10000,
            random_state=42
        )

        results.append({
            "Antibiotic": label,
            "AUROC_Det": roc_auc_score(
                det_true[:, i], det_probs[:, i]
            ),
            "AUROC_Bayes": roc_auc_score(
                bayes_true[:, i], bayes_probs[:, i]
            ),
            "AUROC_Diff": auroc_diff,
            "AUROC_p": auroc_p,

            "AUPRC_Det": average_precision_score(
                det_true[:, i], det_probs[:, i]
            ),
            "AUPRC_Bayes": average_precision_score(
                bayes_true[:, i], bayes_probs[:, i]
            ),
            "AUPRC_Diff": auprc_diff,
            "AUPRC_p": auprc_p
        })

    results_df = pd.DataFrame(results)

    # Build organized output path: results/{set_name}/{species}_{set_name}_results.csv
    set_dir = os.path.join(output_dir, set_name)
    os.makedirs(set_dir, exist_ok=True)
    filename = f"{species}_{set_name}_results.csv"
    filepath = os.path.join(set_dir, filename)

    results_df.to_csv(filepath, index=False)
    print(f"Saved results to: {filepath}")

    return results_df


# Getting the results for set A E. coli
det_labels, det_true, det_probs, det_pred = load_deterministic_json(
    "Results/SetA_results/ec_detlogs/ecoli_2015_2018_detmodel_seta_logs.json"
)
bayes_labels, bayes_true, bayes_probs, bayes_pred = load_bayesian_json(
    "Results/SetA_results/ec_lasttwolayer_bayesian/ecoli_2015_2018_lasttwolayerbayes_seta_run1.json"
)

results_df = run_test_per_antibiotic(
    det_labels, det_true, det_probs,
    bayes_labels, bayes_true, bayes_probs,
    species="ecoli",
    set_name="SetA_results"
)


# Getting the results for set A S. aureus
det_labels, det_true, det_probs, det_pred = load_deterministic_json(
    "Results/SetA_results/sa_detlogs/sa_2015_2018_detmodel_seta_logs.json"
)
bayes_labels, bayes_true, bayes_probs, bayes_pred = load_bayesian_json(
    "Results/SetA_results/sa_lasttwolayer_bayesian/sa_2015_2018_lasttwolayerbayes_seta_run1.json"
)

results_df = run_test_per_antibiotic(
    det_labels, det_true, det_probs,
    bayes_labels, bayes_true, bayes_probs,
    species="sa",
    set_name="SetA_results"
)



# Getting the results for set B E. coli
det_labels, det_true, det_probs, det_pred = load_deterministic_json(
    "Results/setB_results/ec_detlogs/ecoli_2015_2018_detmodel_zeroshot_setB.json"
)
bayes_labels, bayes_true, bayes_probs, bayes_pred = load_bayesian_json(
    "Results/setB_results/ec_lasttwolayer_bayesian/ecoli_2015_2018_lasttwolayerbayesmodel_zeroshot_setB_run1.json"
)

results_df = run_test_per_antibiotic(
    det_labels, det_true, det_probs,
    bayes_labels, bayes_true, bayes_probs,
    species="ecoli",
    set_name="setB_results"
)


# Getting the results for set B S. aureus
det_labels, det_true, det_probs, det_pred = load_deterministic_json(
    "Results/setB_results/sa_detlogs/sa_2015_2018_detmodel_zeroshot_setB.json"
)
bayes_labels, bayes_true, bayes_probs, bayes_pred = load_bayesian_json(
    "Results/setB_results/sa_lasttwolayer_bayesian/sa_2015_2018_lasttwolayerbayesmodel_zeroshot_setB_run1.json"
)

results_df = run_test_per_antibiotic(
    det_labels, det_true, det_probs,
    bayes_labels, bayes_true, bayes_probs,
    species="sa",
    set_name="setB_results"
)



# Getting the results for set C E. coli
det_labels, det_true, det_probs, det_pred = load_deterministic_json(
    "Results/setC_results/ec_detlogs/ecoli_2015_2018_detmodel_finetuned_setc.json"
)
bayes_labels, bayes_true, bayes_probs, bayes_pred = load_bayesian_json(
    "Results/setC_results/ec_lasttwolayer_bayesian/ecoli_2015_2018_lasttwolayerbayes_setc_finetuned_run1.json"
)

results_df = run_test_per_antibiotic(
    det_labels, det_true, det_probs,
    bayes_labels, bayes_true, bayes_probs,
    species="ecoli",
    set_name="setC_results"
)


# Getting the results for set C S. aureus
det_labels, det_true, det_probs, det_pred = load_deterministic_json(
    "Results/setC_results/sa_detlogs/sa_2015_2018_detmodel_finetuned_setc.json"
)
bayes_labels, bayes_true, bayes_probs, bayes_pred = load_bayesian_json(
    "Results/setC_results/sa_lasttwolayer_bayesian/sa_2015_2018_lasttwolayerbayes_setc_finetuned_run1.json"
)

results_df = run_test_per_antibiotic(
    det_labels, det_true, det_probs,
    bayes_labels, bayes_true, bayes_probs,
    species="sa",
    set_name="setC_results"
)






# Getting the results for set D E. coli
det_labels, det_true, det_probs, det_pred = load_deterministic_json(
    "Results/setD_results/ec_detlogs/ecoli_2015_2018_detmodel_finetuned_setd.json"
)
bayes_labels, bayes_true, bayes_probs, bayes_pred = load_bayesian_json(
    "Results/setD_results/ec_lasttwolayer_bayesian/ecoli_2015_2018_lasttwolayerbayes_setd_finetuned_run1.json"
)

results_df = run_test_per_antibiotic(
    det_labels, det_true, det_probs,
    bayes_labels, bayes_true, bayes_probs,
    species="ecoli",
    set_name="setD_results"
)


# Getting the results for set D S. aureus
det_labels, det_true, det_probs, det_pred = load_deterministic_json(
    "Results/setD_results/sa_detlogs/sa_2015_2018_detmodel_zeroshot_setD.json"
)
bayes_labels, bayes_true, bayes_probs, bayes_pred = load_bayesian_json(
    "Results/setD_results/sa_lasttwolayer_bayesian/sa_2015_2018_lasttwolayerbayesmodel_zeroshot_setD_run1.json"
)

results_df = run_test_per_antibiotic(
    det_labels, det_true, det_probs,
    bayes_labels, bayes_true, bayes_probs,
    species="sa",
    set_name="setD_results"
)




# Getting the results for UMG E. coli
det_labels, det_true, det_probs, det_pred = load_deterministic_json(
    "Results/UMG_results/ec_detlogs/ecoli_2015_2018_detmodel_finetuned_setumg.json"
)
bayes_labels, bayes_true, bayes_probs, bayes_pred = load_bayesian_json(
    "Results/UMG_results/ec_lasttwolayer_bayesian/ecoli_2015_2018_lasttwolayerbayes_setumg_finetuned_run1.json"
)

results_df = run_test_per_antibiotic(
    det_labels, det_true, det_probs,
    bayes_labels, bayes_true, bayes_probs,
    species="ecoli",
    set_name="UMG_results"
)



# Getting the results for UMG S. aureus
det_labels, det_true, det_probs, det_pred = load_deterministic_json(
    "Results/UMG_results/sa_detlogs/sa_2015_2018_detmodel_zeroshot_setUMG.json"
)
bayes_labels, bayes_true, bayes_probs, bayes_pred = load_bayesian_json(
    "Results/UMG_results/sa_lasttwolayer_bayesian/sa_2015_2018_lasttwolayerbayesmodel_zeroshot_setUMG_run1.json"
)

results_df = run_test_per_antibiotic(
    det_labels, det_true, det_probs,
    bayes_labels, bayes_true, bayes_probs,
    species="sa",
    set_name="UMG_results"
)








