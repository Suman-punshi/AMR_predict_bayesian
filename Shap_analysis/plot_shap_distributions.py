import json
import os
import numpy as np
import matplotlib.pyplot as plt

import json
import os
import numpy as np
import matplotlib.pyplot as plt

def plot_shap_distributions_from_json(
    label_mapping,                 
    json_path="shap_summary.json",
    output_dir="dir",   
    output_label_prefix="label_",
    show=True,
    figure_width_mm=170,  
    figure_height_mm=80, 
    dpi=300
):
    # Convert mm → inches
    figure_width = figure_width_mm / 25.4
    figure_height = figure_height_mm / 25.4

    # Ensure directory exists
    if not os.path.isdir(output_dir):
        raise ValueError(f"Output directory {output_dir} does not exist!")

    # Load the SHAP summary JSON
    with open(json_path, "r") as f:
        shap_data = json.load(f)

    for label_key, val_dict in shap_data.items():
        resistant = np.array(val_dict["resistant_mean"])
        susceptible = np.array(val_dict["susceptible_mean"])
        num_bins = len(resistant)
        bins = np.arange(num_bins)

        # Get antibiotic name from mapping (fallback to label_key if not found)
        antibiotic_name = label_mapping.get(str(label_key), label_key)

        # Create plots for each label
        plt.figure(figsize=(figure_width, figure_height))
        plt.plot(bins, resistant, label="Resistant", color='red', linewidth=1)
        plt.plot(bins, susceptible, label="Susceptible", color='blue', linewidth=1)
        plt.xlabel("m/z Bin Index", fontsize=10)
        plt.ylabel("Mean SHAP Value", fontsize=10)
        plt.title(f"{antibiotic_name}", fontsize=11)   # use antibiotic name
        plt.legend(fontsize=9)
        plt.grid(True, linewidth=0.3)
        plt.tight_layout()

        # Build file path inside given directory
        out_path = os.path.join(output_dir, f"{output_label_prefix}{antibiotic_name}_shap_plot.png")

        # Save only in PNG
        plt.savefig(
            out_path,
            format="png",
            dpi=dpi,
            bbox_inches="tight"
        )

        if show:
            plt.show()
        plt.close()




mapping_ec = {"label_0": "Ciprofloxacin", 
              "label_1": "Ceftriaxone", 
              "label_2": "Piperacillin-Tazobactam",
              "label_3": "Cefepime",
              "label_4": "Tobramycin"}

# ecoli --- set A
plot_shap_distributions_from_json(
    label_mapping=mapping_ec,
    json_path="Shap_analysis/shap_summary_ecoli.json",
    output_dir="Shap_analysis/shap_plots",  
    output_label_prefix="ecoli_",
    show=False
)


mapping_kp = {"label_0": "Ciprofloxacin", 
              "label_1": "Ceftriaxone", 
              "label_2": "Cefepime",
              "label_3": "Meropenem",
              "label_4": "Tobramycin"}


# kp --- set A
plot_shap_distributions_from_json(
    label_mapping=mapping_kp,
    json_path="Shap_analysis/shap_summary_kp.json",
    output_dir="Shap_analysis/shap_plots",  
    output_label_prefix="kp_",
    show=False
)

mapping_sa = {"label_0": "Ciprofloxacine", 
              "label_1": "Fusidic Acid", 
              "label_2": "Oxacillin",
              "label_3": "Ceftriaxone",
              "label_4": "Clindamycin"}



# sa --- set A
plot_shap_distributions_from_json(
    label_mapping=mapping_sa,
    json_path="Shap_analysis/shap_summary_sa.json",
    output_dir="Shap_analysis/shap_plots", 
    output_label_prefix="sa_", 
    show=False
)
