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
    specie_name = "K. pneumoniae",
    show=True,
    figure_width_mm=170,  
    figure_height_mm=80, 
    dpi=300
):
    # Convert mm → inches
    figure_width = figure_width_mm / 25.4
    figure_height = figure_height_mm / 25.4

    # Ensure directory exists; if not, create it
    os.makedirs(output_dir, exist_ok=True)


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
        plt.title(f"{specie_name}-{antibiotic_name}", fontsize=11)   # use antibiotic name
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




def plot_shap_six_panel(
    species_label_mapping_json_paths,
    antibiotics_of_interest,
    output_dir="dir",
    figure_width_mm=170,   # full-page width
    figure_height_mm=225,  # full-page height
    dpi=300
):
    """
    Creates a 6-panel SHAP summary figure (2 columns x 3 rows) for selected antibiotics across species.

    Args:
        species_label_mapping_json_paths (dict): dict with species as keys and dict with 'json_path' and 'label_mapping'
        antibiotics_of_interest (list of tuples): [(species, antibiotic_name), ...] for the 6 panels
        output_path (str): path to save the figure
        figure_width_mm (float): figure width in mm
        figure_height_mm (float): figure height in mm
        dpi (int): resolution in dpi
    """

    # Ensure directory exists; if not, create it
    os.makedirs(output_dir, exist_ok=True)
    # Convert mm → inches
    fig_width_in = figure_width_mm / 25.4
    fig_height_in = figure_height_mm / 25.4

    fig, axes = plt.subplots(3, 2, figsize=(fig_width_in, fig_height_in))
    axes = axes.flatten()

    for i, (species, antibiotic_name) in enumerate(antibiotics_of_interest):
        json_path = species_label_mapping_json_paths[species]['json_path']
        label_mapping = species_label_mapping_json_paths[species]['label_mapping']

        # Load JSON data
        with open(json_path, "r") as f:
            shap_data = json.load(f)

        # Find the corresponding label key for the antibiotic
        label_key = None
        for k, v in label_mapping.items():
            if v.lower() == antibiotic_name.lower():
                label_key = k
                break
        if label_key is None:
            raise ValueError(f"{antibiotic_name} not found in label mapping for {species}!")

        val_dict = shap_data[label_key]
        resistant = np.array(val_dict["resistant_mean"])
        susceptible = np.array(val_dict["susceptible_mean"])
        bins = np.arange(len(resistant))

        ax = axes[i]

        # Color-invariant: use dashed lines for susceptible, solid lines for resistant
        ax.plot(bins, resistant, color='red', linewidth=1, label="Resistant")
        ax.plot(bins, susceptible, color='blue', linewidth=1, label="Susceptible")
    

        ax.set_title(f"{species} - {antibiotic_name}", fontsize=10, fontweight='bold')
        ax.set_xlabel("m/z Bin Index", fontsize=8)
        ax.set_ylabel("Mean SHAP Value", fontsize=8)
        ax.grid(True, linewidth=0.25)
        ax.tick_params(axis='both', which='major', labelsize=7)

    # Add a single legend in top-right
    handles, labels_ = axes[0].get_legend_handles_labels()

    out_path = os.path.join(output_dir, f"SHAP_6PANEL.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")





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



species_label_mapping_json_paths = {
    "E. coli": {"json_path": "Shap_analysis\shap_summary_ecoli.json", "label_mapping": mapping_ec},
    "K. pneumoniae": {"json_path": "Shap_analysis\shap_summary_kp.json", "label_mapping": mapping_kp},
    "S. aureus": {"json_path": "Shap_analysis/shap_summary_sa.json", "label_mapping": mapping_sa},
}

antibiotics_of_interest = [
    ("E. coli", "Ceftriaxone"),
    ("E. coli", "Cefepime"),
    ("K. pneumoniae", "Ceftriaxone"),
    ("K. pneumoniae", "Cefepime"),
    ("S. aureus", "Ceftriaxone"),
    ("S. aureus", "Oxacillin"),
]

plot_shap_six_panel(
    species_label_mapping_json_paths=species_label_mapping_json_paths,
    antibiotics_of_interest=antibiotics_of_interest,
    output_dir="Shap_analysis/shap_plots"
)
