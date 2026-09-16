import numpy as np
import matplotlib.pyplot as plt
import shap
import os

def plot_shap_beeswarm_panel(
    species_raw_paths,
    antibiotics_of_interest,
    label_mappings,
    output_dir="Shap_analysis/shap_plots",
    output_filename="SHAP_BEESWARM_4PANEL.png",
    max_display=15,
    n_rows=2,
    n_cols=2,
    figure_width_mm=170,   
    figure_height_mm=150, 
    dpi=300
):
    """
    Creates a multi-panel SHAP beeswarm figure across species-antibiotic combinations
    Args:
        species_raw_paths (dict): e.g.
            {
                "E. coli": "Shap_analysis/shap_values_ec.zip",
                "S. aureus": "Shap_analysis/shap_values_sa.zip"
            }
        antibiotics_of_interest (list of tuples): [(species, antibiotic_name), ...]
            defines which panels to plot, in order.
        label_mappings (dict): e.g.
            {
                "E. coli": {"label_0": "Ciprofloxacin", ...},
                "S. aureus": {"label_0": "Ciprofloxacine", ...}
            }
        output_dir (str): directory to save the figure.
        output_filename (str): filename for the saved figure.
        max_display (int): max number of features shown per beeswarm panel.
        n_rows, n_cols (int): subplot grid dimensions (must match len(antibiotics_of_interest)).
        figure_width_mm, figure_height_mm (float): journal-compliant figure size in mm.
        dpi (int): resolution for saved figure.
    """
    # create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    # define the figure size in inches (matplotlib uses inches)
    fig_width_in = figure_width_mm / 25.4
    fig_height_in = figure_height_mm / 25.4
    # create the figure and axes
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width_in, fig_height_in))
    axes = axes.flatten()
    # loop through the antibiotics of interest and plot each in its respective subplot
    for i, (species, antibiotic_name) in enumerate(antibiotics_of_interest):

        # Load raw SHAP + feature values for this species
        data = np.load(species_raw_paths[species])
        shap_array = data["shap_values"]       # (L, N, F)
        feature_array = data["feature_values"]  # (N, F)

        # Resolve antibiotic name to label index using the mapping
        label_mapping = label_mappings[species]
        label_key = None
        for k, v in label_mapping.items():
            if v.lower() == antibiotic_name.lower():
                label_key = k
                break
        if label_key is None:
            raise ValueError(f"{antibiotic_name} not found in label mapping for {species}!")

        label_index = int(label_key.split("_")[-1])
        shap_for_label = shap_array[label_index]  # (N, F)

        num_features = shap_for_label.shape[1]
        feature_names = [f"mz_{j}" for j in range(num_features)]

        explanation = shap.Explanation(
            values=shap_for_label,
            data=feature_array,
            feature_names=feature_names
        )

        # Activate the target subplot so SHAP draws into it
        plt.sca(axes[i])
        shap.plots.beeswarm(explanation, max_display=max_display, show=False)

        # Subplot label only — no title
        axes[i].set_xlabel(axes[i].get_xlabel(), fontsize=8)
        axes[i].tick_params(axis='both', which='major', labelsize=7)
        axes[i].text(
            -0.15, 1.05, f"{species} - {antibiotic_name}",
            transform=axes[i].transAxes,
            fontsize=9, fontweight='bold', va='bottom'
        )

    plt.tight_layout()
    out_path = os.path.join(output_dir, output_filename)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")





mapping_ec = {"label_0": "Ciprofloxacin",
              "label_1": "Ceftriaxone",
              "label_2": "Piperacillin-Tazobactam",
              "label_3": "Cefepime",
              "label_4": "Tobramycin"}

mapping_sa = {"label_0": "Ciprofloxacine",
              "label_1": "Fusidic Acid",
              "label_2": "Oxacillin",
              "label_3": "Ceftriaxone",
              "label_4": "Clindamycin"}

species_raw_paths = {
    "E. coli": "Shap_analysis/shap_raw_values_ec.zip",
    "S. aureus": "Shap_analysis/shap_raw_values.zip"
}

label_mappings = {
    "E. coli": mapping_ec,
    "S. aureus": mapping_sa
}

antibiotics_of_interest = [
    ("E. coli", "Ceftriaxone"),
    ("E. coli", "Cefepime"),
    ("S. aureus", "Ceftriaxone"),
    ("S. aureus", "Oxacillin"),
]

plot_shap_beeswarm_panel(
    species_raw_paths=species_raw_paths,
    antibiotics_of_interest=antibiotics_of_interest,
    label_mappings=label_mappings
)