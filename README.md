# Uncertainty-Aware Antimicrobial Resistance Prediction

This repository contains the full code accompanying the paper **“Uncertainty-Aware Antimicrobial Resistance Prediction”**, including data preprocessing, deterministic and Bayesian model training, uncertainty quantification, SHAP explainability, and UMAP feature visualization.



# 1. Dataset Preparation

This project uses the **DRIAMS A/B/C/D** and **MS-UMG** datasets.

You can either **use the preprocessed/binned datasets** (recommended) or **rebuild them from raw spectra**.


## Option 1 — Use Pre-Binned Data (Recommended)

All preprocessed datasets (3 Da bins, 2000–20000 m/z) are available on Kaggle:

**Kaggle Binned Dataset:**  
*(https://www.kaggle.com/datasets/sumanpunshi/driams-processed-set)*

These CSVs already contain:
- First 6000+ columns → binned MALDI-TOF intensity features  
- Remaining columns → metadata (species, antibiotic labels, codes)

You can use them directly with the training scripts.

## Option 2 — Preprocess Raw Data Yourself

you can also bin the raw spectra manually

### **Step 1 — Download Raw DRIAMS Data**
Download the DRIAMS datasets and MS-UMG from *https://datadryad.org/dataset/doi:10.5061/dryad.bzkh1899q* and *https://zenodo.org/records/13911744*. Place them into the project directory.

### **Step 2 — Run the binning script**

```bash
python Binning_DRIAMS_UMG.py
```

# 2. Training and Evaluation

The project contains three main experiment folders, one for each species.

## E. coli Experiments (Ecoli_notebooks/)
All of the following notebooks are available in the Kaggle collection *https://www.kaggle.com/work/collections/16594503*. For exact reproducibility, we recommend forking the notebooks directly on Kaggle and running them there.

- ecoli-2015-2018-detmodel-seta.ipynb -> Training and testing deterministic model on set A
- ecoli-2015-2018-lastlayerbayes-seta.ipynb -> Training and testing Bayesian model with last layers made bayesian on set A
- ecoli-2015-2018-lasttwolayersbayes-seta.ipynb -> Training and testing Bayesian model with last two layers made bayesian on set A
- ecoli-2015-2018-detmodel-setab.ipynb -> Zero-shot testing of deterministic and Bayesian Model on set B
- e-coli-2015-2018-detmodel-setac.ipynb -> Fine-tuning of deterministic and Bayesian model on set C
- ecoli-2015-2018-setad.ipynb -> Fine-tuning of deterministic and Bayesian model on set D
- ecoli-2015-2018-ftp-a-umg.ipynb -> Fine-tuning of deterministic and Bayesian model on MS-UMG


## K. pneumoniae Experiments ( Kpneumoniae_notebooks/)
All of the following notebooks are available in the Kaggle collection *https://www.kaggle.com/work/collections/16595021*. For exact reproducibility, we recommend forking the notebooks directly on Kaggle and running them there.

- kp-2015-2018-detmodel-seta.ipynb -> Training and testing deterministic model on set A
- kp-2015-2018-lastlayersbayes-seta.ipynb -> Training and testing Bayesian model with last layers made bayesian on set A
- kp-2015-2018-lasttwolayersbayes-seta.ipynb -> Training and testing Bayesian model with last two layers made bayesian on set A
- kp-2015-2018-setab.ipynb -> Zero-shot testing of deterministic and Bayesian Model on set B
- kp-2015-2018-ad.ipynb -> Fine-tuning of deterministic and Bayesian model on set D
- kp-2015-2018-ftp-a-umg.ipynb -> Fine-tuning of deterministic and Bayesian model on MS-UMG

## S. aureus Experiments (Saureus_notebooks/)
All of the following notebooks are available in the Kaggle collection *https://www.kaggle.com/work/collections/16595142*. For exact reproducibility, we recommend forking the notebooks directly on Kaggle and running them there.

- saureus-2015-2018-detmodel-seta.ipynb -> Training and testing deterministic model on set A
- saureus-2015-2018-lastlayersbayes-seta.ipynb -> Training and testing Bayesian model with last layers made bayesian on set A
- saureus-2015-2018-lasttwolayersbayes-seta.ipynb -> Training and testing Bayesian model with last two layers made bayesian on set A
- sa-2015-2018-setab.ipynb -> Zero-shot testing of deterministic and Bayesian Model on set B
- saureus-2015-2018-setac.ipynb -> Fine-tuning of deterministic and Bayesian model on set C
- sa-2015-2018-setad.ipynb -> Fine-tuning of deterministic and Bayesian model on set D
- sa-2015-2018-a-umg.ipynb -> Fine-tuning of deterministic and Bayesian model on MS-UMG

To make execution easy, each notebook is fully self-contained and includes all utility functions required for that experiment.


# 3. Results and Analysis (Results/)

This folder contain four subfolders,

- SetA_results/ -> this folder contain logs for all the experiments done on set A for all species. For example, ec_detlogs/ contains the logs on deterministic model, ec__lasttwolayerbayesian_logs/ contain the logs for the last two layer Bayesian model. Once you run the experiment notebooks on kaggle, you can upload the generate logs in the relevant sub folder and the run the python scripts. average_runs_lasttwolayer_bayesian.py averages and the metrics of five run from Bayesian model. graphing_results.py plots the histograms and uncertainty distributions from the logs.
- SetB_results/, SetC_results/, SetD_results/, UMG_results/ -> these folders also have structure similar to SetA_results folder.

# 4. SHAP Analysis (Shap_analysis/) 

All the notebooks for computing SHAP values are available in kaggle collection *https://www.kaggle.com/work/collections/16612328*. We recommend forking the notebooks directly on Kaggle and running them there.

- This folder contains the SHAP logs for each specie, tested on set A.
- plot_shap_distributions.py plot the shap values from the SHAP logs file
