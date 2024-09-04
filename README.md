# ASD-Diagnosis-Using-Multimodal-MRI

**Official code for our paper "Diagnosis and Pathogenic Analysis of Autism Spectrum Disorder Using Fused Brain Connection Graph".**

This repository includes four GNN models of ASD Diagnose: GCN, GAT, ChebyNet and graphSAGE, two plotting codes and the MWU test code. The Python file ending with "+loss" is the model that introduces the Wasserstein Graph Distance.

Both CPU and GPU environments are supported.

## Requirements

Recommended version:

* **Python**: python 3.7 
* **torch**: torch 1.12.0

## Figures

Our charts are placed below：
### Figure 1: The overall framework of the proposed model. It consists of three modules.
<img src="./figures/framework.png" alt="frontpage" width="750"/>


### Figure 2: fMRI

<img src="./figures/fmri.png" alt="frontpage" width="450"/>

### Figure 3: DTI
<img src="./figures/dti.png" alt="frontpage" width="450"/>

### Figure 4: Visualizations of ASD classification on TNR (left panel), TPR (middle panel), and AUC (right panel) (with standard deviation) of baseline methods and the proposed models, averaged over five cross-validation folds. ``(+ $\mathcal{R}_{g}$)'' denotes the proposed regularization-based GNN model.
<img src="./figures/bar_plot_1.png" alt="frontpage" width="450"/>


### Figure 5: ASD related brain regions visualization. The left (right) panel shows the significant regions of the left (right) half of the brain from the lateral and dorsal views. And the middle panel displays the top 15 regions from the medial view.
<img src="./figures/bar_plot_2.png" alt="frontpage" width="450"/>










