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
![model](https://github.com/lobster2023/ASD-Diagnose-with-GNNs/assets/133120607/fd9a7606-cdfd-4a8e-8705-5d3eff93902f)

### Figure 2: ROC curve for GCN, GAT, ChebyNet and graphSAGE.
![roc](https://github.com/lobster2023/ASD-Diagnose-with-GNNs/assets/133120607/64e3e045-aac1-4c79-8c84-4e81d8cb8525)

### Figure 3: Hyper-parameter 𝑎 search.
![para_new](https://github.com/lobster2023/ASD-Diagnose-with-GNNs/assets/133120607/1f77c961-7553-4d4d-a2fc-d6ddf27f6605)

### Figure 4: Mann-Whitney U test of 90 functional areas from the ASD group (orange) and the control group (blue).
![MWU-test-GCN_1](https://github.com/lobster2023/ASD-Diagnose-with-GNNs/assets/133120607/4fbd633c-9366-457b-924a-0589ed6d447c)
![MWU-test-GCN_2](https://github.com/lobster2023/ASD-Diagnose-with-GNNs/assets/133120607/3605a8f5-fa6d-4443-9f98-d5638ebc7351)
![MWU-test-GCN_3](https://github.com/lobster2023/ASD-Diagnose-with-GNNs/assets/133120607/136488c0-9612-4024-b685-49dbce649a06)
![MWU-test-GCN_4](https://github.com/lobster2023/ASD-Diagnose-with-GNNs/assets/133120607/bd5a710e-1ce1-4d5b-b963-1f92d62aff64)
![MWU-test-GCN_5](https://github.com/lobster2023/ASD-Diagnose-with-GNNs/assets/133120607/fa2d719f-e414-44e6-aa0e-65fb90183876)

### Figure 5: ASD related brain regions visualization. The left (right) panel shows the significant regions of the left (right) half of the brain from the lateral and dorsal views. And the middle panel displays the top 15 regions from the medial view.
![brainRegions](https://github.com/lobster2023/ASD-Diagnose-with-GNNs/assets/133120607/a767db45-88b7-40c0-bc43-e683d09293a9)










