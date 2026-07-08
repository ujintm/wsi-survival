# Title
Comparative Study of MIL Aggregation Methods for WSI-based Survival Prediction on TCGA-BRCA

# Overview
This project compares four Multiple Instance Learning (MIL) aggregation methods — **Mean Pooling, Attention MIL, TransMIL, and MambaMIL** — for patient survival prediction from Whole Slide Images (WSIs).

Unlike prior work, which mostly compares MIL aggregators on classification tasks using ResNet-based patch features, this study asks a different question: **once patch features come from a strong pretrained foundation model, does aggregation complexity still matter for survival prediction?** We also examine whether combining WSI-derived features with clinical variables changes this picture.

Experiments are conducted on the TCGA-BRCA (Breast Invasive Carcinoma) cohort. After matching WSI data with clinical records, **707 patients** (avg. ~2.2 WSIs/patient) were included, with **patient-level** train/test splitting to prevent data leakage.

# Architecture
<img width="3774" height="2384" alt="그림1" src="https://github.com/user-attachments/assets/5037c6cb-ecae-477a-b586-66b4d98aebf3" />

The pipeline has two aggregation levels. **Level 1 (patch → slide)** applies the compared aggregator — Attention MIL, TransMIL, or MambaMIL — to a slide's patch features to produce a slide-level representation; this is the only stage that changes across experiments. **Level 2 (slide → patient)** applies a fixed attention aggregator (identical across all experiments) to combine a patient's multiple slide representations into a patient-level representation. This is concatenated with clinical features, passed through the survival prediction network to output a risk score, and trained with the Cox survival loss.

# Problem
- A single WSI contains tens of thousands of patches, and patients often have multiple slides
- Simple pooling (mean/max) can't capture which patches — or which slides — matter most for prognosis
- It's unclear whether increasingly complex aggregators (Attention → Transformer → Mamba) still help once patch features already come from a rich, pretrained foundation model
- Real clinical prediction should combine imaging information with clinical variables (age, stage, etc.), a setting under-explored in prior aggregation comparisons

# Method
- **Feature extraction:** Patch-level features extracted with **UNI**, a pathology foundation model, used frozen (no fine-tuning)
- **Hierarchical two-level aggregation** (patch → slide → patient):
  - **Level 1 (patch → slide):** the compared aggregator — Mean Pooling / Attention MIL / TransMIL / MambaMIL — swapped in while everything else is held fixed
  - **Level 2 (slide → patient):** a fixed attention aggregator combining multiple slide-level representations per patient, kept identical across all experiments
- Patient-level representation is concatenated with clinical features and passed through a fully connected head to output a **risk score**
- Trained with a **Cox proportional hazards** partial log-likelihood loss
- Evaluated with **C-index** (appropriate for right-censored survival data — accuracy-style metrics don't apply)
- MambaMIL trained with `use_fast_path=False` due to CUDA kernel incompatibility with the current GPU architecture (numerically equivalent, but slower)
- Gradient accumulation (batch size 1, 16-bag accumulation) used to stabilize Cox loss given the low absolute number of events

# Results (C-index, TCGA-BRCA, 707 patients, 94 events)

| Aggregation Method | C-index |
|---|---|
| **Attention MIL** | **0.7795** |
| TransMIL | 0.7553 |
| Mean Pooling | 0.7321 |
| MambaMIL | 0.7172 |

**Key finding:** aggregation complexity did **not** monotonically improve performance. With strong UNI-based features, the simpler Attention MIL outperformed both TransMIL and MambaMIL. Two likely explanations:
1. High-quality foundation-model features may already encode enough patch-level semantic information that additional structural modeling (spatial correlation, long-range sequence modeling) has less left to contribute.
2. TCGA-BRCA's low event rate (94/707 patients) may disadvantage higher-capacity models like MambaMIL, which need more learning signal than a sparse event set can reliably provide.

# Visualization (heatmap)
<img width="320" height="320" alt="survival_attention_heatmap" src="https://github.com/user-attachments/assets/5baaa412-025d-412d-84e3-a85c60064135" />

The model highlights regions contributing to survival risk prediction. Interpretation of these regions still requires validation by expert pathologists.

# Current Progress
Re-running experiments on an expanded, event-enriched cohort (**151 deaths**, up from 94) with **5-fold cross-validation**, to get more statistically robust comparisons across aggregation methods than the original single train/validation split allowed.

# Future Work
- Complete 5-fold CV results on the expanded (151-event) cohort
- Bootstrapped confidence intervals / significance testing across aggregators
- Resolve MambaMIL's CUDA fast-path incompatibility and re-evaluate
- Compare additional feature extractors (ResNet, PLIP, UNI) under the same aggregation setup
- Multi-modal extension incorporating genomic/omics data alongside WSI and clinical features


# References
[1] Ilse et al., "Attention-based Deep Multiple Instance Learning," ICML 2018.
[2] Lu et al., "Data-efficient and weakly supervised computational pathology on whole-slide images," Nature Biomedical Engineering, 2021.
[3] Shao et al., "TransMIL: Transformer based correlated multiple instance learning for whole slide image classification," NeurIPS 2021.
[4] Yang et al., "MambaMIL: Enhancing long sequence modeling with sequence reordering in computational pathology," MICCAI 2024.
[5] Chen et al., "Towards a general-purpose foundation model for computational pathology," Nature Medicine, 2024.
