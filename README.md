# Title
Attention-based MIL for Survival Prediction on Whole Slide Images  

# Architecture
<img width="704" height="384" alt="Gemini_Generated_Image_cysjcicysjcicysj" src="https://github.com/user-attachments/assets/c74183ea-cc0a-403c-a3dc-ea073d6ffb0c" />  
  
Figure: Architecture of the proposed model. Generated using NanoBanana.  
The model replaces traditional mean pooling with learnable gated attention to preserve discriminative pathological regions for survival prediction.  

# Overview
This project proposes an attention-based Multiple Instance Learning (MIL) framework for survival prediction using Whole Slide Images (WSIs).  
We address the limitation of mean pooling, which fails to capture the importance of discriminative regions in gigapixel pathology images.  
We redesign the aggregation and training objective on top of the CLAM framework for survival prediction.  
Experiments are conducted on a subset of the TCGA-BRCA dataset (NIH), consisting of 190 patients. 
A stratified train/validation split is applied based on the event distribution.


# Problem
- WSIs contain tens of thousands of patches
- Traditional pooling methods (mean/max) are non-learnable
- Important pathological regions are diluted in mean pooling

# Method
- Patch features extracted using a pretrained ResNet-50 within the CLAM pipeline
- Adapted CLAM for survival analysis by removing the instance-level classifier and optimizing for Cox-based risk prediction
- Gated Attention MIL for learnable aggregation
- Cox proportional hazards loss for survival prediction
- Integration with 9 clinical variables
  
The model learns to assign higher weights to patches contributing to survival risk.

# Results
- Validation C-index: 0.7451
- Clinical-only Cox model: 0.6863
- Mean pooling: severe overfitting (Train C-index: 1.0, poor generalization)

# Visualization (heatmap)
<img width="320" height="320" alt="survival_attention_heatmap" src="https://github.com/user-attachments/assets/5baaa412-025d-412d-84e3-a85c60064135" />  
  
The model identifies regions contributing to survival prediction.  
However, interpretation of these regions requires validation by expert pathologists.  

# Future Work
- Transformer-based MIL (TransMIL)
- Multi-scale modeling
- Segmentation-based ROI extraction
  
