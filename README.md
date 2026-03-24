# Title
Attention-based MIL for Survival Prediction on Whole Slide Images  

# Architecture
<img width="2816" height="1536" alt="Gemini_Generated_Image_cysjcicysjcicysj" src="https://github.com/user-attachments/assets/c74183ea-cc0a-403c-a3dc-ea073d6ffb0c" />
Figure: Architecture of the proposed model. Generated using NanoBanana.  
The model replaces traditional mean pooling with learnable gated attention to preserve discriminative pathological regions for survival prediction.  

# Overview
This project proposes an attention-based Multiple Instance Learning (MIL) framework for survival prediction using Whole Slide Images (WSIs).  
We address the limitation of mean pooling, which fails to capture the importance of discriminative regions in gigapixel pathology images.  

# Problem
- WSIs contain tens of thousands of patches
- Traditional pooling methods (mean/max) are non-learnable
- Important pathological regions are diluted in mean pooling

# Method
- Patch feature extraction using pretrained ResNet-50 (CLAM pipeline)
- Adapted CLAM for survival analysis by removing instance-level classifier
- Gated Attention MIL for learnable aggregation
- Cox proportional hazards loss for survival prediction
- Integration with 9 clinical variables
  
The model learns to assign higher weights to patches contributing to survival risk.

# Results
- Validation C-index: 0.7451
- Clinical-only Cox model: 0.6863
- Mean pooling: severe overfitting (Train C-index: 1.0, poor generalization)

# Visualization (heatmap)
<img width="2048" height="2048" alt="survival_attention_heatmap" src="https://github.com/user-attachments/assets/5baaa412-025d-412d-84e3-a85c60064135" />
Attention heatmaps highlight regions contributing to survival prediction.

# Future Work
- Transformer-based MIL (TransMIL)
- Multi-scale modeling
- Segmentation-based ROI extraction
  
