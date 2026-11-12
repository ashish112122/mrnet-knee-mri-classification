# Methodology

## Problem Formulation

Each exam in the MRNet dataset is associated with three independent binary labels: **abnormality**, **ACL tear**, and **meniscal tear**. Rather than a single multi-class model, each label is treated as a separate binary classification problem. All models are trained with `BCEWithLogitsLoss`, and positive-class weights are computed as the ratio of negative to positive examples to handle class imbalance.

## MRI Volume Representation

Each MRI exam consists of a 3D volume per plane (sagittal, coronal, axial), stored as a NumPy array of shape `(num_slices, H, W)`. Slices are grayscale and replicated to 3 channels before being passed through CNN backbones. Images are resized to 224×224 and normalised with ImageNet statistics.

## Experiment Progression

### Experiment 1 — ResNet18 Sagittal Mean Pooling (Baseline)

The simplest possible architecture. A frozen, ImageNet-pretrained ResNet18 encodes each sagittal slice into a 512-dimensional vector. These are averaged across all slices (mean pooling) to produce a single exam embedding. A linear layer maps this to three logits.

Only the classifier head is trained (~1.5K parameters). This establishes a performance floor.

### Experiment 2 — ResNet18 Sagittal Attention MIL

Identical backbone to Experiment 1, but mean pooling is replaced with an **Attention Multiple Instance Learning** (MIL) module. The attention module learns a scalar weight for each slice, allowing the model to focus on clinically relevant frames (e.g., slices passing through the ACL). The weighted average replaces the unweighted mean.

Trainable parameters increase to ~130K (attention module + classifier). AMP (mixed-precision) is used for memory efficiency.

### Experiment 3 — ResNet18 Three-View Mean Pooling

The same frozen ResNet18 encoder processes all three MRI planes independently. Per-plane mean-pooled embeddings (512-d each) are concatenated into a 1,536-d vector, then classified with a linear layer.

This tests whether multi-plane information improves over single-plane, without increasing training complexity.

### Experiment 4 — TripleMRNet-Inspired Three-Plane AlexNet

Inspired by the TripleMRNet architecture. Three **independent, fine-tuned AlexNet** encoders (one per plane) each produce a 256-d slice embedding, which is mean-pooled to a plane embedding. The three plane embeddings are concatenated (768-d) and fed into a classifier with dropout (0.5).

Crucially, this is the only experiment that **fine-tunes the backbone** (at a low learning rate of 1e-5) rather than keeping it frozen. Three separate binary models are trained — one per label — for 10 epochs each. AdamW is used with differential learning rates.

## Evaluation

All models are evaluated on the held-out validation set using:

- **ROC-AUC**: primary ranking metric
- **Average Precision (AP)**: area under the precision-recall curve
- **Accuracy, Precision, Recall (Sensitivity), Specificity, F1**: at a 0.5 decision threshold

A fixed random seed (42) is used across all experiments for reproducibility.
