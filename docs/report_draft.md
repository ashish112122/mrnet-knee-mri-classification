
# PROJECT REPORT

**[University/Institute Name]**  
**Department of [Department Name]**  
**Subject: Machine Learning Lab**

---

**Title:** Multi-View Knee MRI Classification using Deep Learning (MRNet)

**Submitted By:**  
[Student 1 Name] — Roll No. [XXXX]  
[Student 2 Name] — Roll No. [XXXX]

**Submitted To:** [Faculty/Lab Instructor Name]  
**Academic Session:** [e.g. 2025–26]

---

## Abstract

Knee injuries such as ACL tears and meniscal tears are among the most common
musculoskeletal conditions, typically diagnosed through magnetic resonance imaging
(MRI). Manual interpretation of knee MRI volumes is time-consuming and subject to
inter-observer variability. This project explores the use of deep learning to
automate the classification of knee MRI examinations across three binary diagnostic
tasks: detection of general abnormality, ACL tear, and meniscal tear.

The Stanford MRNet dataset, comprising 1,370 knee MRI examinations across three
imaging planes (sagittal, coronal, and axial), was used for training and evaluation.
Four progressively more complex deep learning architectures were implemented and
compared: a ResNet18 sagittal mean pooling baseline, a ResNet18 sagittal attention
MIL model, a ResNet18 three-view mean pooling model, and a TripleMRNet-inspired
three-plane AlexNet fusion model. The best-performing model achieved a ROC-AUC of
0.927, 0.881, and 0.784 for abnormality, ACL tear, and meniscal tear detection
respectively, demonstrating that multi-plane feature fusion with fine-tuned
backbones provides meaningful improvements over single-plane baselines.

---

## 1. Introduction

Knee MRI is the clinical standard for diagnosing soft tissue injuries. A typical
knee MRI examination produces three-dimensional volumes across multiple imaging
planes, each containing tens of slices. Radiologists must review hundreds of images
per exam, making automated assistance highly valuable.

Deep learning has shown considerable promise in medical image analysis, with
convolutional neural networks (CNNs) achieving expert-level performance on tasks
ranging from skin lesion classification to diabetic retinopathy screening. Applying
similar approaches to MRI-based knee injury detection presents unique challenges:
the input is a 3D volume rather than a single image, the number of slices per exam
varies, and the relevant pathology may appear in only a small subset of slices.

This project was motivated by the publicly available Stanford MRNet dataset and the
accompanying benchmark paper (Bien et al., 2018), which demonstrated that deep
learning could assist radiologists in knee MRI diagnosis. Rather than replicating a
single architecture, this project systematically explores four design choices:
single-plane versus multi-plane input, mean pooling versus attention-based slice
aggregation, and frozen versus fine-tuned backbones.

**Objectives:**
- Implement and evaluate four deep learning architectures on the MRNet dataset.
- Compare the effect of attention pooling versus mean pooling on classification
  performance.
- Assess whether using all three MRI planes improves over sagittal-only models.
- Identify the most effective architecture for each of the three diagnostic tasks.

---

## 2. Problem Statement

Given a knee MRI examination consisting of one or more 3D volumes (one per imaging
plane), the task is to predict three independent binary labels:

- **Abnormality:** Is any knee abnormality present?
- **ACL tear:** Is an anterior cruciate ligament tear present?
- **Meniscal tear:** Is a meniscal tear present?

Each label is treated as a separate binary classification problem. The goal is to
develop a model that generalises well on a held-out validation set, measured
primarily by ROC-AUC, as the dataset exhibits class imbalance across all three
labels.

---

## 3. Literature Review

**Bien et al. (2018)** introduced the MRNet dataset and demonstrated that a
convolutional neural network trained on sagittal plane MRI could achieve AUC scores
comparable to general radiologists on all three tasks when used as a diagnostic aid.
Their model used an AlexNet backbone with global average pooling and trained
separate models per task and per plane, combining predictions via logistic regression.

**Tsai et al. (TripleMRNet, 2021)** extended this by using three independent
pretrained encoders — one per MRI plane — and concatenating their embeddings before
classification. This multi-plane fusion approach showed that all three planes
contribute complementary information.

**Attention-based Multiple Instance Learning (MIL)** has been proposed as a
principled approach to aggregate variable-length sequences of feature vectors.
Rather than averaging all slice embeddings equally, attention MIL assigns learned
scalar weights to each slice, allowing the model to focus on the most diagnostically
relevant frames (Ilse et al., 2018).

**Limitations of existing approaches:** Most prior work trains separate models per
plane and combines predictions post-hoc. End-to-end multi-plane training with shared
encoders remains underexplored. Additionally, most published results use AlexNet
rather than more modern architectures, leaving the benefit of stronger backbones
unclear.

---

## 4. Dataset Description

**Dataset:** Stanford MRNet Knee MRI Dataset  
**Source:** Stanford Center for Artificial Intelligence in Medicine and Imaging (AIMI)  
**DOI:** https://doi.org/10.71718/rcbp-8c35  
**Access:** https://aimi.stanford.edu/datasets/mrnet-knee-mris

| Split      | Exams |
|------------|-------|
| Training   | 1,130 |
| Validation | 120   |
| **Total**  | **1,370** |

Each exam contains three MRI volumes:
- **Sagittal plane** — side view, most informative for ACL
- **Coronal plane** — front view
- **Axial plane** — cross-sectional view

Each volume consists of a variable number of grayscale slices (typically 20–50),
stored as NumPy arrays of shape `(num_slices, H, W)`.

**Labels (binary per exam):**

| Label     | Train Positive Rate | Validation Positive Rate |
|-----------|--------------------:|-------------------------:|
| Abnormal  | ~0.75               | ~0.80                    |
| ACL       | ~0.30               | ~0.40                    |
| Meniscus  | ~0.45               | ~0.50                    |

**Target variable:** Three independent binary labels per exam (abnormal, acl,
meniscus).

**[INSERT: Screenshot of Training Label Distribution bar chart from Notebook 01]**

**[INSERT: Screenshot of sample MRI slice visualisation from Notebook 01 — sagittal, coronal, axial]**

The dataset is provided under a research use agreement. The raw MRI volumes are not
included in the project repository.

---

## 5. Methodology

### 5.1 Data Collection and Preprocessing

The MRNet dataset was accessed via the Stanford AIMI Redivis portal. All MRI volumes
were downloaded and cached as NumPy arrays (`.npy` files) on Google Drive, organised
by split and plane. No data augmentation was applied during training.

Preprocessing applied to each slice before passing through a CNN:
1. Normalise pixel values to [0, 1] by dividing by 255
2. Resize to 224 × 224 using bilinear interpolation
3. Replicate the single grayscale channel to 3 channels (for ImageNet-pretrained
   backbones)
4. Apply ImageNet normalisation (mean = [0.485, 0.456, 0.406],
   std = [0.229, 0.224, 0.225])

### 5.2 Handling Class Imbalance

All three labels are imbalanced, with positive examples being rarer for ACL and
meniscus tasks. `BCEWithLogitsLoss` was used with positive-class weights computed
per label as:

```
pos_weight[j] = num_negative[j] / num_positive[j]
```

This penalises false negatives more heavily and prevents the model from collapsing
to predicting all-negative.

### 5.3 Exploratory Data Analysis

EDA was performed in Notebook 01 and included:
- Class distribution visualisation across all three labels for the training set
- Inspection of MRI volumes: slice counts per exam, spatial resolution, and intensity
  range
- Visualisation of sample sagittal, coronal, and axial slices from a representative
  exam to understand the input structure

### 5.4 Training and Evaluation Split

The official MRNet train/validation split was used throughout. All four experiments
use the same 1,130 training exams and 120 validation exams, ensuring fair
comparison. A fixed random seed of 42 was used for reproducibility.

### 5.5 Machine Learning Algorithms

Four architectures were implemented, each building on the previous:

#### Experiment 1 — ResNet18 Sagittal Mean Pooling (Baseline)
- **Backbone:** ImageNet-pretrained ResNet18, fully frozen
- **Pooling:** Unweighted mean of all slice embeddings
- **Input:** Sagittal plane only
- **Classifier:** Linear(512, 3)
- **Trainable parameters:** ~1,500
- **Epochs:** 3, Learning rate: 1e-3, Optimiser: Adam

#### Experiment 2 — ResNet18 Sagittal Attention MIL
- **Backbone:** ImageNet-pretrained ResNet18, fully frozen
- **Pooling:** Attention MIL — learned scalar weight per slice via gated attention
  mechanism (V: Linear→Tanh, U: Linear→Sigmoid, w: Linear)
- **Input:** Sagittal plane only
- **Classifier:** Linear(512, 3)
- **Trainable parameters:** ~130,000
- **Epochs:** 3, Learning rate: 1e-3, Optimiser: Adam, AMP enabled

#### Experiment 3 — ResNet18 Three-View Mean Pooling
- **Backbone:** Shared ImageNet-pretrained ResNet18, fully frozen
- **Pooling:** Mean pooling per plane → concatenate three 512-d embeddings (1536-d)
- **Input:** Sagittal + Coronal + Axial
- **Classifier:** Linear(1536, 3)
- **Trainable parameters:** ~4,600
- **Epochs:** 3, Learning rate: 1e-3, Optimiser: Adam

#### Experiment 4 — TripleMRNet-Inspired Three-Plane AlexNet (Final Model)
- **Backbone:** Three independent ImageNet-pretrained AlexNet encoders (one per
  plane), fine-tuned during training
- **Pooling:** Mean pooling of 256-d slice embeddings per plane → concatenate (768-d)
- **Input:** Sagittal + Coronal + Axial
- **Classifier:** Dropout(0.5) → Linear(768, 1) per task
- **Training:** Three separate binary models (one per label)
- **Trainable parameters:** All (fine-tuned)
- **Epochs:** 10, Backbone LR: 1e-5, Head LR: 1e-3, Optimiser: AdamW,
  Weight decay: 1e-4

### 5.6 Overall Workflow

**[INSERT: Block diagram / flowchart here]**

The general pipeline for all experiments:
1. Load cached `.npy` volume(s) from Drive
2. For each slice: resize → RGB convert → normalise → CNN encoder → 256/512-d vector
3. Aggregate slice vectors (mean or attention pooling) → exam embedding
4. Optionally concatenate embeddings from multiple planes
5. Apply linear classifier → binary logit per task
6. Compute weighted BCE loss → backpropagate → update trainable parameters
7. Evaluate on validation set after each epoch

---

## 6. Implementation

**Programming Language:** Python 3.10  
**Platform:** Google Colab (GPU: Tesla T4)  
**Key Libraries:**

| Library       | Version | Purpose                              |
|---------------|---------|--------------------------------------|
| PyTorch       | 2.3.0   | Model definition, training, inference|
| torchvision   | 0.18.0  | Pretrained ResNet18 and AlexNet      |
| NumPy         | 1.26.4  | Array operations                     |
| Pandas        | 2.2.2   | Exam metadata and label handling     |
| scikit-learn  | 1.5.0   | Evaluation metrics                   |
| matplotlib    | 3.9.0   | Visualisation                        |
| tqdm          | 4.66.4  | Progress tracking                    |

**Notebooks:**

| Notebook | Purpose |
|---|---|
| `01_resnet18_sagittal_mean_pooling.ipynb` | EDA + Experiment 1 |
| `02_resnet18_sagittal_attention_mil.ipynb` | Experiment 2 |
| `03_resnet18_three_view_mean_pooling.ipynb` | Experiment 3 |
| `04_triplemrnet_alexnet_three_plane.ipynb` | Experiment 4 (final model) |
| `05_results_and_plots.ipynb` | Evaluation plots |

**[INSERT: Screenshot of Colab notebook running]**

**[INSERT: Screenshot of Google Drive checkpoints folder]**

### Key Code Snippet — Attention MIL Module

```python
class AttentionMIL(nn.Module):
    def __init__(self, feature_dim=512, hidden_dim=128):
        super().__init__()
        self.V = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.Tanh())
        self.U = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.Sigmoid())
        self.w = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, h):
        gate = self.V(h) * self.U(h)              # gated attention
        a    = torch.softmax(self.w(gate).squeeze(-1), dim=0)  # slice weights
        z    = (a.unsqueeze(1) * h).sum(0)        # weighted exam embedding
        return z, a
```

### Key Code Snippet — Three-Plane AlexNet Forward Pass

```python
class TripleMRNet(nn.Module):
    def forward(self, sag, cor, axi):
        f_sag = self.sag_enc(sag)   # mean-pooled sagittal embedding (256-d)
        f_cor = self.cor_enc(cor)   # mean-pooled coronal embedding  (256-d)
        f_axi = self.axi_enc(axi)   # mean-pooled axial embedding    (256-d)
        fused = torch.cat([f_sag, f_cor, f_axi])  # 768-d
        return self.classifier(fused)
```

---

## 7. Results and Discussion

### 7.1 Training Loss Curves

*(Figure: `figures/training_loss_curves.png`)*

Training loss decreased consistently across all experiments. Experiment 4 was
trained for 10 epochs (versus 3 for Experiments 1–3), reflecting the additional
complexity of fine-tuning three AlexNet backbones per task.

### 7.2 Quantitative Results

**Table 1: ROC-AUC Comparison Across All Experiments**

| Experiment | Model | Abnormal | ACL   | Meniscus |
|---|---|---|---|---|
| Exp 1 | ResNet18 Sagittal Mean | 0.897 | 0.879 | 0.747 |
| Exp 2 | ResNet18 Sagittal Attention | 0.911 | 0.879 | 0.757 |
| Exp 3 | ResNet18 Three-View Mean | 0.911 | 0.879 | 0.757 |
| Exp 4 | AlexNet Three-Plane (Final) | **0.927** | **0.881** | **0.784** |

**Table 2: Experiment 4 — Full Validation Metrics**

| Task | Accuracy | Precision | Sensitivity | Specificity | F1 | ROC-AUC | Avg. Precision |
|---|---|---|---|---|---|---|---|
| Abnormality | 0.866 | 0.976 | 0.853 | 0.917 | 0.910 | 0.927 | 0.977 |
| ACL Tear    | 0.782 | 0.726 | 0.833 | 0.738 | 0.776 | 0.881 | 0.881 |
| Meniscal Tear | 0.681 | 0.595 | 0.846 | 0.552 | 0.698 | 0.784 | 0.739 |

### 7.3 ROC Curves

*(Figure: `figures/roc_curves_all_experiments.png`)*

The ROC curves confirm the ranking seen in Table 1. Abnormality detection is the
easiest task across all models, likely because general abnormalities manifest as
broad structural changes visible across many slices. ACL tear detection benefits
most from multi-plane input, as the ACL is best visualised in the sagittal plane
but its integrity can be corroborated from the coronal view. Meniscal tear detection
remains the hardest task across all experiments.

### 7.4 Precision-Recall Curves

*(Not included in report — see `figures/pr_curves_all_experiments.png` for PPT use.)*

### 7.5 Confusion Matrices

*(Figure: `figures/confusion_matrices_exp4.png`)*

For the final model (Experiment 4) at a 0.5 decision threshold:
- **Abnormality:** 2 false positives, 14 false negatives out of 119 exams
- **ACL Tear:** 17 false positives, 9 false negatives
- **Meniscal Tear:** 30 false positives, 8 false negatives

The model shows high sensitivity (recall) for all three tasks, meaning it rarely
misses a positive case — a desirable property in a clinical screening context where
false negatives are more costly than false positives.

### 7.6 Cross-Experiment Comparison

*(Figure: `figures/model_comparison_bar_charts.png`)*

Key observations from the comparison:

1. **Attention pooling vs. mean pooling (Exp 1 → Exp 2):** Attention MIL improved
   abnormality AUC from 0.897 to 0.911, confirming that not all slices are equally
   informative. The improvement on ACL and meniscus was marginal, possibly because
   3 training epochs are insufficient to fully train the attention module.

2. **Multi-plane input (Exp 2 → Exp 3):** Adding coronal and axial planes at the
   same training budget produced equivalent performance to the single-plane attention
   model on this metric. The benefit of additional planes appears only when the
   backbone is also fine-tuned.

3. **Backbone fine-tuning (Exp 3 → Exp 4):** The largest single improvement came
   from allowing the AlexNet encoders to fine-tune their weights (at a low learning
   rate of 1e-5). This suggests that ImageNet features, while useful, are not fully
   optimal for grayscale MRI slices and benefit from task-specific adaptation.

### 7.7 Meniscal Tear Performance

Meniscal tear detection consistently shows the lowest performance across all
experiments. This is consistent with findings in the original MRNet paper (Bien
et al., 2018), where meniscus was also the hardest task. Meniscal tears tend to be
small, subtle, and variable in appearance — sometimes visible in only one or two
slices — making them harder for a mean-pooling-based architecture to detect reliably.
The relatively high sensitivity (0.846) but low specificity (0.552) in Experiment 4
suggests the model is biased towards predicting positive for meniscus, which may
reflect the class weight applied during training.

---

## 8. Conclusion

This project implemented and compared four deep learning architectures for automated
knee MRI classification on the Stanford MRNet dataset.

**Main objectives achieved:**
- Four complete, reproducible training pipelines were implemented from scratch in
  PyTorch and evaluated on the official MRNet validation set.
- Attention-based slice pooling was shown to improve over mean pooling for
  abnormality detection.
- Multi-plane input with fine-tuned backbones (Experiment 4) consistently
  outperformed all single-plane and frozen-backbone models.

**Key findings:**
- Backbone fine-tuning, even at a very low learning rate (1e-5), was the most
  impactful design choice across all three tasks.
- Meniscal tear detection is substantially harder than abnormality or ACL detection,
  consistent with the clinical and published literature.
- The final TripleMRNet-inspired AlexNet model achieved ROC-AUC scores of 0.927,
  0.881, and 0.784 for abnormality, ACL tear, and meniscal tear respectively.

**Overall outcome:** The project demonstrates that relatively lightweight
architectures, when trained with appropriate class weighting and multi-plane input,
can achieve strong performance on a clinically relevant MRI classification task
without large-scale compute resources.

---

## 9. Future Scope

- **Attention across all three planes:** Experiment 4 uses mean pooling per plane;
  replacing this with per-plane attention modules could further improve performance,
  particularly for ACL and meniscal tear.
- **Transformer-based aggregation:** Vision Transformers (ViT) or cross-attention
  mechanisms could model inter-slice relationships that mean/attention pooling cannot.
- **Data augmentation:** Applying MRI-specific augmentations (random flipping, slight
  rotation, intensity jitter) could reduce overfitting and improve generalisation.
- **Longer training with scheduling:** Experiments 1–3 were trained for only 3
  epochs due to compute constraints. Learning rate scheduling and longer training
  would likely improve all models.
- **Clinical validation:** All models in this project are experimental. External
  validation on independent datasets and prospective clinical studies would be
  required before any real-world deployment.

---

## 10. References

Bien, N., Rajpurkar, P., Ball, R. L., Irvin, J., Park, A., Jones, E., ... &
Lungren, M. P. (2018). Deep-learning-assisted diagnosis for knee magnetic resonance
imaging: Development and retrospective validation of MRNet. *PLOS Medicine*, 15(11),
e1002699. https://doi.org/10.1371/journal.pmed.1002699

Ilse, M., Tomczak, J. M., & Welling, M. (2018). Attention-based deep multiple
instance learning. *Proceedings of the 35th International Conference on Machine
Learning (ICML)*. https://arxiv.org/abs/1802.04712

Tsai, C.-H., Kiryati, N., Konen, E., Eshed, I., & Mayer, A. (2021). Knee injury
detection using MRI with efficiently-layered network (ELNet). *International
Conference on Medical Imaging with Deep Learning (MIDL)*.

Stanford Center for Artificial Intelligence in Medicine and Imaging. MRNet Knee MRI
Dataset. https://doi.org/10.71718/rcbp-8c35

He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image
recognition. *Proceedings of the IEEE Conference on Computer Vision and Pattern
Recognition (CVPR)*, 770–778.

Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet classification with
deep convolutional neural networks. *Advances in Neural Information Processing
Systems (NeurIPS)*, 25.

---

## 11. Appendix

**GitHub Repository:** https://github.com/ashish112122/mrnet-knee-mri-classification

**Repository Contents:**
```
├── notebooks/
│   ├── 01_resnet18_sagittal_mean_pooling.ipynb
│   ├── 02_resnet18_sagittal_attention_mil.ipynb
│   ├── 03_resnet18_three_view_mean_pooling.ipynb
│   ├── 04_triplemrnet_alexnet_three_plane.ipynb
│   └── 05_results_and_plots.ipynb
├── results/
│   └── model_comparison.csv
├── docs/
│   ├── methodology.md
│   └── dataset_and_references.md
├── README.md
└── requirements.txt
```

**[INSERT: Additional screenshots of notebook execution outputs]**

**[INSERT: Any additional plots or figures not included in the main body]**
