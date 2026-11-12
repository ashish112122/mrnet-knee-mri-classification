# Knee MRI Classification with Deep Learning (MRNet)

A deep learning project classifying knee MRI examinations across three diagnostic tasks using the Stanford MRNet dataset. Four progressively more complex architectures are explored, from a single-view mean pooling baseline to a three-plane AlexNet fusion model inspired by TripleMRNet.

---

## Tasks

Each model performs **binary classification** for three independent labels:

| Label | Description |
|---|---|
| `abnormal` | Presence of any knee abnormality |
| `acl` | ACL tear |
| `meniscus` | Meniscal tear |

---

## Dataset

**Stanford MRNet Knee MRI Dataset**

- 1,370 knee MRI examinations (1,130 train / 120 validation)
- Three MRI planes per exam: sagittal, coronal, axial
- Three binary labels: abnormality, ACL tear, meniscal tear

**The dataset is not included in this repository.** To reproduce these experiments, request access through the Stanford AIMI portal and accept the applicable research use agreement.

> Dataset access: https://aimi.stanford.edu/datasets/mrnet-knee-mris  
> Dataset DOI: https://doi.org/10.71718/rcbp-8c35

---

## Experiments

| # | Notebook | Architecture | Views | Pooling |
|---|---|---|---|---|
| 1 | `01_resnet18_sagittal_mean_pooling.ipynb` | Frozen ResNet18 | Sagittal | Mean |
| 2 | `02_resnet18_sagittal_attention_mil.ipynb` | Frozen ResNet18 + Attention MIL | Sagittal | Learned attention |
| 3 | `03_resnet18_three_view_mean_pooling.ipynb` | Frozen ResNet18 | Sagittal + Coronal + Axial | Mean per view, concat |
| 4 | `04_triplemrnet_alexnet_three_plane.ipynb` | 3× Fine-tuned AlexNet (TripleMRNet-inspired) | Sagittal + Coronal + Axial | Mean per view, concat |

The notebooks are numbered in the order they were developed. Each one builds on the insights of the previous, moving from a single-plane baseline through attention pooling and multi-view fusion.

---

## Results

Full per-label metrics are in [`results/model_comparison.csv`](results/model_comparison.csv).

### ROC-AUC Summary

| Model | Abnormal | ACL | Meniscus |
|---|---|---|---|
| ResNet18 Sagittal Mean (Exp 1) | 0.897 | 0.879 | 0.747 |
| ResNet18 Sagittal Attention (Exp 2) | 0.911 | 0.879 | 0.757 |
| ResNet18 Three-View Mean (Exp 3) | 0.911 | 0.879 | 0.757 |
| TripleMRNet AlexNet Three-Plane (Exp 4) | **0.927** | **0.881** | — |

> Experiment 4 trains three separate binary models (one per label) for 10 epochs with backbone fine-tuning, which explains the stronger performance.

---

## Methodology

All models follow the same general pattern:

1. Load pre-processed MRI volumes (`.npy` slices per plane)
2. Pass each slice through a CNN backbone to get per-slice embeddings
3. Aggregate slice embeddings into a single exam-level vector (mean or attention pooling)
4. Optionally concatenate embeddings from multiple planes
5. Apply a linear classifier and train with BCEWithLogitsLoss + positive-class weighting

See [`docs/methodology.md`](docs/methodology.md) for a detailed write-up.

---

## Setup

### Requirements

```bash
pip install -r requirements.txt
```

### Running the Notebooks

The notebooks are written for **Google Colab** with GPU acceleration (Tesla T4). They expect the MRNet data mounted from Google Drive:

```
MyDrive/mrnet/
├── train_exams.parquet
├── valid_exams.parquet
└── cache/
    ├── train/
    │   ├── sagittal/<exam_id>.npy
    │   ├── coronal/<exam_id>.npy
    │   └── axial/<exam_id>.npy
    └── valid/
        └── ...
```

1. Obtain the MRNet dataset from [Stanford AIMI](https://aimi.stanford.edu/datasets/mrnet-knee-mris)
2. Upload and cache the `.npy` volumes following the layout above
3. Open a notebook in Colab, mount Drive, and update `DRIVE_ROOT` if your path differs
4. Run all cells top to bottom

---

## References

Bien, N., Rajpurkar, P., Ball, R. L., Irvin, J., Park, A., Jones, E., ... & Lungren, M. P. (2018). Deep-learning-assisted diagnosis for knee magnetic resonance imaging: Development and retrospective validation of MRNet. *PLOS Medicine*, 15(11), e1002699.  
https://doi.org/10.1371/journal.pmed.1002699

Stanford MRNet Dataset DOI: https://doi.org/10.71718/rcbp-8c35

---

## Disclaimer

This is an academic research project. The models are experimental and are not suitable for clinical diagnosis or medical decision-making.
