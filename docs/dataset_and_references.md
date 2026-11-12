# Dataset and References

## Stanford MRNet Dataset

This project uses the **MRNet Knee MRI Dataset** provided by the Stanford Center for Artificial Intelligence in Medicine and Imaging (AIMI).

- **Access:** https://aimi.stanford.edu/datasets/mrnet-knee-mris
- **Dataset DOI:** https://doi.org/10.71718/rcbp-8c35

### Contents

| Split | Exams |
|---|---|
| Training | 1,130 |
| Validation | 120 |
| **Total** | **1,370** |

Each exam contains three MRI volumes:
- **Sagittal plane**
- **Coronal plane**
- **Axial plane**

Three binary labels are provided per exam:
- `abnormal` — any knee abnormality present
- `acl` — ACL tear
- `meniscus` — meniscal tear

### Access and Redistribution

The dataset is provided under a **Dataset Research Use Agreement**. Users must register and accept the agreement before downloading.

**Do not upload the dataset, cached `.npy` volumes, or any dataset-derived files to public repositories.** Check the terms of your agreement before redistributing model checkpoints trained on this data.

---

## References

### Primary Paper

Bien, N., Rajpurkar, P., Ball, R. L., Irvin, J., Park, A., Jones, E., ... & Lungren, M. P. (2018). Deep-learning-assisted diagnosis for knee magnetic resonance imaging: Development and retrospective validation of MRNet. *PLOS Medicine*, 15(11), e1002699.  
https://doi.org/10.1371/journal.pmed.1002699

### Dataset Citation

Stanford Center for Artificial Intelligence in Medicine and Imaging. (n.d.). MRNet Knee MRI Dataset [Data set].  
https://doi.org/10.71718/rcbp-8c35

### TripleMRNet (Experiment 4 Inspiration)

Tsai, C.-H., Kiryati, N., Konen, E., Eshed, I., & Mayer, A. (2021). Knee injury detection using MRI with efficiently-layered network (ELNet). In *International Conference on Medical Imaging with Deep Learning (MIDL)*.

> Note: Experiment 4 is a *practical, pretrained-AlexNet-variant* inspired implementation, not a bit-for-bit reproduction of any single paper. The original TripleMRNet paper contains contradictions regarding ImageNet initialisation versus training from scratch; this project follows the pretrained variant.
