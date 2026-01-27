# Industrial Surface Defect Detection via Bayesian-Optimized YOLOv8

## 📝 Abstract
This research presents a high-precision, automated system for detecting and categorizing six distinct types of industrial metal surface defects: **crazing, inclusion, patches, pitted surface, rolled-in scale, and scratches**. By integrating the **YOLOv8** architecture with a **Bayesian Optimization** framework (via Optuna), this project achieves a **mAP@50 of 0.985** and a **mAP@50-95 of 0.827**.

---

## 📂 Project Structure & Data Access

The primary dataset (5.3GB) is hosted externally due to high-resolution data requirements.

* **Google Drive:** [🔗 Click here to access Full Dataset, Models, and Optuna Logs](https://drive.google.com/drive/folders/1eKstN2mfJiM5ZFBpYg0UdJlgeEeezo3L?usp=sharing)

### Local Directory Layout
```text
.
├── src/                        # Inference scripts and core logic
├── models/                     # Production-ready weights (.pt)
├── optuna_results_focused/     # SQLite database and 20+ trial folders
│   └── trial_20_phase2/        # Global Optimum (mAP: 0.985)
├── runs/                       # Detailed training logs and validation batches
├── thesis/                     # MSc Dissertation (PDF and LaTeX source)
├── CITATION.cff                # Academic citation metadata
└── pyproject.toml              # Dependency configuration via uv
```
---
🚀 Reproduction Guide
This project utilizes uv for lightning-fast dependency management.

1. Environment Setup

```bash
git clone [https://github.com/Sandeepsanyare/automated-visual-defect-detection-yolo.git](https://github.com/Sandeepsanyare/automated-visual-defect-detection-yolo.git)
cd automated-visual-defect-detection-yolo
uv sync
```
2. Run Inference

To execute detection using the best-performing model (Trial 20):

```bash
python src/main.py --weights models/best.pt --source data/test_images/
```

📊 Results Summary
The system demonstrates robust performance across variable lighting conditions and complex textures:
Top Performing Trial: Trial 20 (Phase 2)

mAP@50 Metric: 0.985

mAP@50-95 Metric: 0.827

🎓 Author Information
Author: Sandeep Kumar

Institution: Arden University

Student ID: stu230944

Email: stu230944@ardenuniversity.ac.uk

If you find this research useful for your work, please use the "Cite this repository" button in the sidebar.
