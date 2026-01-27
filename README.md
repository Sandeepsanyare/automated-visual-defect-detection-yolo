Industrial Surface Defect Detection via Bayesian-Optimized YOLOv8


Abstract
This research presents a high-precision, automated system for detecting and categorizing six distinct types of industrial metal surface defects: crazing, inclusion, patches, pitted surface, rolled-in scale, and scratches. By integrating the YOLOv8 architecture with a Bayesian Optimization framework (via Optuna), this project circumvents the limitations of manual hyperparameter tuning. The implementation utilizes a customized sliding window tiling technique to process high-resolution industrial imagery while preserving pixel-level detail, ultimately achieving a mAP@50 of 0.985 and a mAP@50-95 of 0.827.

Project Structure & Data Access
Due to the high-resolution nature of the dataset and the extensive weights generated during 95+ training epochs, the primary data is hosted externally to maintain repository efficiency.

Google Drive Link: Full Dataset, Models, and Optuna Logs

Local Repository Layout:

/src: Core implementation logic.

/models: Exported .pt weights for the top-performing trials.

/optuna_results_focused: SQLite databases and visualizations for hyperparameter studies.

/runs: Detailed logs of training iterations.

/thesis: Comprehensive dissertation documentation.

Key Methodologies
1. Model Architecture

Utilized YOLOv8 (You Only Look Once v8) for real-time object detection, chosen for its superior trade-off between computational efficiency and mean Average Precision (mAP).

2. Bayesian Optimization (Optuna)

Implemented an automated search space for hyperparameters including learning rate, momentum, and augmentation coefficients. Trial 20 was identified as the global optimum within the defined constraints.

3. Tiling & Image Processing

To address the "small object" detection problem in large industrial images, a sliding window tiling technique was applied, ensuring that morphological characteristics of fine cracks and hair-like defects are not lost during downsampling.

🛠️ Installation & Usage
This project uses uv for lightning-fast dependency management.

Bash
# Clone the repository
git clone https://github.com/Sandeepsanyare/automated-visual-defect-detection-yolo.git
cd automated-visual-defect-detection-yolo

# Install dependencies using uv
uv sync

# Run inference with the best model (Trial 20)
python src/main.py --weights models/best.pt --source data/test_images/
Results Summary
The system demonstrates robust performance across variable lighting conditions and complex textures:

Top Performing Trial: Trial 20.

mAP@50: 0.985.

mAP@50-95: 0.827.

Dissertation
The full master's dissertation, titled "Industrial Surface Defect Detection via Bayesian-Optimized YOLOv8," is available in the /thesis directory. It provides an in-depth literature review of traditional CV methods vs. deep learning, the evolution of the YOLO architecture, and a discussion on Zero Defect Manufacturing (ZDM).


---
## 🎓 Author Information

**Author:** Sandeep Kumar  
**Institution:** [Arden University](https://arden.ac.uk/)  
**Student ID:** stu230944  
**Email:** [stu230944@ardenuniversity.ac.uk](mailto:stu230944@ardenuniversity.ac.uk)

---
*If you find this research useful for your work, please use the "Cite this repository" button in the sidebar.*

