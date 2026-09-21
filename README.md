# InsuLens - Explainable AI for Diabetic Retinopathy Screening

An intelligent, hybrid AI screening pipeline for early detection of Diabetic Retinopathy (DR) in rural India, built in MATLAB.

---

## 🎥 Demonstration Video

A complete walkthrough video of the system executing on retinal fundus scans with Grad-CAM heatmaps, lesion segmentation, and clinical fail-safe report generation is included:
- **`demo_video.mp4`** (included in repository root)

---

## 💡 What It Does

InsuLens analyzes retinal fundus images using a **Hybrid AI Architecture** that combines a Deep Learning CNN (ResNet-50) with classical algorithmic lesion detection, featuring a built-in **Fail-Safe Switch** for clinical safety.

### Key Features
- **Quality Assessment** — Automatically rejects ungradable images (blur, poor lighting, underexposure)
- **CLAHE Preprocessing** — Normalizes lighting, contrast, and color balance across heterogeneous camera sources
- **Retinal Segmentation** — Locates optic disc and segments blood vessels using morphological mathematical operators
- **Lesion Detection** — Detects Microaneurysms (MA), Hemorrhages (HE), and Hard Exudates (EX)
- **Deep Learning Classification** — ResNet-50 trained via Transfer Learning (**83.90% validation accuracy** across all 5 DR severity classes)
- **Hybrid Fail-Safe Switch** — Automatically cross-verifies AI confidence against classical mathematical algorithms; falls back to classical code when AI confidence < 65%
- **Discordance Flagging** — Detects and flags disagreements between AI and classical algorithms for mandatory ophthalmologist review
- **Explainable AI (XAI)** — Generates Grad-CAM visual attention heatmaps and annotated lesion overlays so clinicians can see *why* a decision was reached

---

## 🚀 How to Run

1. Open **MATLAB** (R2024a or later recommended)
2. Ensure **Deep Learning Toolbox** and **Image Processing Toolbox** are installed
3. In MATLAB, navigate to `DR_Project/`
4. Run `runHybridPipeline.m` — a file picker dialog will open, allowing you to select any fundus image to screen
5. The pipeline will automatically run quality assessment, preprocessing, vessel segmentation, lesion detection, AI classification, Grad-CAM heatmap generation, and display the comprehensive clinical dashboard!

---

## 📁 Repository Structure

```
insulens/
├── demo_video.mp4               — Full video demonstration of the screening pipeline
├── README.md                    — Project documentation & setup guide
├── .gitignore                   — Git exclusions
└── DR_Project/
    ├── runHybridPipeline.m      — Main Hybrid AI + Classical screening pipeline (RUN THIS)
    ├── test_pipeline.m          — Classical-only diagnostic pipeline
    ├── quality_assessment/      — Image quality scoring & ungradable rejection
    ├── preprocessing/           — Green-channel extraction & CLAHE enhancement
    ├── segmentation/            — Optic disc localization & blood vessel segmentation
    ├── lesion_detection/        — Microaneurysms, Hemorrhages, and Exudates detectors
    ├── classification/          — Classical rule-based ICDR grading engine
    ├── explainability/          — Grad-CAM heatmaps & clinical report generator
    ├── ai_modules/              — Deep learning inference & fail-safe arbitrator
    ├── ai_training/             — ResNet-50 transfer learning & training scripts
    │   └── trained_DR_ResNet.mat — Pre-trained ResNet-50 weights (83.9% validation accuracy)
    ├── datasets/
    │   └── APTOS_Sample/        — Sample fundus images for each DR grade (0 to 4)
    └── raw_images/
        └── A. Segmentation/     — IDRiD benchmark dataset with ground-truth lesion masks
```

---

## 📊 Dataset Information

- **Sample Test Images**: Bundled under `DR_Project/datasets/APTOS_Sample/` (sample images across Grade 0 to Grade 4) and `DR_Project/raw_images/A. Segmentation/` (IDRiD benchmark with lesion ground truths).
- **Full Kaggle APTOS 2019 Dataset**: To retrain the CNN on the full 3,662 high-resolution dataset, download it from [Kaggle APTOS 2019 Blindness Detection](https://www.kaggle.com/c/aptos2019-blindness-detection/data) and place images in `DR_Project/datasets/APTOS_Train/`.

---

## 💻 Hardware Requirements & Edge AI

- **OS**: macOS (Apple Silicon M-series optimized), Windows 10/11, or Linux
- **RAM**: 8 GB minimum (16 GB recommended)
- **Offline Capable**: 100% Edge AI execution — zero cloud dependency, ideal for rural camps and mobile screening units
- **Compatible Cameras**: Remidio NM-FOP, Forus 3nethra, smartphone ophthalmoscope attachments (e.g. MII Ret Cam, Volk iNview)

---

## ⚖️ Clinical Disclaimer

This software is an engineering research prototype designed for **early-risk screening and triage assistance only**. It does not constitute a definitive medical diagnosis. All screening results must be evaluated and verified by a licensed ophthalmologist or healthcare professional.
