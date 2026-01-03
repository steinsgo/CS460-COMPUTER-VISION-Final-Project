# 🖥️ ComputerVision Final
## 🌦️ Global Context-guided Multi-Degradation UHD Image Restoration

This repository contains our **Computer Vision final project**, focusing on **ultra-high-resolution (UHD, 4K) image restoration under multiple degradations**.  
We reproduce an instance-level routing restoration framework and introduce a lightweight reliability enhancement (**Safety Net**) to improve robustness under ambiguous degradations.

---

## 👥 Team Members

- **Benhuang Liu**  
  📧 1220004875@student.must.edu.mo

- **Bowen Xu†**  
  📧 1220012282@student.must.edu.mo

- **Chenyu Li†**  
  📧 1220012551@student.must.edu.mo

† Contributing authors

---

## 📌 Project Overview

Restoring UHD images captured under adverse conditions is challenging because different degradations require conflicting priors.

- 🌫️ **Dehazing** emphasizes low-frequency color and depth correction  
- 🌧️ **Deraining** / ❄️ **Desnowing** require high-frequency, edge-aware removal  
- 💨 **Deblurring** focuses on structure and motion recovery  

Instead of using patch-level or token-level Mixture-of-Experts (MoE), which is computationally expensive at 4K resolution, this project adopts an **instance-level (image-level) routing strategy**: one routing decision per image, keeping inference close to a single-path cost.

---

## 🧠 Method Summary

### 🔀 Instance-level Routing
A lightweight router analyzes global image context and predicts the most suitable degradation-specific route for each input image.

### 🧩 Latent Encoder–Decoder Backbone
We reproduce a VAE-style encoder–decoder that performs restoration primarily in latent space, improving efficiency for UHD images.

### 🛡️ Safety Net (Our Extension)
To improve reliability, we introduce a **confidence-based abstention mechanism**.  
When router confidence is below a threshold **τ**, the model falls back to a unified dense path, preventing severe degradation caused by incorrect routing.

### 🧱 UHD Tiled Inference
To handle 4K images under memory constraints, tiled inference with overlap-add blending is used to reduce visible seams.

---

## 🗂️ Tasks & Datasets

We evaluate four UHD degradation tasks:

- ❄️ **UHD-Snow** — Desnowing  
- 🌧️ **UHD-Rain** — Deraining  
- 🌫️ **UHD-Haze** — Dehazing  
- 💨 **UHD-Blur** — Deblurring  

Dataset statistics used in this project:

| Dataset   | Training Samples | Testing Samples | Task       |
|-----------|------------------|-----------------|------------|
| UHD-Snow  | 2000             | 200             | Desnowing  |
| UHD-Rain  | 2000             | 500             | Deraining  |
| UHD-Haze  | 2290             | 231             | Dehazing   |
| UHD-Blur  | 1964             | 300             | Deblurring |

---

## ⚙️ Environment Setup

```bash
conda create -n uhd_cv python=3.8 -y
conda activate uhd_cv
pip install -r requirements.txt

```

## 🏋️ Training

This project follows a **reproduction-oriented training setup**.

### Stage 1: Backbone / VAE Pre-training
The encoder–decoder backbone is first trained to learn compact and stable latent representations from clean images.

### Stage 2: Unified Restoration Training
The unified restoration model is trained on multiple degradation datasets with balanced task sampling. The instance-level router and degradation-aware components are jointly optimized.

For this course project, we focus on reproducing the training protocol and validating inference behavior using provided configurations and pretrained weights.

Example training commands (if scripts are available):

```bash
python basicsr/train_cls.py --config weight/six_deg/Cls_6d.yml
python basicsr/train.py --config weight/six_deg/VAE_6d.yml
python basicsr/train_fgt.py --config weight/six_deg/train_6d.yml
```

---

## ▶️ Inference

Single-image inference example:

```bash
python inference.py \
  -w weight/six_deg/6d_UHDprocessor.pth \
  -c weight/six_deg/infer_6d.yml \
  -i /path/to/input_image.jpg \
  -o results/<task_name>
```
---
## ✅ Conclusion

In this final project, we reproduced an instance-level routing framework for
ultra-high-resolution image restoration under multiple degradations.
By introducing a simple **Safety Net** mechanism, we improved robustness against
routing uncertainty without increasing inference complexity.

This project provided practical experience with UHD-scale vision systems,
evaluation protocols, and reliability considerations in modern image restoration
pipelines.

