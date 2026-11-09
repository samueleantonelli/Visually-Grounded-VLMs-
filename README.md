# Visually-Grounded-VLMs

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.2+](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper: Phil. Trans. R. Soc. B](https://img.shields.io/badge/Paper-RSTB--2026-green.svg)](https://royalsocietypublishing.org/journal/rstb)

Official repository for the paper:
> **Interactive Abilities of Visually Grounded Conversational Models in the LLM Era**  
> *Samuele Antonelli, Raffaella Bernardi, Raquel Fernández*  
> Philosophical Transactions of the Royal Society B (RSTB), 2026.

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Running Experiments](#running-experiments)
  - [1. Behavioral Evaluation (REC)](#1-behavioral-evaluation-rec)
  - [2. Representational Probing](#2-representational-probing)
  - [3. Statistical & Linguistic Analyses](#3-statistical--linguistic-analyses)
- [Main Experimental Results](#main-experimental-results)
- [Citation](#citation)
- [License](#license)

---

## Overview

Human conversation is a joint activity where speakers coordinate through efficiency principles, balancing conciseness and informativeness (Grice's Maxim of Quantity, Clark's Principle of Least Collaborative Effort). While multimodal large language models (VLMs) demonstrate strong vision-language capabilities, their interactive coordination abilities remain poorly understood.

This repository provides an empirical evaluation framework combining two complementary paradigms:

1. **Behavioral Evaluation (Referring Expression Comprehension - REC):** Tests whether model listeners (principally **Qwen2.5-VL-7B**, alongside **CogVLM-17B** and **Grounding DINO-L**) benefit from human-like concise descriptions compared to verbose model-generated descriptions under varying visual clutter (0 to 4 same-category distractors).
2. **Representational Probing:** Diagnostic linear classifiers trained on hidden state activations across layers (focused on Layer 18) to determine whether pragmatic discriminativity is linearly encoded in latent representations, accompanied by:
   - **Text-Only Ablation:** Quantifying how much decodability stems from surface linguistic regularities rather than grounded multimodal reasoning.
   - **Random Labels Control:** Empirical baseline establishing chance performance ($AUROC \approx 0.50$).
   - **McNemar Discrepancy Test:** Evaluating whether visual representations play an asymmetric corrective role over unimodal predictions.

---

## Repository Structure

```text
Visually-Grounded-VLMs/
├── README.md
├── requirements.txt
├── data/
│   ├── splits/                  # Image-ID level disjoint partitions (train ~70%, val ~15%, test ~15%)
│   └── refoi/                   # RefOI benchmark annotations, boxes, and metadata
├── behavioral/
│   ├── eval_rec.py              # REC inference & IoU computation (0.50, 0.75, mean IoU)
│   └── eval_baselines.py        # Evaluators for CogVLM-17B & Grounding DINO-L
├── representational/
│   └── probe_pipeline.py        # Last-token extraction across layers & linear probe training
├── analysis/
│   ├── cluster_regression.py    # Trial-level logistic regression (Huber-White SE on image_id)
│   └── error_and_linguistic.py  # Lexical cue distribution & McNemar cross-modal test
└── scripts/
    └── run_all.sh               # End-to-end execution script
```

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/samueleantonelli/Visually-Grounded-VLMs.git
cd Visually-Grounded-VLMs
```

### 2. Set up environment
```bash
python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

> **Note:** For running Qwen2.5-VL-7B, ensure a CUDA GPU with appropriate VRAM and `flash_attention_2` support is available.

---

## Data Preparation

Experiments are based on the **RefOI benchmark** (Ma et al., 2025):

1. Download the images and annotations from RefOI.
2. Place raw images in `data/images/`.
3. Split files must be partitioned strictly by `image_id` to prevent visual data leakage:
   - **Training Set (~70%):** 9,175 pairs
   - **Validation Set (~15%):** 1,961 pairs
   - **Test Set (~15%):** 1,915 pairs (5,341 discriminative vs. 7,710 underdiscriminative total instances)

Format expected in `data/splits/test_rec.csv`:
```csv
image_id,image_file,speaker,condition,distractor_count,gold_box,description
1042,dog_scene_1.jpg,Human,Spoken,2,"[63, 184, 695, 907]","The black dog in the front."
```

---

## Running Experiments

### 1. Behavioral Evaluation (REC)

Prompt the listener model to output bounding box coordinates `[Xmin, Ymin, Xmax, Ymax]` and compute IoU against gold standards:

```bash
python behavioral/eval_rec.py \
    --data_path data/splits/test_rec.csv \
    --images_dir data/images/ \
    --model_id "Qwen/Qwen2.5-VL-7B-Instruct" \
    --output_path rec_results.csv
```

Outputs:
- Accuracy at $IoU \ge 0.50$
- Accuracy at $IoU \ge 0.75$
- Continuous metric: Mean IoU
- Breakdown stratified across 0 to 4 same-category distractors

---

### 2. Representational Probing

Extract last-token hidden states across sampled layers ($0, 2, 4, \dots, 26, 27$) without conversational prompt templates, and train a class-balanced linear classifier on discriminativity:

```bash
python representational/probe_pipeline.py \
    --data_csv data/splits/probing_dataset.csv \
    --images_dir data/images/ \
    --model_id "Qwen/Qwen2.5-VL-7B-Instruct"
```

This script evaluates:
1. **Multimodal Probing:** Hidden states extracted from (Image + Text).
2. **Text-Only Ablation:** Hidden states extracted without the image input.
3. **Random Label Control:** Permuted training labels to test chance level ($AUROC = 0.50$).

---

### 3. Statistical & Linguistic Analyses

#### Trial-level Clustered Logistic Regression
Fit logistic regression with Huber-White cluster-robust standard errors grouped by `image_id`:

```bash
python analysis/cluster_regression.py \
    --rec_results_csv rec_results.csv
```

#### Lexical Cues & McNemar Test
Quantify spatial terms, ordinals, definite articles, color terms, and sentence length, and compute the McNemar test ($\chi^2$) on cross-modal test discrepancies:

```bash
python analysis/error_and_linguistic.py \
    --test_results_csv test_probe_predictions.csv
```

---

## Main Experimental Results

### Listener Comprehension Accuracy ($IoU \ge 0.50$)

| Speaker | Condition | Human Listeners | Qwen2.5-VL-7B (Full) | Qwen2.5-VL-7B ($\ge 1$ Distractor) | CogVLM-17B | Grounding DINO-L |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Human** | Spoken | 92% | 88.31% | **85.35%** | 63% | 60% |
| **Human** | Written | 89% | 88.15% | **84.90%** | 60% | 60% |
| **Model** | Brief | 36% | 88.19% | 82.33% | 61% | 73% |
| **Model** | Default | 40% | 88.72% | 81.56% | 37% | 58% |

### Linear Probing at Layer 18 (Qwen2.5-VL-7B)

| Probing Condition | Accuracy | F1 Score | AUROC |
| :--- | :---: | :---: | :---: |
| **Multimodal (Image + Text)** | **80%** | **0.78** | **0.89** |
| **Unimodal (Text-Only Ablation)** | 74% | 0.73 | 0.84 |
| **Control (Random Labels)** | 49% | 0.43 | 0.50 |

### Key Takeaways

1. **Relative Insensitivity to Brevity:** Unlike human listeners who collapse when interpreting model descriptions (36-40%), Qwen2.5-VL-7B maintains high accuracy across all prompt types. However, when competitors are present ($\ge 1$ distractor), a statistically significant advantage for human descriptions emerges ($p = 0.006$).
2. **Text-Only Shortcut Learning:** Unimodal probing achieves an AUROC of 0.84 without visual input, showing that linear discriminativity is heavily driven by superficial textual cues (definite articles, spatial markers).
3. **Asymmetric Corrective Role of Vision:** In discrepant test cases ($N = 265$), adding visual features resolves ambiguity in 69.06% of trials, significantly outperforming cases where vision causes interference ($\chi^2 = 38.5, p < 0.001$).

---

## Citation

If you find this work or codebase useful, please cite our paper:

```bibtex
@article{antonelli2026interactive,
  title={Interactive Abilities of Visually Grounded Conversational Models in the LLM Era},
  author={Antonelli, Samuele and Bernardi, Raffaella and Fern{\'a}ndez, Raquel},
  journal={Philosophical Transactions of the Royal Society B},
  year={2026}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
