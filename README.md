# Project Overview

Machine learning classification of three rock types from Raman spectroscopy data
acquired on a conveyor belt system at two belt speeds.

**Rocks classified:**
- S10Granite
- Holstein Sandstone
- Leitendorf Limestone

**Belt speeds:**
- 1.83 Hz
- 5.10 Hz

**Two modelling approaches were explored:**
1. **1D CNN** on raw spectral intensity profiles (1060 wavenumber points per sample)
2. **ResNet Transfer Learning** on 2D spectral images (224×224 px, converted from BMP)

---

## Repository Structure

```
/
├── README.md
                         MAIN NOTEBOOKS

├── 1d_cnn_per_speed_csv.ipynb
├── 1d_cnn_combined_speed_csv.ipynb
├── 1d_cnn_raw_txt.ipynb
├── 1d_cnn_als_correction_and_binary_cl.ipynb
├── 1d_cnn_speed_regime_comparison.ipynb
├── resnet18_baseline.ipynb
├── resnet_full_evaluation.ipynb
├── split_fraction_study.ipynb
├── rock_csvs
├── rock_txts
├── rocks_spectral_224

                     AUTO-GENERATED ON RUN

├── results_1d_cnn_per_speed_csv/
├── results_1d_cnn_combined_speed_csv/
├── results_1d_cnn_raw_txt/
├── results_1d_cnn_als_correction_and_binary_cl/
├── results_1d_cnn_speed_regime_comparison/
├── results_resnet18_baseline/
├── results_resnet_full_evaluation/
└── results_split_fraction_study/
```

Each results folder is created automatically when the corresponding notebook runs.

---

## Notebook Descriptions

### 1. `1d_cnn_per_speed_csv.ipynb`
**1D CNN on spectral profiles, separate speeds, CSV input**

The main 1D CNN notebook. Loads pre-processed spectral profiles from CSV files
(one row = one valid Raman profile, 1060 intensity values).
Trains and evaluates separately on 1.83 Hz and 5.10 Hz data.

**Architecture:** 3 convolutional layers (Conv1d) with MaxPooling, Dropout,
LayerNorm, and a final Linear(64→3) classification head. ~17,500 parameters.

**Key findings:**
- Best accuracy: ~90.5% on 1.83 Hz, ~88.6% on 5.10 Hz
- The 3rd conv layer was the decisive architectural improvement (from ~75% to ~90%)
- Main confusion: Sandstone/Limestone (spectral overlap)
- Granite nearly perfect (~95%) due to strong fluorescence signature

**Input:** `rock_csvs/`

**Output folder:** `results_1d_cnn_per_speed_csv/`

---

### 2. `1d_cnn_combined_speed_csv.ipynb`
**1D CNN: both speeds merged into a single dataset**

Same architecture as notebook 1, but both 1.83 Hz and 5.10 Hz data are
concatenated into one dataset before splitting. Tests whether a single model
can generalise across both belt speeds.

**Key finding:** Combined training achieves ~90% accuracy, comparable to
per-speed models, suggesting the spectral features are speed-invariant.

**Input:** `rock_csvs/` (both speeds)

**Output folder:** `results_1d_cnn_combined_speed_csv/`

---

### 3. `1d_cnn_raw_txt.ipynb`
**1D CNN: reads raw TXT files directly (no pre-processing step)**

Identical model to notebook 1, but includes a custom TXT parser that reads
the raw acquisition files directly. Handles the specific file format:
- Skips header lines (`C:\...`)
- Skips index and wavelength rows
- Filters invalid profiles (all-zero rows, spike rows with values ≥ 3.0)
- Converts comma decimal separator (`+0,33` format) to float

Useful for testing directly on new raw data without a CSV conversion step.

**Input:** `rocks_txts/`

**Output folder:** `results_1d_cnn_raw_txt/`

---

### 4. `1d_cnn_als_correction_and_binary_cl.ipynb`
**ALS baseline correction experiment + binary Sandstone vs Limestone classifier**

Two distinct experiments in one notebook:

**Part 1) ALS Baseline Correction:**
Applies Asymmetric Least Squares (ALS) baseline correction to Granite spectra
to remove the fluorescence background hump, then retrains the 3-class classifier.
Result: no improvement in accuracy (~90%). Conclusion: the model was already
classifying Granite correctly using the fluorescence shape as a feature.

**Part 2) Binary Classifier (Sandstone vs Limestone only):**
Isolates the hardest sub-problem: distinguishing Sandstone from Limestone, which
share similar Raman peak positions (quartz vs calcite).
Result: ~90% binary accuracy. Confirms that the Sandstone/Limestone confusion
is a genuine physical spectral overlap, not a modelling failure.

**Input:** `rock_csvs/`

**Output folder:** `results_1d_cnn_als_correction_and_binary_cl/`

---

### 5. `1d_cnn_speed_regime_comparison.ipynb`
**Full 1D CNN comparison: 1.83 Hz vs 5.10 Hz vs Combined**

The most comprehensive 1D CNN notebook. Trains three separate models
(1.83 Hz only, 5.10 Hz only, both speeds combined) and presents a
side-by-side comparison of all results including loss curves, accuracy
curves, and confusion matrices.

**Input:** `rock_csvs/`

**Output folder:** `results_1d_cnn_speed_regime_comparison/`

---

### 6. `resnet18_baseline.ipynb`
**ResNet-18 on 2D spectral images: baseline version**

The initial ResNet-18 transfer learning notebook. Loads 224×224 JPG images
(converted from raw BMP spectral images using PIL.Image.LANCZOS resize).
Trains with ImageNet pretrained weights, replacing the final fc layer with
Dropout(0.3) + Linear(512→3).

Basic functionality: training loop, confusion matrix, and TensorBoard logging.
No advanced evaluation features.

**Best result:** ~99.63–99.67% on 1.83 Hz.

**Input:** `rocks_spectral_224/`

**Output folder:** `results_resnet18_baseline/`

---

### 7. `resnet_full_evaluation.ipynb`
**ResNet-18 full evaluation pipeline with all experiments**

The definitive and most complete ResNet notebook. Extends notebook 6 with a full
suite of experiments and evaluation tools. Supports ResNet-18, ResNet-34,
ResNet-50, and EfficientNet-B0 via a unified `build_model()` function.

**All features:**

| Cell | Feature |
|------|---------|
| Config cell | Single cell to control ARCH, MAX_PER_CLASS, FREEZE_BACKBONE, USE_STRONG_AUG, TEST_SPLIT |
| LR Finder | Sweeps learning rate from 1e-7 to 1.0, plots loss curve, suggests optimal LR |
| Training | 4 hyperparameter combos (2 LRs × 2 WDs), 20 epochs, warmup + ReduceLROnPlateau |
| F1/Precision/Recall | Per-class classification report saved as .txt after each experiment |
| Confusion matrices | Saved during training and post-run for all experiments |
| Training curves | Loss and accuracy plots for all experiments |
| Architecture comparison | ResNet-18/34/50 + EfficientNet-B0 trained for 10 epochs each |
| K-fold CV | 5-fold StratifiedKFold, reports mean ± std accuracy |
| Grad-CAM | Heatmap overlay showing which spectral regions drive each classification |
| Misclassification + Grad-CAM | Shows only wrongly classified images with CAM overlay |
| Cross-speed Grad-CAM | Same rock class at 1.83 Hz vs 5.10 Hz side by side |
| t-SNE | 2D projection of the feature space before the classifier head |
| TTA | 5-augmentation test-time averaging, compares baseline vs TTA accuracy |

**Final results:**

| Speed | Best Accuracy | K-fold Mean | K-fold Std |
|-------|--------------|-------------|------------|
| 1.83 Hz | 99.59% | 99.48% | ±0.13% |
| 5.10 Hz | 99.57% | 98.15% | ±2.90% |

**Best config (both speeds):** `batch_size=64, lr=0.0001, weight_decay=1e-4, epochs=20`

**Architecture winner:** ResNet-18 outperforms all larger models
(ResNet-34: 99.38%, ResNet-50: 99.29%, EfficientNet-B0: 98.84%)

**Input:** `rocks_spectral_224/`

**Output folder:** `results_resnet_full_evaluation/`

---

### 8. `split_fraction_study.ipynb`
**ResNet-18 data efficiency and train/test split sensitivity study**

Runs a 5 × 9 grid sweep** (5 data fractions × 9 test-split sizes) on both belt speeds = 90 independent training runs, each a full ResNet-18 trained for 15 epochs with the best fixed hyperparameters from notebook 7. All runs share a single seed.
Results are saved per-run to `results_all_runs.csv` and as labelled plots to three
sub-folders.

**Sweep parameters:**

| Axis | Values |
|------|--------|
| Data fraction (% of all images used before splitting) | 20%, 40%, 60%, 80%, 100% |
| Test-split size (% of sub-sampled data reserved as test set) | 10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, 90% |
---

#### Block 1 — Overall Accuracy Results

| Metric | 1.83 Hz | 5.10 Hz |
|--------|---------|---------|
| Max accuracy (best config) | 99.7% | 99.8% |
| Mean accuracy (all 36 configs) | 96.6% | 97.5% |
| Min accuracy (worst config) | 88.0% | 89.6% |
| Best config | data=100%, test-split=10% | data=40%, test-split=10% |
| Worst config | data=20%, test-split=90% | data=20%, test-split=90% |

---

**Training samples required vs. accuracy (1.83 Hz):**

| n_train | Config | Accuracy |
|---------|--------|----------|
| 241 | 20% data, 90% test | 88.0% |
| 483 | 20% data, 80% test | 90.7% |
| 724 | 20% data, 70% test | 91.6% |
| 966 | 20% data, 60% test | 93.4% |
| 2,174 | 20% data, 10% test | 99.6% |
| 4,348 | 40% data, 10% test | 99.6% |
| 8,697 | 80% data, 10% test | 99.6% |
| 10,870 | 100% data, 10% test | 99.7% |


---

#### Block 2 — Per-Rock Recall Results

| Rock | Mean recall — 1.83 Hz | Mean recall — 5.10 Hz |
|------|----------------------|----------------------|
| S10Granite | 98.7% | 98.6% |
| Holstein Sandstone | 96.1% | 97.0% |
| Leitendorf Limestone | 94.9% | 96.8% |

---

#### Final Comparison — 1.83 Hz vs 5.10 Hz

---

**FIN-01 — Full heatmaps: overall accuracy + per-rock recall side by side**

---

**FIN-02 — Accuracy difference heatmap: 1.83 Hz minus 5.10 Hz**

---

**FIN-03 — Summary bar charts: accuracy range and mean per-rock recall**

---

**FIN-04 — Normalised confusion matrices: best and worst configs on both datasets**

---

**FIN-05 — Per-rock recall difference: 1.83 Hz minus 5.10 Hz**

---

**Input:** `rocks_spectral_224/`

**Output folder:** `results_split_fraction_study/`

---

## Key Findings Summary

| Finding | Detail |
|---------|--------|
| Best model | ResNet-18, 99.7% (1.83 Hz), 99.8% (5.10 Hz) |
| 1D CNN ceiling | ~90%, limited by Sandstone/Limestone spectral overlap |
| ResNet-18 advantage | ~10 pp absolute improvement over 1D CNN |
| Granite | Near-perfect in all models and all data regimes: fluorescence hump is highly discriminative |
| Main error source | Sandstone/Limestone confusion; physically unavoidable spectral similarity between quartz and calcite peaks |
| Overfitting | Not observed: training/test curves remain parallel; K-fold ±0.13% (1.83 Hz) |
| 5.10 Hz instability | Dataset-intrinsic: reconfirmed by seed testing (notebook 7) and replicated in split study (notebook 8) |
| Grad-CAM | Model focuses on fluorescence hump (Granite) and Raman peak region (Sandstone/Limestone) |
| t-SNE | Three clearly separated clusters: model learned genuine spectral representations |
| Data efficiency | ResNet-18 is not data-limited: ~2,200 training images suffice for ≥99.6% accuracy. Using the full dataset (~10,870 images) yields a marginal gain to 99.7% |
| Minimum viable dataset | ~240 training images → 88% accuracy; ~500 → 90%; ~1,000 → 93%; ~2,200 → 99.6% |
| Overall accuracy is optimistic | A config reporting ~93% overall can have Leitendorf recall as low as 84–88%; per-rock recall is the correct deployment metric |
| Error mode under scarcity | Leitendorf → Holstein is the dominant failure (16–17% misclassification rate at worst config); Granite also begins to bleed at very low training counts |

---

## Data Paths (Modify according to your path for each test)

| Data type | Path |
|-----------|------|
| Raw TXT files | `rocks_txts/` |
| CSV profiles | `rock_csvs/` |
| Resized JPG images (224×224) | `rocks_spectral_224/` |
