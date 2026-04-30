# Rock Classification via Raman Spectroscopy — ML Pipeline

Machine learning classification of three rock types from Raman spectroscopy data
acquired on a conveyor belt system at two belt speeds.

**Rocks classified:**
- S10Granite
- Holstein Sandstone
- Leitendorf Limestone

**Belt speeds:**
- 1.83 Hz
- 5.10 Hz

**Three modelling approaches were explored:**
1. **1D CNN** on raw spectral intensity profiles (1060 wavenumber points per sample)
2. **ResNet-18 Transfer Learning** on 2D spectral images (224×224 px, converted from BMP)
3. **1D MLP Autoencoder** for out-of-distribution (OOD) detection on raw spectra

---

## Repository Structure

```
/
├── README.md

                    CLASSIFICATION NOTEBOOKS

├── 1d_cnn_per_speed_csv.ipynb                ← Main 1D CNN (per speed, CSV input)
├── 1d_cnn_combined_speed_csv.ipynb           ← 1D CNN both speeds merged
├── 1d_cnn_raw_txt.ipynb                      ← 1D CNN from raw TXT (no CSV step)
├── 1d_cnn_als_correction_and_binary_cl.ipynb ← ALS correction + binary classifier
├── 1d_cnn_speed_regime_comparison.ipynb      ← Full 1D CNN comparison
├── resnet18_baseline.ipynb                   ← ResNet-18 baseline
├── resnet_full_evaluation.ipynb              ← ResNet-18 full evaluation (MAIN)
├── split_fraction_study.ipynb                ← Data efficiency sensitivity study
├── inference_new_data.ipynb                  ← Inference on new rocks and baseline results
├── rock_classifier_combined.ipynb            ← Combined train+eval+OOD+inference
├── rock_classifier_multisource.ipynb         ← Multi-source training (generalisation fix)
├── cnn_1d_train_and_inference.ipynb          ← 1D CNN inference on new rock profiles
├── ood_autoencoder_1d.ipynb                  ← 1D autoencoder OOD detector

                         DATA FOLDERS

├── rock_csvs/                               ← Converted CSV profiles
├── rock_txts/                               ← Raw PROFILES txt
├── rocks_spectral_224/                      ← Resized 224x224 JPG images (training)
├── for_test_data_spectral_224/              ← Resized 224x224 JPG images (new test data)
├── profiles/                                ← Raw PROFILES txt for new rock folders (new test data)
└── profiles_csv/                            ← Converted CSV profiles for new rock folders (new test data)

                     AUTO-GENERATED ON RUN

├── results_1d_cnn_per_speed_csv/
├── results_1d_cnn_combined_speed_csv/
├── results_1d_cnn_raw_txt/
├── results_1d_cnn_als_correction_and_binary_cl/
├── results_1d_cnn_speed_regime_comparison/
├── results_resnet18_baseline/
├── results_resnet_full_evaluation/
├── results_split_fraction_study/
├── results_inference_new_data/
├── results_rock_classifier_combined/
├── results_rock_classifier_multisource/
├── results_1d_cnn_inference/
└── results_ood_autoencoder_1d/
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
- Filters invalid profiles (all-zero rows, spike rows with values >= 3.0)
- Converts comma decimal separator (`+0,33` format) to float

**Input:** `rocks_txts/`
**Output folder:** `results_1d_cnn_raw_txt/`

---

### 4. `1d_cnn_als_correction_and_binary_cl.ipynb`
**ALS baseline correction experiment + binary Sandstone vs Limestone classifier**

**Part 1 — ALS Baseline Correction:**
Applies Asymmetric Least Squares (ALS) baseline correction to Granite spectra
to remove the fluorescence background hump, then retrains the 3-class classifier.
Result: no improvement (~90%). The model was already using the fluorescence
shape as a feature so removing it loses information.

**Part 2 — Binary Classifier (Sandstone vs Limestone only):**
~90% binary accuracy. Confirms that the Sandstone/Limestone confusion is a genuine
physical spectral overlap (quartz vs calcite peak similarity) and not a modelling failure.

**Input:** `rock_csvs/`
**Output folder:** `results_1d_cnn_als_correction_and_binary_cl/`

---

### 5. `1d_cnn_speed_regime_comparison.ipynb`
**Full 1D CNN comparison: 1.83 Hz vs 5.10 Hz vs Combined**

Trains three separate models and presents a side-by-side comparison of all results
including loss curves, accuracy curves, and confusion matrices.

**Input:** `rock_csvs/`
**Output folder:** `results_1d_cnn_speed_regime_comparison/`

---

### 6. `resnet18_baseline.ipynb`
**ResNet-18 on 2D spectral images: baseline version**

The initial ResNet-18 transfer learning notebook. Loads 224x224 JPG images
converted from raw BMP spectral images using PIL LANCZOS resize.
Trains with ImageNet pretrained weights, replacing the final fc layer with
Dropout(0.3) + Linear(512->3).

**Best result:** ~99.63-99.67% on 1.83 Hz.

**Input:** `rocks_spectral_224/`
**Output folder:** `results_resnet18_baseline/`

---

### 7. `resnet_full_evaluation.ipynb` — MAIN RESNET NOTEBOOK
**ResNet-18 full evaluation pipeline with all experiments**

The definitive ResNet notebook. Supports ResNet-18/34/50 and EfficientNet-B0
via a unified `build_model()` function.

| Feature | Description |
|---------|-------------|
| Config cell | Controls ARCH, MAX_PER_CLASS, FREEZE_BACKBONE, USE_STRONG_AUG, TEST_SPLIT |
| LR Finder | Sweeps 1e-7 to 1.0, plots smoothed loss, suggests optimal LR |
| Training | 4 hyperparameter combos (2 LRs x 2 WDs), 20 epochs, warmup + ReduceLROnPlateau |
| F1/Precision/Recall | Per-class classification report saved as .txt |
| Confusion matrices | Raw + normalised, saved per experiment |
| Architecture comparison | ResNet-18/34/50 + EfficientNet-B0 trained for 10 epochs each |
| K-fold CV | 5-fold StratifiedKFold, reports mean +/- std accuracy |
| Grad-CAM | Heatmap showing which spectral regions drive each classification |
| t-SNE | 2D projection of feature space before the classifier head |
| TTA | 5-augmentation test-time averaging, compares baseline vs TTA accuracy |

**Final results:**

| Speed | Best Accuracy | K-fold Mean | K-fold Std |
|-------|--------------|-------------|------------|
| 1.83 Hz | 99.59% | 99.48% | +/-0.13% |
| 5.10 Hz | 99.57% | 98.15% | +/-2.90% |

**Best config:** `batch_size=64, lr=1e-4, weight_decay=1e-4, epochs=20`
**Architecture winner:** ResNet-18 outperforms ResNet-34 (99.38%), ResNet-50 (99.29%), EfficientNet-B0 (98.84%)

**Input:** `rocks_spectral_224/`
**Output folder:** `results_resnet_full_evaluation/`

---

### 8. `split_fraction_study.ipynb`
**ResNet-18 data efficiency and train/test split sensitivity study**

Runs a 5 x 9 grid sweep (5 data fractions x 9 test-split sizes) on both belt
speeds = 90 independent training runs, each a full ResNet-18 trained for 15 epochs
with the best fixed hyperparameters from notebook 7.

**Sweep parameters:**

| Axis | Values |
|------|--------|
| Data fraction (% of all images used before splitting) | 20%, 40%, 60%, 80%, 100% |
| Test-split size (% of sub-sampled data reserved as test set) | 10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, 90% |

**Key results:**

| Metric | 1.83 Hz | 5.10 Hz |
|--------|---------|---------|
| Max accuracy | 99.7% | 99.8% |
| Mean accuracy | 96.6% | 97.5% |
| Min accuracy | 88.0% | 89.6% |

**Training samples vs accuracy (1.83 Hz):**

| n_train | Accuracy |
|---------|----------|
| 241 | 88.0% |
| 483 | 90.7% |
| 2,174 | 99.6% |
| 10,870 | 99.7% |

**Key finding:** ResNet-18 is not data-limited. ~2,200 training images achieve
~99.6% accuracy. Using the full dataset adds only 0.1 pp.

**Per-rock mean recall:**

| Rock | 1.83 Hz | 5.10 Hz |
|------|---------|---------|
| S10Granite | 98.7% | 98.6% |
| Holstein Sandstone | 96.1% | 97.0% |
| Leitendorf Limestone | 94.9% | 96.8% |

**Input:** `rocks_spectral_224/`
**Output folder:** `results_split_fraction_study/`

---

### 9. `inference_new_data.ipynb`
**First inference on new rock folders — original single-source ResNet**

The first test of the trained ResNet-18 model on completely unseen rock folders.
Loads the saved `.pth` weights from `resnet_full_evaluation.ipynb` directly with
no retraining. Runs all 9 new rock folders through the model and records
predictions, confidence scores, and per-folder statistics.

This notebook established the baseline failure results that motivated all
subsequent work (multi-source training, OOD detection, 1D approaches).

**New test data used:** `for_test_data_spectral_224/` (all 9 new rock folders,
4,556 images total converted from BMP by `bmp_to_spectral_224.py`)

**Inference results (original single-source ResNet, no OOD detection):**

| Folder | Expected | Predicted | Mean conf | Verdict |
|--------|----------|-----------|-----------|---------|
| Dunite-Ecologite_2Rocks_1-83Hz | Unknown | Leitendorf | 97.5% | Wrong + overconfident |
| Gneis_1-83Hz | S10Granite | S10Granite | 96.5% | Correct |
| Granite_3SamplesPhilipp_1-83Hz_1 | S10Granite | Leitendorf | 99.8% | Incorrect |
| Granite_3SamplesPhilipp_1-83Hz_2 | S10Granite | Leitendorf | 99.7% | Incorrect |
| Limestone_CalcsilicaContaminated_U9_U3_1-83Hz | Leitendorf | Holstein | 98.6% | Incorrect |
| Limestone_CalcsilicaContaminated_U9_U3_5-10Hz | Leitendorf | Holstein | 97.4% | Incorrect |
| Limestone_Rax_1-83Hz_1 | Leitendorf | Leitendorf | 100.0% | Correct |
| Limestone_Rax_1-83Hz_2 | Leitendorf | Leitendorf | 99.6% | Correct |
| SandstoneNew_1-83Hz | Holstein | Leitendorf | 98.5% | Incorrect |

**Key findings:**
- 3 correct, 5 incorrect, 1 true OOD with no mechanism to detect it
- The model is overconfident on wrong predictions — Dunite at 97.5%, Gran_Phil
  at 99.8%. Softmax confidence is not a reliable OOD signal for this problem
- Root cause diagnosis: model trained on one quarry per class learned
  source-specific appearance, not general rock mineralogy
- This result directly motivated the multi-source training (notebook 11) and
  the 1D autoencoder OOD detector (notebook 12)

**Input:** `for_test_data_spectral_224/` + saved `.pth` from `resnet_full_evaluation.ipynb`
**Output folder:** `results_inference_new_data/`

---

### 10. `rock_classifier_combined.ipynb`
**Combined: training + evaluation + OOD detection + inference on new rocks**
 
Combines all phases into one notebook. Retrains ResNet-18 on the original 3 classes,
then introduces temperature scaling calibration and energy-score OOD detection
. Runs inference on all 9 new rock folders from
`for_test_data_spectral_224/`. Also includes Grad-CAM on new rock samples and
a t-SNE of training + new rock features combined.
 
**New test data used:** `for_test_data_spectral_224/` (all 9 new rock folders,
4,556 images total converted from BMP by `bmp_to_spectral_224.py`)
 
**Inference results on new rock folders (single-source model + energy-score OOD):**
 
| Folder | Expected | Predicted | OOD flagged | Verdict |
|--------|----------|-----------|-------------|---------|
| Dunite-Ecologite_2Rocks_1-83Hz | Unknown | Leitendorf | 146/500 (29%) | Missed OOD |
| Gneis_1-83Hz | S10Granite | Unknown | 294/500 (59%) | False OOD |
| Granite_3SamplesPhilipp_1-83Hz_1 | S10Granite | Leitendorf | 18/1000 (2%) | Incorrect |
| Granite_3SamplesPhilipp_1-83Hz_2 | S10Granite | Leitendorf | 19/490 (4%) | Incorrect |
| Limestone_CalcsilicaContaminated_U9_U3_1-83Hz | Leitendorf | Holstein | 190/1000 (19%) | Incorrect |
| Limestone_CalcsilicaContaminated_U9_U3_5-10Hz | Leitendorf | Holstein | 370/1000 (37%) | Incorrect |
| Limestone_Rax_1-83Hz_1 | Leitendorf | Leitendorf | 1/25 (4%) | Correct |
| Limestone_Rax_1-83Hz_2 | Leitendorf | Leitendorf | 2/20 (10%) | Correct |
| SandstoneNew_1-83Hz | Holstein | Unknown | 18/21 (86%) | False OOD |
 
**Key findings:**
- Energy score OOD detection misfires in both directions: Gneis and SstNew are
  incorrectly flagged as Unknown while Dunite (the true OOD rock) is not reliably caught
- Root cause of classification failures (possibly): single-source training: model learned
  quarry-specific appearance, not general mineralogy
- Root cause of OOD failures: 224x224 image resizing compresses the wavenumber axis,
  destroying the peak position information that makes Dunite physically distinct
  
**Input:** `rocks_spectral_224/` (training) + `for_test_data_spectral_224/` (inference)
**Output folder:** `results_rock_classifier_combined/`
 
---
 
### 11. `rock_classifier_multisource.ipynb`
**Multi-source training: the generalisation fix**
 
Identified and fixed the root cause of misclassification on new rock samples.
The original model was trained on one specific quarry per class and learned
source-specific spectral appearance rather than general mineralogy.
 
Multi-source training pools multiple sources per class so the model learns the
general mineral signature of each category rather than one specific sample:
 
| Class | Sources added |
|-------|--------------|
| S10Granite | + Gneis + Gran_Phil_1 + Gran_Phil_2 |
| Holstein_Sandstone | + SandstoneNew |
| Leitendorf_Limestone | + CalcSil (1.83Hz + 5.10Hz) + Lst_Rax_1 + Lst_Rax_2 |
| OOD (kept out entirely) | Dunite-Ecologite — never seen during training |
 
**New test data used:** `for_test_data_spectral_224/` (Dunite only, as the only
remaining unseen folder — all other folders became training data)
 
**Inference results on new rock folders (multi-source model + energy-score OOD):**
 
| Folder | Expected | Predicted | OOD flagged | Verdict |
|--------|----------|-----------|-------------|---------|
| Dunite-Ecologite_2Rocks_1-83Hz | Unknown | Leitendorf | 17/500 (3.4%) | Missed OOD |
| Gneis_1-83Hz | S10Granite | S10Granite | — | Correct (now in training) |
| Granite_3SamplesPhilipp_1-83Hz_1 | S10Granite | S10Granite | — | Correct (now in training) |
| Granite_3SamplesPhilipp_1-83Hz_2 | S10Granite | S10Granite | — | Correct (now in training) |
| Limestone_CalcsilicaContaminated_U9_U3_1-83Hz | Leitendorf | Leitendorf | — | Correct (now in training) |
| Limestone_CalcsilicaContaminated_U9_U3_5-10Hz | Leitendorf | Leitendorf | — | Correct (now in training) |
| Limestone_Rax_1-83Hz_1 | Leitendorf | Leitendorf | — | Correct (now in training) |
| Limestone_Rax_1-83Hz_2 | Leitendorf | Leitendorf | — | Correct (now in training) |
| SandstoneNew_1-83Hz | Holstein | Holstein | — | Correct (now in training) |
 
**Key findings:**
- All labelled new rock variants classified correctly once included in training
- Dunite OOD detection still failed (only 17/500 flagged as Unknown) — the
  image feature space cannot distinguish Dunite's olivine peaks from Limestone
  after 224x224 resizing flattens the spectral information
- Multi-source training is the correct and necessary fix for generalisation;
  it is not a workaround but a reflection of how supervised classifiers must work:
  they can only generalise within spectral variability seen during training
**Input:** `rocks_spectral_224/` + `for_test_data_spectral_224/`
**Output folder:** `results_rock_classifier_multisource/`
 
---
 
### 12. `ood_autoencoder_1d.ipynb`
**1D MLP autoencoder for out-of-distribution detection**
 
Addresses OOD detection at the raw spectral level rather than the image level.
Trains an MLP autoencoder (1060->512->128->32->128->512->1060) on the
three known rock classes only using `rock_csvs/`. OOD score = per-spectrum
MSE reconstruction error.

**New test data used:** `profiles_csv/` — 1060-point raw spectral profiles for all
9 new rock folders, converted from PROFILES_*.txt by `convert_new_rocks.py`
 
**Inference results on new rock folders (1D autoencoder OOD, threshold=0.01191):**
 
| Folder | Expected | Mean MSE | OOD flagged | Verdict |
|--------|----------|----------|-------------|---------|
| Dunite-Ecologite_2Rocks_1-83Hz | Unknown | ~0.006 | ~15% at 5% FPR | Partially missed |
| Gneis_1-83Hz | S10Granite (in-distrib) | ~0.003 | <5% | In-distribution |
| Granite_3SamplesPhilipp_1-83Hz_1 | S10Granite (in-distrib) | ~0.002 | <5% | In-distribution |
| Granite_3SamplesPhilipp_1-83Hz_2 | S10Granite (in-distrib) | ~0.002 | <5% | In-distribution |
| Limestone_CalcsilicaContaminated_U9_U3_1-83Hz | Leitendorf (in-distrib) | ~0.006 | ~10% | Slightly elevated |
| Limestone_CalcsilicaContaminated_U9_U3_5-10Hz | Leitendorf (in-distrib) | ~0.006 | ~11% | Slightly elevated |
| Limestone_Rax_1-83Hz_1 | Leitendorf (in-distrib) | ~0.004 | <5% | In-distribution |
| Limestone_Rax_1-83Hz_2 | Leitendorf (in-distrib) | ~0.004 | ~10% | Slightly elevated |
| SandstoneNew_1-83Hz | Holstein (in-distrib) | ~0.001 | <5% | In-distribution |
 
**OOD detection performance (Dunite vs all known rocks):**
 
| Metric | Value |
|--------|-------|
| AUC | 0.68 |
| Random baseline AUC | 0.50 |
| Known rock MSE (mean) | 0.002 - 0.007 |
| Dunite MSE (worst sample) | 0.024 |
| Dunite MSE (median sample) | ~0.005 |
 
**Key findings:**
- AUC 0.68 confirms a genuine OOD signal: better than random (0.50) but not
  reliable enough for deployment at a strict 5% false positive rate
- VIZ-01 visually confirms reconstruction failure on Dunite's olivine peaks:
  the autoencoder produces large orange error regions precisely at peak positions
  it has never seen
- Partial detection is explained by the mixed composition of the test folder:
  it contains both Dunite (strongly anomalous) and Eclogite (partially overlapping
  with training spectra), pulling the median error below the threshold
- The in-distribution rocks (Gneis, Gran_Phil, Lst_Rax, SstNew) all have low
  reconstruction error, confirming the autoencoder correctly recognises their
  mineral signatures as similar to training classes
  
**Input (training):** `rock_csvs/` (original 3 classes only)
**Input (test):** `profiles_csv/` (new rock folders)
**Output folder:** `results_ood_autoencoder_1d/`
 
---
 
### 13. `cnn_1d_train_and_inference.ipynb`
**1D CNN trained on original data, inference on new rock profiles**
 
Trains the original 1D CNN architecture on `rock_csvs/` only (no new rocks added),
then runs inference on all new rock profile CSVs from `profiles_csv/` using the
1D autoencoder from notebook 11 as an OOD pre-filter.
 
**New test data used:** `profiles_csv/` — 1060-point raw spectral profiles for all
9 new rock folders
 
**Inference results on new rock folders (1D CNN + autoencoder OOD pre-filter):**
 
| Folder | Expected | Predicted | OOD flagged | Verdict |
|--------|----------|-----------|-------------|---------|
| Dunite-Ecologite_2Rocks_1-83Hz | Unknown | Leitendorf | 3.4% | Missed OOD |
| Gneis_1-83Hz | S10Granite | S10Granite | 8.2% | Correct |
| Granite_3SamplesPhilipp_1-83Hz_1 | S10Granite | Leitendorf | 0.1% | Incorrect |
| Granite_3SamplesPhilipp_1-83Hz_2 | S10Granite | Leitendorf | 0.0% | Incorrect |
| Limestone_CalcsilicaContaminated_U9_U3_1-83Hz | Leitendorf | Holstein | 9.7% | Incorrect |
| Limestone_CalcsilicaContaminated_U9_U3_5-10Hz | Leitendorf | Holstein | 11.4% | Incorrect |
| Limestone_Rax_1-83Hz_1 | Leitendorf | Leitendorf | 0.0% | Correct |
| Limestone_Rax_1-83Hz_2 | Leitendorf | Leitendorf | 10.0% | Correct |
| SandstoneNew_1-83Hz | Holstein | Leitendorf | 0.0% | Incorrect |
 
**Key findings:**
- The 1D CNN fails on the exact same folders as the ResNet with the same wrong
  predictions — Gran_Phil as Limestone, CalcSil as Sandstone, SstNew as Limestone
- This is the definitive proof that the problem is not model architecture:
  two completely different models (image-based ResNet and spectrum-based 1D CNN)
  produce identical failure patterns
- The TR-00 spectrum plot confirms: SstNew has a sparse quartz peak pattern
  that genuinely resembles Leitendorf Limestone more than Holstein Sandstone in
  feature space. Gran_Phil's spectrum has a different quartz/carbonate balance
  that maps to the Limestone region. The models are classifying by mineral
  composition which is physically correct behaviour
- The only fix is multi-source training (notebook 10), not a better model

**Input (training):** `rock_csvs/` (original 3 classes only)
**Input (test):** `profiles_csv/` + autoencoder from notebook 11
**Output folder:** `results_1d_cnn_inference/`
 
---
 
## Data Paths (change accordingly in each model)
 
| Data type | Path |
|-----------|------|
| Raw TXT files (original training) | `rocks_txts/` |
| CSV profiles (original training) | `rock_csvs/` |
| Resized JPG images 224x224 (training) | `rocks_spectral_224/` |
| Resized JPG images 224x224 (new test) | `for_test_data_spectral_224/` |
| Raw PROFILES TXT (new rock folders) | `profiles/` |
| CSV profiles (new rock folders) | `profiles_csv/` |
