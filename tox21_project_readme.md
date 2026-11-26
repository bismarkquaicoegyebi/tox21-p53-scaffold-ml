# Tox21 SR-p53 Molecular Stress Response Prediction

End-to-end machine learning pipeline predicting small-molecule activation of the **p53 cellular stress response pathway** (`SR-p53`) using RDKit cheminformatics and PyTorch on CUDA (NVIDIA T4 GPU). Built to evaluate rare-event classification and out-of-distribution generalization across unseen chemical scaffolds.

---

## Key Highlights & Experimental Results

- **Dataset & Preprocessing:** Sanitized 7,831 raw NIH Tox21 records down to **6,767 valid compounds** (**423 active / 6.25%**, **6,344 inactive / 93.75%**) after filtering `NaN` assay labels and invalid SMILES strings (`7` invalid SMILES removed).
- **Molecular Representation:** Generated **2,048-bit Morgan fingerprints** (`radius=2`, ECFP4 equivalent) via RDKit `GetMorganGenerator`, yielding a `(6767, 2048)` binary matrix averaging **28.88 active bits** per molecule (~1.41% density).
- **Structural Data Leakage Analysis:** Exposed sharp overfitting in tree ensembles when moving from random stratified splits to **Bemis-Murcko scaffold splitting**—mirroring purged/grouped out-of-sample validation in quantitative finance.
- **Biological Case Study:** On held-out scaffold test compound **pentachlorophenol** (`TOX1106`, MW `266.34 g/mol`, true active label `1`), the regularized PyTorch MLP predicted a **0.9756** activation probability versus **0.4091** (Logistic Regression) and **0.2033** (Random Forest).

| Model | Validation Protocol | ROC-AUC | PR-AUC | `TOX1106` Test Prob. |
| :--- | :--- | :---: | :---: | :---: |
| Logistic Regression (`balanced`) | 5-Fold Stratified CV | 0.7784 ± 0.0307 | 0.3153 ± 0.0400 | — |
| Random Forest (`300 trees`) | 5-Fold Stratified CV | 0.8590 ± 0.0205 | 0.4194 ± 0.0311 | — |
| PyTorch MLP (T4 GPU) | 5-Fold Stratified CV | 0.7530 ± 0.0380 | 0.3070 ± 0.0362 | — |
| Logistic Regression (`balanced`) | Bemis-Murcko Scaffold (20% Test) | 0.7561 | 0.2245 | 0.4091 |
| Random Forest (`300 trees`) | Bemis-Murcko Scaffold (20% Test) | 0.7151 | 0.1250 | 0.2033 |
| **PyTorch MLP (T4 GPU)** | **Bemis-Murcko Scaffold (20% Test)** | **0.7785** | **0.2542** | **0.9756 (True = 1)** |

---

## Methodology Overview

1. **Data Sanitization:** Filtered missing `SR-p53` labels (`1,057` untested) and parsed SMILES via `Chem.MolFromSmiles` (`7` invalid structures removed).
2. **Featurization:** Encoded local atomic neighborhoods into 2,048-bit ECFP4 vectors (`float32`).
3. **Imbalance-Aware Baselines:** Trained `class_weight="balanced"` Logistic Regression and Random Forest classifiers evaluated on **ROC-AUC** and **Precision-Recall AUC (PR-AUC)** to avoid accuracy inflation on a 93.75%-negative target.
4. **Regularized Neural Architecture (`FingerprintMLP`):**
   - `Linear(2048 -> 512) -> BatchNorm1d -> SiLU -> Dropout(0.3)`
   - `Linear(512 -> 128) -> BatchNorm1d -> SiLU -> Dropout(0.2)`
   - `Linear(128 -> 1)` trained with `AdamW` (`lr=1e-3`, `weight_decay=1e-4`) and `BCEWithLogitsLoss(pos_weight = N_neg / N_pos)` over 25 epochs on CUDA.
5. **Bemis-Murcko Scaffold Split (80/20):** Grouped compounds by core ring scaffolds (`5,414` train at `6.74%` positive; `1,353` test at `4.29%` positive) so no chemical backbones leaked across train and test sets.

---

## Known Limitations & Roadmap

- **Probability Calibration:** High inverse-prevalence weighting (`pos_weight ~ 15`) improves tail recall but shifts raw output probabilities; Platt or isotonic scaling is required prior to thresholded deployment.
- **Fingerprint Collisions & 3D Geometry:** Fixed 2,048-bit hashing discards stereochemistry and global physicochemical descriptors (`LogP`, TPSA, MW). Planned next step: concatenate RDKit 2D descriptors and benchmark a Message Passing Neural Network (MPNN).
- **Nested Validation:** Introduce a nested scaffold validation fold for early stopping on PR-AUC and multi-seed bootstrap confidence intervals.

---

## Tech Stack

`Python 3` · `PyTorch (CUDA)` · `RDKit` · `scikit-learn` · `NumPy` · `pandas`