# Drug Toxicity Prediction using Machine Learning (Tox21)

## Project Overview

Drug development often fails due to unexpected toxicity. Early prediction of toxic compounds can significantly reduce costs and improve patient safety.

This project builds a **machine learning pipeline** to predict drug toxicity using:

- Molecular structures (SMILES)
- Molecular descriptors
- Fingerprint features (Morgan fingerprints)

We analyze multiple toxicity targets from the **Tox21 dataset** and compare different ML models.

---

## Dataset

- **Primary Dataset:** Tox21  
- ~12,000 chemical compounds  
- Includes:
  - SMILES (molecular structure)
  - 12 toxicity targets (multi-label classification)
- Contains missing values and class imbalance  

---

## Approach

### 1. Data Preprocessing
- Removed invalid molecules using RDKit  
- Handled missing values per target (no imputation)  
- Selected targets with sufficient positive samples  

---

### 2. Feature Engineering 

We extracted two types of features:

#### Molecular Descriptors
- Molecular Weight (MolWt)
- LogP (lipophilicity)
- HBD / HBA
- Rotatable Bonds
- Aromatic Rings
- TPSA
- Number of Atoms

#### Morgan Fingerprints
- Radius = 2  
- 2048-bit vector representation  

 Final feature vector = **Descriptors + Fingerprints**

---

### 3. Modeling

We trained **separate models for each target**:

-  Random Forest  
-  Logistic Regression  
-  XGBoost (best performing)  

---

### 4. Handling Class Imbalance 

- Used `class_weight='balanced'` (RF, LR)  
- Used `scale_pos_weight` (XGBoost)  
- Stratified train-test split  

---

### 5. Evaluation Metric

- Primary metric: **ROC-AUC**  
- Better than accuracy for imbalanced datasets  

---

##  Visualization

- Missing value analysis  
- Feature distribution plots  
- Correlation heatmaps  
- Model comparison bar charts  

---

##  Key Insights

- Morgan fingerprints significantly improve performance  
- Molecular descriptors provide interpretability  
- Class imbalance strongly affects recall  
- Different targets require separate models  
- Tree-based models outperform linear models  

---

##  Highlights

- End-to-end ML pipeline  
- Multi-target toxicity prediction  
- Feature engineering using RDKit  
- Model comparison and evaluation  
- Clean and scalable implementation  

---

##  Tech Stack

- Python  
- Pandas, NumPy  
- Scikit-learn  
- XGBoost  
- RDKit  
- Matplotlib, Seaborn  

---

##  How to Run

pip install -r requirements.txt

notebook Tox_Predict.ipynb


## Model Explainability

To understand model decisions, we analyze feature importance:

- Molecular descriptors like **LogP, TPSA, and HBA** contribute significantly
- Morgan fingerprints capture structural patterns linked to toxicity

Future work includes:
- SHAP (SHapley Additive Explanations)
- Visual interpretation of important features

This helps identify which molecular properties drive toxicity.

---

---

## Conclusion

This project demonstrates an effective pipeline for predicting drug toxicity using machine learning.

- Feature engineering (descriptors + fingerprints) is critical
- Tree-based models outperform linear approaches
- Handling class imbalance improves results significantly

 This approach can be extended to real-world drug discovery pipelines.

---
