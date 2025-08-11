# Employee Turnover Prediction — Portobello Tech

## Developed for
University of San Diego — AI/ML Bootcamp (Powered by Fullstack Academy)  
Course Unit: Machine Learning  
Submission Date: July 12, 2025  
Author: Carllos Watts-Nogueira

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen.svg)
![Model Accuracy](https://img.shields.io/badge/Accuracy-97%25-success.svg)
![ML Task](https://img.shields.io/badge/Task-Classification%20%26%20Clustering-orange.svg)
![Imbalance Handling](https://img.shields.io/badge/SMOTE-Enabled-lightgrey.svg)
![Dashboard](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)
![Made with ❤️](https://img.shields.io/badge/Made%20with-%E2%9D%A4-red.svg)

## Objective
To predict employee attrition using historical HR data and deliver actionable insights for retention planning.

## Overview
This project aims to predict employee turnover at Portobello Tech using machine learning techniques. By analyzing HR data, we identify at-risk employees and recommend actionable retention strategies. The best-performing model achieved 97% accuracy and segmented employees into risk zones based on their probability of leaving.

## Key Tasks

- Perform data quality assessment and remove duplicates
- Conduct EDA, correlation heatmap, and distribution analysis
- Apply unsupervised clustering (KMeans) on employees who left
- Handle class imbalance using SMOTE
- Encode categorical features and preprocess dataset
- Train and evaluate three classifiers: Logistic Regression, Random Forest, Gradient Boosting
- Select best model using ROC/AUC and confusion matrix metrics
- Predict turnover probabilities and categorize into 4 risk zones
- Visualize and interpret satisfaction levels per zone

## Key ML Components

| Task                      | Technique / Tool                        |
|---------------------------|-----------------------------------------|
| Data Cleaning             | pandas, duplicate removal               |
| Feature Encoding          | One-hot encoding for salary & department |
| Clustering                | KMeans + Elbow Method (optimal k = 3)   |
| Imbalance Correction      | SMOTE oversampling                      |
| Classification Models     | Logistic Regression, Random Forest, Gradient Boosting |
| Model Evaluation          | Accuracy · Precision · Recall · F1 · ROC AUC |
| Risk Segmentation         | Probability scores → 4 retention zones  |
| Visualization             | seaborn, matplotlib, Streamlit dashboard |

## Project Goals

- Clean and prepare HR data  
- Identify key features influencing employee turnover  
- Cluster employees using satisfaction and evaluation metrics  
- Handle class imbalance with SMOTE  
- Train and evaluate multiple classifiers  
- Segment employees into turnover risk zones  
- Provide retention strategy recommendations

## Best Performing Model
- **Random Forest Classifier**
  - Accuracy: 97%
  - Precision: 0.94
  - Recall: 0.90
  - F1 Score: 0.92
  - ROC AUC: High

## Tools & Libraries
`pandas`, `numpy`, `seaborn`, `matplotlib`, `scikit-learn`, `imblearn`, `jupyter`

## Risk Zones Defined

| Zone              | Probability Range | Strategy Focus              |
|------------------|-------------------|-----------------------------|
| 🟢 Safe Zone      | < 20%             | Monitor periodically        |
| 🟡 Low-Risk Zone  | 20–60%            | Check-in with managers      |
| 🟠 Medium-Risk    | 60–90%            | Personalized incentives     |
| 🔴 High-Risk Zone | > 90%             | Immediate HR intervention   |

## Visualizations
- Correlation heatmaps
- KMeans scatter plots
- ROC curves and confusion matrices
- Bar charts showing turnover probability zones and satisfaction levels

## Folder Guide
- `data/` — original and cleaned datasets  
- `notebooks/` — modular stages of analysis  
- `models/` — saved classifiers  
- `results/` — performance plots and metrics  
- `dashboards/` — interactive streamlit view *(optional)*  # employee-turnover-prediction-ml

## Strategic Insights
- Overworked and under-promoted staff show higher attrition
- High evaluation + low satisfaction is a red flag
- Customized engagement by risk zone boosts retention

## Getting Started

```bash
# Install dependencies
pip install -r requirements.txt

# Run dashboard locally
streamlit run dashboards/turnover_dashboard.py
```

### Final Submission Update — Enhanced Version

This version includes improvements based on review/feedback and rubric alignment:

- **Elbow Method**: Added to justify optimal clustering (`k=3`)
- **Feature Engineering Variants**:
  - Original (One-hot encoded, unscaled)
  - Interaction Features (`PolynomialFeatures`)
  - Standardized (`StandardScaler`)
  - Outlier-Clipped (IQR filtering)
- **Model Selection**:
  - Final models saved: `Gradient Boosting` and `Random Forest (Standardized)`
  - Evaluated using ROC AUC, Precision, Recall, and F1 Score
- **New Artifacts**:
  - Updated notebook: `MLFinalUpdate.ipynb`
  - Script version: `mlfinalupdate.py`
  - HTML report: `employee_turnover_portobello.html`
  - Visualizations: clustering, distributions, confusion matrices, ROC curves
  - Saved models: `best_model_gradient_boosting.pkl`, `best_model_random_forest_standardized.pkl`

All files are located in the `final_submission/` folder.


