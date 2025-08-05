# Employee Turnover Prediction — Portobello Tech

## Developed for
University of San Diego — AI/ML Bootcamp (Powered by Fullstack Academy)  
Course Unit: Machine Learning  
Submission Date: July 12, 2025  
Author: Carllos Watts-Nogueira

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


