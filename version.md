###### **# Version 1.1 – Multilabel XGBoost (13 July 2025)**

Updates:

-Switched to multilabel classification — now predicts multiple needs per sentence.

-Trained a separate XGBoost model per need, tuned with Optuna.

-Added confidence scores using .predict\_proba().

-CLI now has color-coded output (with colorama) for readability.

-Predictions are logged to outputs/prediction\_log.csv.

-Improved accuracy for nuanced, overlapping needs.

-Previous version (v1.0): single-label, ~48% accuracy, no probabilities.

##### 

###### **# Version 1.0 - XGBoost Need Predictor**

\- Uses S-BERT embeddings

\- Maps GoEmotions → psychological needs

\- Classifier: Tuned XGBoost with Optuna

\- Accuracy: ~48%

\- Outputs top-3 need probabilities

\- Last trained: \[12th July 2025]

