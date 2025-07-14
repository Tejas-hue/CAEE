import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report
from sentence_transformers import SentenceTransformer
import xgboost as xgb
import optuna


train_df = pd.read_csv("data/goemotions_train_needs.csv")
val_df = pd.read_csv("data/goemotions_val_needs.csv")


train_df['needs'] = train_df['need'].apply(lambda x: x.split('|') if '|' in x else [x])
val_df['needs'] = val_df['need'].apply(lambda x: x.split('|') if '|' in x else [x])


mlb = MultiLabelBinarizer()
y_train = mlb.fit_transform(train_df['needs'])
y_val = mlb.transform(val_df['needs'])

os.makedirs("models", exist_ok=True)
joblib.dump(mlb, "models/label_encoder.pkl")


print("Encoding with Sentence-BERT...")
encoder = SentenceTransformer('all-MiniLM-L6-v2')
X_train = encoder.encode(train_df['text'].tolist(), show_progress_bar=True)
X_val = encoder.encode(val_df['text'].tolist(), show_progress_bar=True)


def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 400),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.3),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "use_label_encoder": False,
        "eval_metric": "logloss"
    }

    scores = []
    for i in range(y_train.shape[1]):
        clf = xgb.XGBClassifier(**params)
        clf.fit(X_train, y_train[:, i])
        preds = clf.predict(X_val)
        acc = np.mean(preds == y_val[:, i])
        scores.append(acc)

    return np.mean(scores)


print("Tuning XGBoost with Optuna...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=5)

print("Best Params:", study.best_params)


final_models = []
for i in range(y_train.shape[1]):
    clf = xgb.XGBClassifier(**study.best_params)
    clf.fit(X_train, y_train[:, i])
    final_models.append(clf)

joblib.dump(final_models, "models/xgb_multi_models.pkl")

# === Step 6: Evaluate ===
print("\n📊 Classification Report:")
all_preds = np.stack([model.predict(X_val) for model in final_models], axis=1)
print(classification_report(y_val, all_preds, target_names=mlb.classes_))
