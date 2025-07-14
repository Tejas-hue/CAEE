import joblib
from sentence_transformers import SentenceTransformer
import numpy as np
import csv
import os
from datetime import datetime
from colorama import Fore, Style, init


init(autoreset=True)

print(Fore.YELLOW + "Loading models...")
models = joblib.load("models/xgb_multi_models.pkl")
mlb = joblib.load("models/label_encoder.pkl")


encoder = SentenceTransformer("all-MiniLM-L6-v2")


os.makedirs("outputs", exist_ok=True)
log_path = "outputs/prediction_logs.csv"
if not os.path.exists(log_path):
    with open(log_path, "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "input_text", "predicted_needs", "scores"])

def predict_needs(text, threshold=0.3, debug=True):
    emb = encoder.encode([text])
    all_scores = {label: clf.predict_proba(emb)[0][1] for clf, label in zip(models, mlb.classes_)}

    
    predictions = [(label, round(prob, 3)) for label, prob in all_scores.items() if prob >= threshold]

    
    if predictions == [('neutral', round(all_scores['neutral'], 3))]:
        fallback_preds = [(label, round(prob, 3)) for label, prob in all_scores.items()
                          if prob >= 0.15 and label != 'neutral']
        if fallback_preds:
            predictions += fallback_preds

    if debug:
        print(Fore.CYAN + "\n Debug: All Need Probabilities")
        for label, prob in sorted(all_scores.items(), key=lambda x: -x[1]):
            print(f"   {label}: {round(prob, 3)}")

    return predictions


print(Fore.GREEN + "\n Context-Aware Psychological Need Predictor (Multilabel)\nType 'exit' to quit.\n")
while True:
    inp = input(Fore.CYAN + "Enter a sentence: ")
    if inp.lower() == "exit":
        break

    results = predict_needs(inp)

    if results:
        print(Fore.MAGENTA + "→ Predicted Needs:")
        for label, score in results:
            color = Fore.GREEN if score > 0.75 else Fore.YELLOW if score > 0.5 else Fore.RED
            print(f"   {color}{label}: {score}")
    else:
        print(Fore.RED + "→ No strong psychological needs detected.")

    
    timestamp = datetime.now().isoformat()
    with open(log_path, "a", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            timestamp,
            inp,
            "|".join([label for label, _ in results]),
            "|".join([str(score) for _, score in results])
        ])
    print()
