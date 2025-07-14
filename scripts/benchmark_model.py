import json
import os
import sys
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer
import joblib
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.emotions_dictionary import get_all_needs  


print("🔍 Loading model and encoder...")
model = joblib.load("models/need_classifier.pkl")
label_encoder: LabelEncoder = joblib.load("models/label_encoder.pkl")
model_classes = label_encoder.classes_


embedder = SentenceTransformer("all-MiniLM-L6-v2")


with open("tests/benchmark_inputs.json", "r", encoding="utf-8") as f:
    benchmark = json.load(f)["samples"]


correct = 0
total = len(benchmark)
print("\n🧪 Running Benchmark...\n")

for i, sample in enumerate(benchmark):
    text = sample["text"]
    expected = set(sample["expected"])

    embedding = embedder.encode([text])
    probs = model.predict_proba(embedding)[0]
    sorted_indices = np.argsort(probs)[::-1]

    top_predictions = [model_classes[idx] for idx in sorted_indices[:3]]
    top_probs = [probs[idx] for idx in sorted_indices[:3]]

    predicted = set(top_predictions)

    is_correct = bool(predicted & expected)
    if is_correct:
        correct += 1

    print(f"{i+1}. \"{text}\"")
    print(f"    ✅ Expected: {expected}")
    print(f"    🔮 Predicted: {dict(zip(top_predictions, [round(p, 2) for p in top_probs]))}")
    print(f"    {'✔️ Correct' if is_correct else '❌ Incorrect'}\n")

# Accuracy
accuracy = round(correct / total * 100, 2)
print(f"🎯 Final Benchmark Score: {accuracy}% ({correct}/{total} correct)")
