import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datasets import load_dataset
from utils.emotions_dictionary import emotion_to_need, get_emotional_needs
import pandas as pd

# Load the dataset
dataset = load_dataset("go_emotions")

# Get the raw label index-to-name map
label_names = dataset['train'].features['labels'].feature.names

def convert_labels_to_needs(example):
    emotion_labels = [label_names[i] for i in example['labels']]

    # Map emotions → unique needs
    needs = get_emotional_needs(emotion_labels)

    # Store needs as a pipe-separated string
    example['needs'] = "|".join(needs)
    
    return example

# Apply transformation to all splits
dataset = dataset.map(convert_labels_to_needs)

# Save each split as CSV with 'text' and 'needs' columns
os.makedirs("data", exist_ok=True)
dataset['train'].to_pandas()[['text', 'needs']].to_csv("data/goemotions_train_needs.csv", index=False)
dataset['validation'].to_pandas()[['text', 'needs']].to_csv("data/goemotions_val_needs.csv", index=False)
dataset['test'].to_pandas()[['text', 'needs']].to_csv("data/goemotions_test_needs.csv", index=False)

print("✅ Multi-label dataset saved in /data")
