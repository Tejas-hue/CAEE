from datasets import load_dataset

# Load GoEmotions
dataset = load_dataset("go_emotions")

# Get label names (28 total)
label_names = dataset['train'].features['labels'].feature.names

print("Label index to name mapping:")
for i, name in enumerate(label_names):
    print(f"{i}: {name}")

