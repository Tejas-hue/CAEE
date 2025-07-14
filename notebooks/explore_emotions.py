from datasets import load_dataset


dataset = load_dataset("go_emotions")


label_names = dataset['train'].features['labels'].feature.names

print("Label index to name mapping:")
for i, name in enumerate(label_names):
    print(f"{i}: {name}")

