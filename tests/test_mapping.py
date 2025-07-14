import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.emotions_dictionary import emotion_to_need, get_emotional_need

# Test known label
print("Testing known label 'anger' →", get_emotional_need('anger'))  # Expect: validation

# Test another label
print("Testing label 'joy' →", get_emotional_need('joy'))  # Expect: celebration

# Test unknown label
print("Testing unknown label 'alienation' →", get_emotional_need('alienation'))  # Expect: unknown

# Print all mappings to visually verify
print("\nFull mapping sample:")
for emotion in ['fear', 'sadness', 'love', 'surprise', 'nervousness']:
    print(f"{emotion} → {get_emotional_need(emotion)}")

# Optional: Show number of categories
print(f"\nTotal emotions: {len(emotion_to_need)}")
print("Unique needs:", set(emotion_to_need.values()))
