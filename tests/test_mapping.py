import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.emotions_dictionary import emotion_to_need, get_emotional_need


print("Testing known label 'anger' →", get_emotional_need('anger')) 


print("Testing label 'joy' →", get_emotional_need('joy')) 


print("Testing unknown label 'alienation' →", get_emotional_need('alienation'))  


print("\nFull mapping sample:")
for emotion in ['fear', 'sadness', 'love', 'surprise', 'nervousness']:
    print(f"{emotion} → {get_emotional_need(emotion)}")


print(f"\nTotal emotions: {len(emotion_to_need)}")
print("Unique needs:", set(emotion_to_need.values()))
