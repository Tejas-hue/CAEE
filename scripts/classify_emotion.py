from utils.emotions_dictionary import get_emotional_need

need = get_emotional_need('anger')  # → 'validation'
need2 = get_emotional_need('alienation')  # → 'unknown'

# Example usage
emotion_label = 'fear'
# Output: safety
print(emotion_to_need[emotion_label])  
