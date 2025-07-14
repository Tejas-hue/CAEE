#Categorizing emotions from GoEmotions
emotion_to_need = {
    'admiration': 'support',
    'amusement': 'connection',
    'anger': 'validation',
    'annoyance': 'validation',
    'approval': 'support',
    'caring': 'connection',
    'confusion': 'clarity',
    'curiosity': 'clarity',
    'desire': 'support',
    'disappointment': 'comfort',
    'disapproval': 'validation',
    'disgust': 'validation',
    'embarrassment': 'understanding',
    'excitement': 'motivation',
    'fear': 'safety',
    'gratitude': 'acknowledgment',
    'grief': 'comfort',
    'joy': 'celebration',
    'love': 'connection',
    'nervousness': 'safety',
    'optimism': 'motivation',
    'pride': 'acknowledgment',
    'realization': 'clarity',
    'relief': 'comfort',
    'remorse': 'understanding',
    'sadness': 'comfort',
    'surprise': 'clarity',
    'neutral': 'neutral'
}
def get_emotional_need(emotion_label):
    return emotion_to_need.get(emotion_label, "unknown")

def get_emotional_needs(emotions: list[str]) -> list[str]:
    """Map a list of emotions to a list of unique psychological needs."""
    needs = set()
    for emotion in emotions:
        if emotion in emotion_to_need:
            needs.add(emotion_to_need[emotion])
    return list(needs) if needs else ["neutral"]

def get_all_needs():
    return sorted(set(emotion_to_need.values()))


