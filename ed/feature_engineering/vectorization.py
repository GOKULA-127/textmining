def emotion_to_vector(emotion_dict):
    emotions = ['joy', 'anger', 'sadness', 'fear', 'neutral']
    return [emotion_dict.get(e, 0) for e in emotions]