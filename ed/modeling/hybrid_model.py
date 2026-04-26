import numpy as np

from preprocessing.clean_text import clean_text
from preprocessing.tokenize import tokenize
from preprocessing.lemmatize import lemmatize

from feature_engineering.emotion_scores import get_emotions
from feature_engineering.vectorization import emotion_to_vector

from modeling.lstm_predict import predict_risk
from ollama_module.ollama_analysis import analyze_conversation


def process_conversation(messages):

    emotion_vectors = []
    emotion_labels = []

    for msg in messages:
        clean = clean_text(msg)
        tokens = tokenize(clean)
        lemmas = lemmatize(tokens)
        text = " ".join(lemmas)

        emotions = get_emotions(text)
        vector = emotion_to_vector(emotions)

        emotion_vectors.append(vector)

        label = vector.index(max(vector))
        emotion_labels.append(label)

    if len(emotion_labels) < 3:
        return None

    seq = emotion_labels[-3:]
    risk, pred = predict_risk(seq)

    anger_trend = np.mean([v[1] for v in emotion_vectors])
    sadness_trend = np.mean([v[2] for v in emotion_vectors])

    risk = min(1.0, risk + 0.3 * max(anger_trend, sadness_trend))

    conversation_text = "\n".join(messages)
    explanation = analyze_conversation(conversation_text)

    return {
        "vectors": emotion_vectors,
        "risk": risk,
        "explanation": explanation
    }