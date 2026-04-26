import pandas as pd

from preprocessing.clean_text import clean_text
from preprocessing.tokenize import tokenize
from preprocessing.lemmatize import lemmatize

from feature_engineering.emotion_scores import get_emotions
from feature_engineering.vectorization import emotion_to_vector

from visualization.plot_emotions import plot_emotions
df = pd.read_csv("data/processed/processed_conversations.csv")

emotion_vectors = []

for msg in df["message"][:100]:  # limit for testing
    clean = clean_text(msg)
    tokens = tokenize(clean)
    lemmas = lemmatize(tokens)
    text = " ".join(lemmas)

    emotions = get_emotions(text)
    vector = emotion_to_vector(emotions)

    emotion_vectors.append(vector)

plot_emotions(emotion_vectors)