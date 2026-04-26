import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

model = load_model("modeling/lstm_model.h5")

def predict_risk(emotion_sequence):
    num_classes = 5  # aligned

    seq = to_categorical([emotion_sequence], num_classes=num_classes)

    pred = model.predict(seq)[0]

    anger = pred[1]
    sadness = pred[2]

    risk = max(anger, sadness)

    return float(risk), pred