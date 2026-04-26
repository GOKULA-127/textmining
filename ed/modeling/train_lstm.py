import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# ---------------- LOAD DATA ----------------
df = pd.read_csv("data/processed/processed_conversations.csv")

grouped = df.groupby("conversation_id")

# ---------------- EMOTION MAPPING (7 → 5) ----------------
def map_emotion(e):
    mapping = {
        0: 4,  # neutral → neutral
        1: 1,  # anger
        2: 1,  # disgust → anger
        3: 3,  # fear
        4: 0,  # happiness → joy
        5: 2,  # sadness
        6: 4   # surprise → neutral
    }
    return mapping[e]

# ---------------- BUILD SEQUENCES ----------------
sequences = []
labels = []

window_size = 3

for _, group in grouped:
    emotions = [map_emotion(e) for e in group["emotion"].values]

    for i in range(len(emotions) - window_size):
        seq = emotions[i:i+window_size]
        label = emotions[i+window_size]

        sequences.append(seq)
        labels.append(label)

X = np.array(sequences)
y = np.array(labels)

# ---------------- ONE-HOT (NOW 5 CLASSES) ----------------
num_classes = 5

X = to_categorical(X, num_classes=num_classes)
y = to_categorical(y, num_classes=num_classes)

# ---------------- SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- MODEL ----------------
model = Sequential()
model.add(LSTM(64, input_shape=(window_size, num_classes)))
model.add(Dense(32, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# ---------------- TRAIN ----------------
model.fit(X_train, y_train, epochs=10, batch_size=32)

# ---------------- EVALUATE ----------------
loss, acc = model.evaluate(X_test, y_test)
print("Test Accuracy:", acc)

# ---------------- SAVE ----------------
model.save("modeling/lstm_model.h5")

print("Aligned model trained and saved!")