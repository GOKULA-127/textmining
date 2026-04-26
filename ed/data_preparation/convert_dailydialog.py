import pandas as pd

text_file = "data/raw/dialogues_text.txt"
emotion_file = "data/raw/dialogues_emotion.txt"

data = []

with open(text_file, "r", encoding="utf-8") as f_text, \
     open(emotion_file, "r", encoding="utf-8") as f_emotion:

    for conv_id, (text_line, emo_line) in enumerate(zip(f_text, f_emotion)):

        utterances = text_line.strip().split("__eou__")
        utterances = [u.strip() for u in utterances if u.strip()]

        emotions = list(map(int, emo_line.strip().split()))

        if len(utterances) != len(emotions):
            continue

        for i, (utt, emo) in enumerate(zip(utterances, emotions)):
            speaker = "A" if i % 2 == 0 else "B"

            data.append({
                "conversation_id": conv_id,
                "time": i,
                "user": speaker,
                "message": utt,
                "emotion": emo
            })

df = pd.DataFrame(data)
df.to_csv("data/processed/processed_conversations.csv", index=False)

print("Converted successfully!")