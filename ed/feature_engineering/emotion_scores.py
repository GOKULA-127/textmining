from transformers import pipeline

emotion_model = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None
)

def get_emotions(text):
    results = emotion_model(text)[0]
    return {r['label']: r['score'] for r in results}