import pandas as pd
from ollama_analysis import analyze_conversation

# Load your processed dataset
df = pd.read_csv("../data/processed/processed_conversations.csv")

grouped = df.groupby("conversation_id")

print("Running Ollama Analysis...\n")

for conv_id, group in grouped:

    conversation = "\n".join([
        f"{row['user']}: {row['message']}"
        for _, row in group.iterrows()
    ])

    result = analyze_conversation(conversation)

    print("\n==============================")
    print(f"Conversation ID: {conv_id}")
    print(result)

    break  # test only first conversation