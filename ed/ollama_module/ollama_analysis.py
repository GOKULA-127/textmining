import requests

def analyze_conversation(conversation):

    prompt = f"""
You are an expert in emotional behavior analysis.

Analyze this conversation:

{conversation}

Tasks:
1. Emotion of each message
2. Emotion trend over time
3. Detect:
   - Escalation
   - Depression
   - Manipulation

4. Final Risk Level (Low / Medium / High)

Give structured output.
"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3",
            "prompt": prompt,
            "stream": False
        }
    )

    return response.json()["response"]