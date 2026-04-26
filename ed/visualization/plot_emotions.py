import matplotlib.pyplot as plt

def plot_emotions(vectors):
    anger = [v[1] for v in vectors]
    joy = [v[0] for v in vectors]

    plt.plot(anger, label="Anger")
    plt.plot(joy, label="Joy")

    plt.legend()
    plt.title("Emotion Drift")
    plt.xlabel("Time")
    plt.ylabel("Intensity")

    plt.show()