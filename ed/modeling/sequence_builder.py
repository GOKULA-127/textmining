def build_sequences(vectors, window_size=3):
    sequences = []

    for i in range(len(vectors) - window_size):
        sequences.append(vectors[i:i+window_size])

    return sequences