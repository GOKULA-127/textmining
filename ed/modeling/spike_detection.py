def detect_spikes(series, threshold=0.4):
    spikes = []

    for i in range(1, len(series)):
        diff = abs(series[i] - series[i-1])

        if diff > threshold:
            spikes.append((i, diff))

    return spikes