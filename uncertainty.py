import numpy as np

def mc_dropout_scores(model, pairs, passes=5):
    predictions = []

    for _ in range(passes):
        preds = model.predict(pairs)
        predictions.append(preds)

    predictions = np.array(predictions)

    mean = predictions.mean(axis=0)
    std = predictions.std(axis=0)

    adjusted = mean - std  # penalize uncertainty

    return adjusted
