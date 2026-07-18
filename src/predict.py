import json
import functools
import numpy as np
import pandas as pd
import joblib

from . import config


@functools.lru_cache(maxsize=1)
def load_model():
    if not config.MODEL_PATH.exists():
        raise FileNotFoundError(
            "No trained model found. Run:  python -m src.train_model")
    model = joblib.load(config.MODEL_PATH)
    with open(config.META_PATH) as f:
        meta = json.load(f)
    return model, meta


def default_candidate():
    """A neutral starting track built from the training medians (UI defaults)."""
    _, meta = load_model()
    r = meta["feature_ranges"]
    cand = {c: r[c]["median"] for c in config.NUMERIC_FEATURES}
    cand["grouped_genre"] = "pop"
    cand["release_month"] = 6
    return cand


def predict_track(candidate: dict):
    """
    candidate: dict with every key in config.ALL_FEATURES.
    Returns: dict with predicted class id, tier metadata, and probabilities
             keyed by class id (probabilities always cover classes 1,2,3).
    """
    model, _ = load_model()
    row = {c: candidate[c] for c in config.ALL_FEATURES}
    X = pd.DataFrame([row])

    classes = list(model.classes_)
    proba = model.predict_proba(X)[0]
    prob_by_class = {int(c): float(p) for c, p in zip(classes, proba)}
    # guarantee all three tiers present
    for c in config.CLASS_ORDER:
        prob_by_class.setdefault(c, 0.0)

    pred_class = int(max(prob_by_class, key=prob_by_class.get))
    return {
        "pred_class": pred_class,
        "tier": config.TIERS[pred_class],
        "prob_by_class": prob_by_class,
        # P(reaches at least Tier B) = P(class 2) + P(class 3)
        "p_at_least_B": prob_by_class[2] + prob_by_class[3],
    }
