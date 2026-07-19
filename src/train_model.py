import argparse
import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib

from . import config
from .data_prep import build_training_frame


def build_pipeline(C=config.DEFAULT_C):
    preprocess = ColumnTransformer([
        ("num", StandardScaler(), config.NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), config.CATEGORICAL_FEATURES),
    ])
    clf = LogisticRegression(solver="lbfgs", max_iter=2000, C=C)
    return Pipeline([("preprocess", preprocess), ("clf", clf)])


def feature_ranges(frame):
    """Percentile ranges + medians for numeric features -> UI slider defaults."""
    ranges = {}
    for col in config.NUMERIC_FEATURES:
        s = frame[col].astype(float)
        ranges[col] = {
            "min": float(s.quantile(0.01)),
            "max": float(s.quantile(0.99)),
            "median": float(s.median()),
        }
    return ranges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(config.active_csv()))
    ap.add_argument("--C", type=float, default=config.DEFAULT_C)
    args = ap.parse_args()

    print(f"Loading {args.data} ...")
    raw = pd.read_csv(args.data)
    frame = build_training_frame(raw)
    print(f"Model-ready rows: {len(frame):,}")
    print("Class balance:", frame[config.TARGET].value_counts().sort_index().to_dict())

    X = frame[config.ALL_FEATURES]
    y = frame[config.TARGET].astype(int)

    pipe = build_pipeline(C=args.C)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
    print(f"5-fold CV accuracy: {scores.mean():.4f} +/- {scores.std():.3f}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=0)
    pipe.fit(X_tr, y_tr)
    report = classification_report(
        y_te, pipe.predict(X_te),
        labels=config.CLASS_ORDER,
        target_names=[config.TIERS[c]["name"] for c in config.CLASS_ORDER],
        output_dict=True, zero_division=0,
    )

    # refit on ALL data for deployment
    pipe.fit(X, y)

    config.MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(pipe, config.MODEL_PATH)

    meta = {
        "cv_accuracy_mean": float(scores.mean()),
        "cv_accuracy_std": float(scores.std()),
        "C": args.C,
        "n_train_rows": int(len(frame)),
        "class_balance": {int(k): int(v)
                          for k, v in y.value_counts().sort_index().items()},
        "classification_report": report,
        "feature_ranges": feature_ranges(frame),
        "genre_choices": sorted(frame["grouped_genre"].unique().tolist()),
    }
    with open(config.META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved model  -> {config.MODEL_PATH}")
    print(f"Saved meta   -> {config.META_PATH}")


if __name__ == "__main__":
    main()
