import ast
import numpy as np
import pandas as pd

from . import config

# Genre grouping (condensed from the notebook's SEED_BUCKETS)

SEED_BUCKETS = {
    "electronic": ["edm", "electronic", "house", "techno", "trance", "dubstep",
                   "drum and bass", "disco", "electro", "ambient", "garage"],
    "hiphop": ["hip hop", "hip-hop", "rap", "drill", "trap", "pop rap", "grime"],
    "pop": ["pop", "dance pop", "synthpop", "electropop", "teen pop",
            "indie pop", "art pop"],
    "rnb_soul": ["r&b", "rnb", "soul", "neo soul", "motown"],
    "rock": ["rock", "alt rock", "indie rock", "punk", "emo", "hard rock",
             "classic rock", "grunge"],
    "metal": ["metal", "metalcore", "death metal", "nu metal", "thrash"],
    "country": ["country", "country pop", "bluegrass", "americana"],
    "latin": ["latin", "reggaeton", "urbano", "bachata", "salsa", "cumbia"],
    "reggae": ["reggae", "dancehall", "dub"],
    "jazz": ["jazz", "bebop", "swing", "bossa nova"],
    "folk_acoustic": ["folk", "acoustic", "singer-songwriter"],
}


def _first_genre(val):
    """Reduce a list-like genre string (e.g. "['dance pop','pop']") to one token."""
    if pd.isna(val):
        return np.nan
    if isinstance(val, (list, tuple)):
        return val[0] if val else np.nan
    s = str(val).strip()
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, (list, tuple)) and parsed:
            return parsed[0]
    except Exception:
        pass
    if "," in s:
        return s.split(",")[0].strip().strip("'\"")
    return s


def group_genre(raw_genre):
    """Map any raw genre string to one of the config.GENRE_CHOICES buckets."""
    token = _first_genre(raw_genre)
    if pd.isna(token):
        return "other"
    token = str(token).lower().strip().strip("'\"")
    for bucket, seeds in SEED_BUCKETS.items():
        if any(seed in token for seed in seeds):
            return bucket
    return "other"

# Prior (artist track-record) features

def _build_priors(df):
    """
    Computeartist priors: for each song, use only the artist's
    EARLIER songs (chronological cumulative stats). Mirrors notebook cell 61.
    Requires columns: Performer, Song, Peak Position, and a year to order by.
    """
    d = df.copy()
    order_col = "release_year" if "release_year" in d.columns else None
    d["_artist"] = d["Performer"].astype(str)

    song = (d.groupby(["_artist", "Song"], as_index=False)
              .agg(min_peak=("Peak Position", "min"),
                   yr=(order_col, "min") if order_col else ("Peak Position", "min")))
    song = song.sort_values(["_artist", "yr", "Song"])

    g = song.groupby("_artist")
    song["prior_song_count"] = g.cumcount()
    song["is_top10"] = (song["min_peak"] <= 10).astype(int)
    song["prior_top10_count"] = g["is_top10"].cumsum().shift(fill_value=0)
    song["prior_best_peak"] = (g["min_peak"].expanding().min()
                                .reset_index(level=0, drop=True).shift())
    song["prior_avg_peak"] = (g["min_peak"].expanding().mean()
                               .reset_index(level=0, drop=True).shift())
    song[["prior_best_peak", "prior_avg_peak"]] = \
        song[["prior_best_peak", "prior_avg_peak"]].fillna(101)

    out = d.merge(
        song[["_artist", "Song", "prior_song_count", "prior_top10_count",
              "prior_best_peak", "prior_avg_peak"]],
        on=["_artist", "Song"], how="left",
    ).drop(columns="_artist")
    return out


def build_training_frame(df):
    """
    Return a frame containing config.ALL_FEATURES + config.TARGET, one row per
    (song). Rebuilds grouped_genre / priors only if absent.
    """
    d = df.copy()

    if "grouped_genre" not in d.columns:
        source = "spotify_genre" if "spotify_genre" in d.columns else None
        d["grouped_genre"] = d[source].apply(group_genre) if source else "other"

    if any(c not in d.columns for c in config.PRIOR_FEATURES):
        d = _build_priors(d)

    # Derive release_month for real data. Prefer a true release date; fall back
    # to the chart-week date (WeekID) as a proxy for when the song was working.
    if "release_month" not in d.columns:
        date_col = next((c for c in ["release_date", "WeekID", "WeekId"]
                         if c in d.columns), None)
        if date_col is not None:
            d["release_month"] = (pd.to_datetime(d[date_col], errors="coerce")
                                    .dt.month)
        else:
            d["release_month"] = 6  # neutral default if no date available
    d["release_month"] = d["release_month"].fillna(6).astype(int)

    d[config.TARGET] = d["Peak Position"].apply(config.peak_to_class)
    d = d[d[config.TARGET].notna()].copy()
    d[config.TARGET] = d[config.TARGET].astype(int)

    keep = config.ALL_FEATURES + [config.TARGET]
    missing = [c for c in keep if c not in d.columns]
    if missing:
        raise ValueError(f"Input data is missing required columns: {missing}")

    return d[keep].dropna().reset_index(drop=True)
