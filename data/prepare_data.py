import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config
from src.data_prep import group_genre

DATA_DIR = Path(__file__).resolve().parent
OUT = DATA_DIR / "real_tracks.csv"


def build_priors(song_level):
    """
    song_level: one row per (Performer, Song) with columns
        Performer, Song, peak, first_date
    Returns the same frame with the four prior_* columns added, computed from
    each artist's strictly-earlier songs (chronological, leakage-safe).
    """
    d = song_level.sort_values(["Performer", "first_date", "Song"]).copy()
    g = d.groupby("Performer", sort=False)

    d["prior_song_count"] = g.cumcount()
    d["is_top10"] = (d["peak"] <= 10).astype(int)
    d["prior_top10_count"] = g["is_top10"].cumsum().shift(fill_value=0)
    # expanding best/avg of PAST songs -> shift by one within artist
    d["prior_best_peak"] = (g["peak"].expanding().min()
                            .reset_index(level=0, drop=True).groupby(d["Performer"]).shift())
    d["prior_avg_peak"] = (g["peak"].expanding().mean()
                           .reset_index(level=0, drop=True).groupby(d["Performer"]).shift())
    d[["prior_best_peak", "prior_avg_peak"]] = \
        d[["prior_best_peak", "prior_avg_peak"]].fillna(101)
    # a brand-new artist's first song has count 0 and no top10 history
    d.loc[d["prior_song_count"] == 0, "prior_top10_count"] = 0
    return d.drop(columns="is_top10")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hot-stuff", default=str(DATA_DIR / "raw" / "Hot_Stuff.csv"))
    ap.add_argument("--audio",
                    default=str(DATA_DIR / "raw" / "Hot_100_Audio_Features.csv"))
    args = ap.parse_args()

    print("Loading chart history ...")
    hs = pd.read_csv(args.hot_stuff,
                     usecols=["WeekID", "Week Position", "Song", "Performer",
                              "SongID"])
    hs["date"] = pd.to_datetime(hs["WeekID"], errors="coerce")

    # one row per song: true peak + first chart date
    agg = (hs.groupby("SongID")
             .agg(Song=("Song", "first"),
                  Performer=("Performer", "first"),
                  peak=("Week Position", "min"),
                  first_date=("date", "min"))
             .reset_index())
    print(f"  unique songs: {len(agg):,}")

    print("Loading audio features ...")
    audio_cols = (["SongID", "spotify_genre", "spotify_track_popularity"]
                  + config.AUDIO_FEATURES)
    af = pd.read_csv(args.audio, usecols=lambda c: c in audio_cols)

    df = agg.merge(af, on="SongID", how="inner")
    df = df.dropna(subset=["danceability"])          # keep songs with features
    print(f"  songs with audio features: {len(df):,}")

    # derived fields
    df["grouped_genre"] = df["spotify_genre"].apply(group_genre)
    df["release_month"] = df["first_date"].dt.month.fillna(6).astype(int)
    df["release_year"] = df["first_date"].dt.year.fillna(2000).astype(int)
    df["Peak Position"] = df["peak"].astype(int)

    df = build_priors(df)

    keep = (["Song", "Performer", "Peak Position", "grouped_genre",
             "release_year", "release_month", "spotify_track_popularity"]
            + config.AUDIO_FEATURES + config.PRIOR_FEATURES)
    out = df[keep].dropna().reset_index(drop=True)
    out.to_csv(OUT, index=False)

    print(f"\nWrote {len(out):,} songs -> {OUT}")
    print("Chart-tier balance:",
          out["Peak Position"].apply(config.peak_to_class).value_counts()
          .sort_index().to_dict())
    print("Genres:", out["grouped_genre"].value_counts().head(6).to_dict())


if __name__ == "__main__":
    main()
