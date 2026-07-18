from pathlib import Path

# Paths

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
SAMPLE_CSV = DATA_DIR / "sample_tracks.csv"
REAL_CSV = DATA_DIR / "real_tracks.csv"
MODEL_PATH = MODEL_DIR / "chart_model.joblib"
META_PATH = MODEL_DIR / "model_meta.json"

# Features the model uses.
#
# Design notes:
#  * We dropped the notebook's main_artist one-hot: it doesn't generalise to new
#    artists and dominated the coefficients. Numeric artist priors replace it.
#  * We use release_MONTH (categorical, cyclical) instead of release_YEAR.
#    A label scoping a future single always has "next year" as the year, so year
#    can't discriminate between candidates and would force the model to
#    extrapolate past its training range. Release month is a genuine decision
#    and stays inside the trained range.

AUDIO_FEATURES = [
    "danceability", "energy", "loudness", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "tempo", "key",
    "time_signature", "spotify_track_duration_ms",
]

PRIOR_FEATURES = [
    "prior_song_count", # how many charting songs the artist already has
    "prior_top10_count", # how many of those hit the top 10
    "prior_best_peak", # best (lowest) peak position to date; 101 = none
    "prior_avg_peak", # average peak position to date; 101 = none
]

CONTEXT_FEATURES = ["spotify_track_popularity"] # artist momentum

NUMERIC_FEATURES = AUDIO_FEATURES + PRIOR_FEATURES + CONTEXT_FEATURES
CATEGORICAL_FEATURES = ["grouped_genre", "release_month"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TARGET = "chart_class"

# Target definition

def peak_to_class(peak):
    try:
        p = int(peak)
    except (TypeError, ValueError):
        return None
    if 1 <= p <= 5:
        return 3 # Tier A - Hit contender
    elif 6 <= p <= 40:
        return 2 # Tier B - Charting potential
    elif 41 <= p <= 100:
        return 1 # Tier C - Long shot
    return None


TIERS = {
    3: {"code": "A", "name": "Hit contender",      "peak_range": "Top 5",    "color": "#1DB954"},
    2: {"code": "B", "name": "Charting potential", "peak_range": "6 - 40",   "color": "#F5A623"},
    1: {"code": "C", "name": "Long shot",          "peak_range": "41 - 100", "color": "#9B9B9B"},
}
CLASS_ORDER = [3, 2, 1]

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

GENRE_CHOICES = [
    "pop", "hiphop", "rnb_soul", "rock", "electronic", "country",
    "latin", "folk_acoustic", "reggae", "metal", "jazz", "other",
]

# INPUT LAYER
#
# The model was trained on Spotify's technical 0-1 scores. Record managers don't
# think in "acousticness = 0.34". FEATURE_UI describes each audio feature in
# plain language with anchored ends, and the two helpers convert between the
# slider value a manager sets and the number the model expects.

LOUDNESS_DB_RANGE = (-20.0, -3.0)   # quiet/dynamic .. loud/compressed

FEATURE_UI = {
    "danceability": {
        "label": "Danceable", "kind": "pct",
        "help": "How much the beat/groove makes people want to move.",
        "low": "A ballad you'd sway to", "high": "A club/party track"},
    "energy": {
        "label": "Energy", "kind": "pct",
        "help": "Overall intensity and power.",
        "low": "Calm, mellow, gentle", "high": "Loud, fast, intense"},
    "valence": {
        "label": "Mood", "kind": "pct",
        "help": "Emotional tone of the song.",
        "low": "Sad, dark, serious", "high": "Happy, upbeat, cheerful"},
    "acousticness": {
        "label": "Acoustic feel", "kind": "pct",
        "help": "Acoustic instruments vs electronic/heavily produced.",
        "low": "Fully produced / electronic", "high": "Acoustic (live guitar, piano)"},
    "speechiness": {
        "label": "Rapping / spoken words", "kind": "pct",
        "help": "How much of the track is rapped or spoken vs sung.",
        "low": "All sung / melodic", "high": "Mostly rap / spoken word"},
    "instrumentalness": {
        "label": "Instrumental", "kind": "pct",
        "help": "How instrumental the track is (little or no vocals).",
        "low": "Vocal-led song (normal)", "high": "Instrumental, few/no vocals"},
    "liveness": {
        "label": "Live feel", "kind": "pct",
        "help": "Does it sound like a live performance or a clean studio cut?",
        "low": "Clean studio recording", "high": "Live / concert feel"},
    "loudness": {
        "label": "Production loudness", "kind": "loudness",
        "help": "How loud and compressed the master is.",
        "low": "Quiet, dynamic", "high": "Loud, punchy, radio-ready"},
    "tempo": {
        "label": "Tempo (BPM)", "kind": "bpm", "min": 60, "max": 200, "step": 1,
        "help": "Beats per minute. ~70 = slow ballad, ~120 = pop, ~160 = uptempo.",
        "low": "Slow (~70)", "high": "Fast (~160)"},
    "spotify_track_duration_ms": {
        "label": "Song length (minutes)", "kind": "minutes",
        "min": 1.5, "max": 7.0, "step": 0.1,
        "help": "Total runtime of the track.",
        "low": "Short (radio edit)", "high": "Long"},
    "key": {
        "label": "Musical key", "kind": "raw", "min": 0, "max": 11, "step": 1,
        "help": "0 = C, 1 = C#, ... 11 = B. Minor influence.",
        "low": "C", "high": "B"},
    "time_signature": {
        "label": "Time signature (beats/bar)", "kind": "raw",
        "min": 3, "max": 5, "step": 1,
        "help": "Almost always 4. Minor influence.",
        "low": "3", "high": "5"},
}

def display_to_model(feat, disp):
    """Convert a slider value a manager set into the model's expected number."""
    kind = FEATURE_UI[feat]["kind"]
    if kind == "pct":
        return disp / 10.0 # 0-10 slider -> 0-1 model scale
    if kind == "minutes":
        return disp * 60000.0
    if kind == "loudness":
        lo, hi = LOUDNESS_DB_RANGE
        return lo + (disp / 100.0) * (hi - lo)
    return disp


def model_to_display(feat, val):
    """Convert a model number into the slider value shown to the manager."""
    kind = FEATURE_UI[feat]["kind"]
    if kind == "pct":
        return val * 10.0 # 0-1 model scale -> 0-10 slider
    if kind == "minutes":
        return val / 60000.0
    if kind == "loudness":
        lo, hi = LOUDNESS_DB_RANGE
        return (val - lo) / (hi - lo) * 100.0
    return val

# DECISION 
#
# Each tier maps to a release action with a recommended promo budget and a
# channel split. These are defaults that the user can override in the UI.
# Overriding changes the recommended decision.

DEFAULT_PLAYBOOK = {
    3: {"action": "Full campaign", "budget": 120_000,
        "channel_mix": {"editorial_playlist_pitch": 0.20, "paid_radio": 0.30,
                        "paid_social": 0.25, "music_video": 0.25},
        "note": "Commit flagship budget; pursue editorial playlists and radio adds."},
    2: {"action": "Targeted campaign", "budget": 45_000,
        "channel_mix": {"algorithmic_playlist_pitch": 0.35, "paid_social": 0.45,
                        "paid_radio": 0.20, "music_video": 0.00},
        "note": "Fund algorithmic playlisting + paid social; hold radio in reserve."},
    1: {"action": "Hold / organic only", "budget": 5_000,
        "channel_mix": {"organic_social": 1.00},
        "note": "Don't commit paid budget yet; consider a remix or a different lead single."},
}

# Expected gross revenue if a track lands in each tier.
DEFAULT_TIER_REVENUE = {3: 450_000, 2: 90_000, 1: 12_000}

# Min probability of reaching >= Tier B before committing a campaign.
DEFAULT_COMMIT_THRESHOLD = 0.35

# Promo response model:
#
# We model the realised fraction of expected revenue as:
#
#   realised(budget) = PROMO_FLOOR + (1 - PROMO_FLOOR) * (1 - e^(-budget/SCALE))
#
#   PROMO_FLOOR : fraction of potential you capture with $0 paid promo (organic).
#   PROMO_SCALE : spend ($) at which you've captured ~63% of the promo-driven gain.
#
# This gives an interior optimal budget (spend too little and you leave upside on
# the table vs. spend too much and extra dollars stop paying for themselves), which
# is the whole point of the tool. The curve is an assumption which is stated
# plainly in the app's limitations.

PROMO_FLOOR = 0.45
PROMO_SCALE = 70_000

# Logistic regression inverse regularisation strength exposed to the user.
DEFAULT_C = 1.0
