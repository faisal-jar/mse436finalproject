import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.predict import load_model, predict_track
from src import decision_engine as de
from src.train_model import build_pipeline
from src.data_prep import build_training_frame

st.set_page_config(page_title="Chart Compass IDSS", layout="wide")

# Model loading / live retrain

@st.cache_resource
def get_base_model():
    return load_model()


@st.cache_data
def get_sample_frame():
    return build_training_frame(pd.read_csv(config.active_training_csv()))


@st.cache_resource
def retrain(C: float):
    frame = get_sample_frame()
    X, y = frame[config.ALL_FEATURES], frame[config.TARGET].astype(int)
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    pipe = build_pipeline(C=C)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    acc = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy").mean()
    pipe.fit(X, y)
    return pipe, float(acc)


def predict_with(pipe, candidate):
    row = {c: candidate[c] for c in config.ALL_FEATURES}
    proba = pipe.predict_proba(pd.DataFrame([row]))[0]
    prob = {int(c): float(p) for c, p in zip(pipe.classes_, proba)}
    for c in config.CLASS_ORDER:
        prob.setdefault(c, 0.0)
    pred = int(max(prob, key=prob.get))
    return {"pred_class": pred, "tier": config.TIERS[pred],
            "prob_by_class": prob, "p_at_least_B": prob[2] + prob[3]}


base_model, meta = get_base_model()
ranges = meta["feature_ranges"]


def audio_slider(feat):
    """Render a slider and return the value in model units."""
    ui = config.FEATURE_UI[feat]
    kind = ui["kind"]
    help_txt = f"{ui['help']}  (left = {ui['low']} · right = {ui['high']})"
    median_disp = config.model_to_display(feat, ranges[feat]["median"])

    if kind == "pct":
        disp = st.slider(ui["label"], 0, 10, int(round(median_disp)),
                         help=help_txt)
    elif kind == "loudness":
        disp = st.slider(ui["label"], 0, 100, int(round(median_disp)),
                         help=help_txt)
    elif kind == "bpm":
        disp = st.slider(ui["label"], ui["min"], ui["max"],
                         int(round(median_disp)), step=ui["step"], help=help_txt)
    elif kind == "minutes":
        disp = st.slider(ui["label"], float(ui["min"]), float(ui["max"]),
                         round(float(median_disp), 1), step=float(ui["step"]),
                         help=help_txt)
    else:  # raw
        disp = st.slider(ui["label"], ui["min"], ui["max"],
                         int(round(median_disp)), step=ui["step"], help=help_txt)
    return config.display_to_model(feat, disp)

# Header

st.title("Chart Compass")
st.caption(
    "Decision support for pre-release single strategy.")

model_vals = {}

# Sidebarr (collapsible sections)

with st.sidebar:

    with st.expander("Artist Track Record", expanded=True):
        st.caption("The biggest driver of how a song charts.")
        is_new = st.checkbox("New / unproven artist (no chart history)")
        if is_new:
            prior_song_count = prior_top10_count = 0
            prior_best_peak = prior_avg_peak = 101
        else:
            prior_song_count = st.slider(
                "Songs that have charted before", 1, 30, 3,
                help="How many past songs made the Billboard Hot 100.")
            prior_top10_count = st.slider(
                "Of those that charted, how many were Top 10 hits", 0, prior_song_count,
                min(1, prior_song_count))
            prior_best_peak = st.slider(
                "Best chart position ever reached", 1, 100, 20,
                help="1 = a number-one hit. Lower is better.")
            prior_avg_peak = st.slider(
                "Typical chart position", 1, 100, 45,
                help="Where their songs usually land, on average.")
        popularity = st.slider(
            "Current momentum / fanbase", 0, 100, 55,
            help="0 = unknown/emerging, 100 = major established star.")

    with st.expander("The Song", expanded=True):
        genre = st.selectbox("Genre", config.GENRE_CHOICES,
                             index=config.GENRE_CHOICES.index("pop"))
        release_month_name = st.select_slider(
            "Planned release month", options=config.MONTHS, value="Jun",
            help="Summer and Q4 run slightly hotter, but the effect is small.")
        release_month = config.MONTHS.index(release_month_name) + 1

        st.markdown("**How does it sound?**")
        for feat in ["danceability", "energy", "valence", "acousticness",
                     "tempo", "spotify_track_duration_ms"]:
            model_vals[feat] = audio_slider(feat)

        st.caption("Fine detail (optional)")
        for feat in ["speechiness", "instrumentalness", "liveness"]:
            model_vals[feat] = audio_slider(feat)

        # Negligible features that are fixed at their typical value & not shown:
        for feat in ["loudness", "key", "time_signature"]:
            model_vals[feat] = ranges[feat]["median"]

    with st.expander("Your Call", expanded=True):
        st.markdown("**Risk Comfort**")
        ct_pct = st.slider(
            "Confidence needed before funding a campaign", 5, 95, 35, step=5,
            format="%d%%",
            help="The bar the song must clear to get a 'commit' recommendation. "
                 "If its chance of reaching the top 40 is below this number, the "
                 "IDSS says hold. Set it higher to only fund safe bets, "
                 "lower to take more of a risk.")
        commit_threshold = ct_pct / 100.0
        st.markdown("**Budget**")
        budget_mode = st.radio(
            "How much to spend", ["Recommended for the tier", "Set my own"],
            help="'Recommended' uses a standard campaign size per tier (full "
                 "about 120k, targeted about 45k, hold about 5k USD). 'Set my "
                 "own' tests any number.")
        budget_override = None
        if budget_mode == "Set my own":
            budget_override = st.number_input(
                "Committed budget (USD)", 0, 1_000_000, 45_000, step=5_000)

    with st.expander("What Each Outcome is Worth (USD)"):
        st.caption("Revenue if the single lands in each tier. Drives the "
                   "expected-value maths.")
        rev_A = st.number_input("Tier A, Hit contender (top 5)", 0, 5_000_000,
                                config.DEFAULT_TIER_REVENUE[3], step=10_000)
        rev_B = st.number_input("Tier B, Charting (6 to 40)", 0, 2_000_000,
                                config.DEFAULT_TIER_REVENUE[2], step=10_000)
        rev_C = st.number_input("Tier C, Long shot (41 to 100)", 0, 500_000,
                                config.DEFAULT_TIER_REVENUE[1], step=5_000)
    tier_revenue = {3: rev_A, 2: rev_B, 1: rev_C}

    with st.expander("Advanced: Model Tuning"):
        st.caption("How closely the model fits the fed historical data. Lower is "
                   "simpler and more cautious vs. higher follows the data harder "
                   "and can overfit.")
        C = st.select_slider("Model fit strength (regularisation C)",
                             options=[0.01, 0.1, 0.5, 1.0, 2.0, 5.0],
                             value=config.DEFAULT_C)
        use_retrain = st.checkbox("Retrain the model at this setting", value=False)

# Assemble candidate

candidate = dict(model_vals)
candidate.update({
    "spotify_track_popularity": popularity,
    "prior_song_count": prior_song_count, "prior_top10_count": prior_top10_count,
    "prior_best_peak": prior_best_peak, "prior_avg_peak": prior_avg_peak,
    "grouped_genre": genre, "release_month": release_month,
})

if use_retrain:
    model, cv_acc = retrain(C)
    pred = predict_with(model, candidate)
    acc_note = f"live-retrained, C={C}, CV acc {cv_acc:.1%}"
else:
    pred = predict_with(base_model, candidate)
    acc_note = f"baseline, CV acc {meta['cv_accuracy_mean']:.1%}"

plan = de.build_plan(pred, commit_threshold=commit_threshold,
                     tier_revenue=tier_revenue, budget_override=budget_override)

# Main Area

tier = pred["tier"]
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Predicted Chart Tier", f"Tier {tier['code']}")
    st.caption(f"{tier['name']}, peak {tier['peak_range']}")
with c2:
    st.metric("Chance of Top-40 (>= Tier B)", f"{pred['p_at_least_B']:.0%}")
    st.caption(f"you need {commit_threshold:.0%}")
with c3:
    st.metric("Recommendation", plan["verdict"].split(" - ")[0])
    st.caption(acc_note)

with st.expander("How to read this"):
    st.markdown("""
- **Chart Tier:** where the model thinks the song peaks. A = Top 5, B = 6 to 40,
  C = 41 to 100.
- **Chance of Top-40:** how likely it reaches Tier B or better. If it clears your
  confidence slider, funding is worth considering.
- **Expected Value:** what the release is worth on average. More promo spend
  raises the revenue, but with diminishing returns, so there is an optimal spend. Expected net is that revenue minus your spend.
- **Optimal Spend & Sensitivity Curve:** the budget where expected net peaks.
  Below it you leave money on the table vs. above it each extra dollar returns less
  than it costs.
""")

left, right = st.columns(2)

with left:
    st.subheader("Chart Tier Probability")
    order = config.CLASS_ORDER
    fig = go.Figure(go.Bar(
        x=[f"{config.TIERS[c]['code']} · {config.TIERS[c]['name']}" for c in order],
        y=[pred["prob_by_class"][c] for c in order],
        marker_color=[config.TIERS[c]["color"] for c in order],
        text=[f"{pred['prob_by_class'][c]:.0%}" for c in order],
        textposition="outside"))
    fig.update_layout(yaxis_tickformat=".0%", yaxis_range=[0, 1.1],
                      height=300, margin=dict(t=10, b=10))
    st.plotly_chart(fig, width='stretch')

    st.subheader("Recommended Action")
    st.markdown(f"### {plan['action']}\n{plan['note']}")
    st.markdown(f"**Committed Budget:** {plan['committed_budget']:,} USD")
    ch = plan["channel_dollars"]
    ch_df = pd.DataFrame({"Channel": [k.replace('_', ' ').title() for k in ch],
                          "Allocation ($)": list(ch.values())})
    st.dataframe(ch_df, hide_index=True, width='stretch')

with right:
    st.subheader("Expected Value of This Decision")
    econ = plan["economics"]
    m1, m2, m3 = st.columns(3)
    m1.metric("Expected Return", f"${econ['expected_return']:,.0f}",
              help="Average revenue you capture at this spend.")
    m2.metric("Expected Net", f"${econ['expected_net']:,.0f}",
              help="Expected return minus the budget you commit.")
    opt_net = de.expected_value(pred["prob_by_class"], plan["optimal_budget"],
                                tier_revenue)["expected_net"]
    m3.metric("Net at Optimal Spend", f"${opt_net:,.0f}",
              help="The highest expected net you can reach, if you spent the "
                   "optimal amount.")

    contrib = [pred["prob_by_class"][c] * tier_revenue[c] for c in order]
    fig2 = go.Figure(go.Bar(
        x=[config.TIERS[c]["name"] for c in order], y=contrib,
        marker_color=[config.TIERS[c]["color"] for c in order]))
    fig2.update_layout(height=260, margin=dict(t=10, b=10),
                       yaxis_title="$ expected", showlegend=False)
    st.plotly_chart(fig2, width='stretch')
    st.caption("Each bar is that tier's chance times its value, before promo. "
               "They sum to the song's baseline potential.")

    opt = plan["optimal_budget"]
    st.info(f"**Optimal Spend: {opt:,.0f} USD.** This maximises "
            f"expected net dollars. The current plan spends "
            f"{plan['committed_budget']:,} USD.")

st.divider()
st.subheader("What is the right amount to spend on this track?")
st.caption("Expected net value as budget rises. It climbs while promo still pays "
           "off, peaks at the best spend, then falls with diminishing returns")
budgets = list(range(0, 200_001, 5_000))
nets = [de.expected_value(pred["prob_by_class"], b, tier_revenue)["expected_net"]
        for b in budgets]
opt = plan["optimal_budget"]
fig3 = go.Figure()
fig3.add_scatter(x=budgets, y=nets, mode="lines", name="Expected Net Value",
                 line=dict(width=3))
fig3.add_vline(x=opt, line_dash="dash", line_color="#1DB954",
               annotation_text="best spend")
fig3.add_vline(x=plan["committed_budget"], line_dash="dot", line_color=tier["color"],
               annotation_text="current plan")
fig3.update_layout(height=320, xaxis_title="Committed Budget ($)",
                   yaxis_title="Expected Net Value ($)", margin=dict(t=10))
st.plotly_chart(fig3, width='stretch')

with st.expander("Model, Data & Limitations"):
    st.markdown(f"""
- **Model:** multinomial logistic regression, about {meta['cv_accuracy_mean']:.0%}
  5-fold CV accuracy on 3 chart tiers.
- **Data:** Billboard Hot 100 with Spotify audio features (Kaggle).
- **Momentum Note:** the momentum input partly reflects realised success. For a
  true cold start, estimate it from the artist's existing catalogue.
- **Promo Spend Assumption** We model spend as lifting realised revenue with diminishing returns. 
  The shape is reasonable but not calibrated, so treat the optimal figure as a planning anchor.
""")